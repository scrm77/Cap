#!/usr/bin/env python3
"""
cap-transcriber — внешний сервис авто-транскрипции и саммари для self-hosted Cap.
Обходит сломанный встроенный workflow Cap: поллит БД, для каждого нового видео
делает Deepgram -> WebVTT в MinIO -> OpenAI summary/chapters -> запись в БД.

Env (все обязательны кроме помеченных):
  MYSQL_HOST=mysql  MYSQL_PORT=3306  MYSQL_USER=cap  MYSQL_PASSWORD=...  MYSQL_DB=cap
  MINIO_ENDPOINT=http://minio:9000  MINIO_ACCESS_KEY=...  MINIO_SECRET_KEY=...  MINIO_BUCKET=cap
  DEEPGRAM_API_KEY=...   OPENAI_API_KEY=...
  ENABLE_SUMMARY=true            # опц.
  POLL_INTERVAL=60               # опц., сек
  STUCK_MINUTES=10               # опц., сколько PROCESSING считать брошенным
  MAX_VIDEO_MB=2000              # опц., потолок размера
  DRY_RUN=false                  # опц., только логировать кандидатов
"""
import os, sys, time, json, logging, tempfile, urllib.request, urllib.error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cap-transcriber")

def env(k, default=None, required=False):
    v = os.environ.get(k, default)
    if required and not v:
        log.error("missing required env %s", k); sys.exit(1)
    return v

CFG = dict(
    mysql_host=env("MYSQL_HOST", "mysql"),
    mysql_port=int(env("MYSQL_PORT", "3306")),
    mysql_user=env("MYSQL_USER", "cap"),
    mysql_pw=env("MYSQL_PASSWORD", required=True),
    mysql_db=env("MYSQL_DB", "cap"),
    minio_endpoint=env("MINIO_ENDPOINT", "http://minio:9000"),
    minio_key=env("MINIO_ACCESS_KEY", required=True),
    minio_secret=env("MINIO_SECRET_KEY", required=True),
    bucket=env("MINIO_BUCKET", "cap"),
    deepgram=env("DEEPGRAM_API_KEY", required=True),
    openai=env("OPENAI_API_KEY", required=True),
    enable_summary=env("ENABLE_SUMMARY", "true").lower() == "true",
    poll=int(env("POLL_INTERVAL", "60")),
    stuck_min=int(env("STUCK_MINUTES", "10")),
    max_mb=int(env("MAX_VIDEO_MB", "2000")),
    dry_run=env("DRY_RUN", "false").lower() == "true",
)

# ---- lazy deps (pymysql, boto3) — импортируем внутри, чтобы dry-run-скелет падал понятно ----
def get_db():
    import pymysql
    return pymysql.connect(
        host=CFG["mysql_host"], port=CFG["mysql_port"], user=CFG["mysql_user"],
        password=CFG["mysql_pw"], database=CFG["mysql_db"], charset="utf8mb4",
        autocommit=True, cursorclass=pymysql.cursors.DictCursor,
    )

def get_s3():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=CFG["minio_endpoint"],
        aws_access_key_id=CFG["minio_key"], aws_secret_access_key=CFG["minio_secret"],
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
        region_name="us-east-1",
    )

# ---------------- candidate selection ----------------
CANDIDATE_SQL = """
SELECT id, ownerId, transcriptionStatus, createdAt
FROM videos
WHERE (
    transcriptionStatus IS NULL
    OR (transcriptionStatus = 'PROCESSING'
        AND updatedAt < (NOW() - INTERVAL %s MINUTE))
)
ORDER BY createdAt DESC
LIMIT 5
"""
# fallback без updatedAt (если колонки нет — узнаем на Э1)
CANDIDATE_SQL_NOUPD = """
SELECT id, ownerId, transcriptionStatus, createdAt
FROM videos
WHERE transcriptionStatus IS NULL
ORDER BY createdAt DESC
LIMIT 5
"""

def fetch_candidates(db):
    with db.cursor() as cur:
        try:
            cur.execute(CANDIDATE_SQL, (CFG["stuck_min"],))
        except Exception as e:
            log.warning("candidate query w/ updatedAt failed (%s), fallback", e)
            cur.execute(CANDIDATE_SQL_NOUPD)
        return cur.fetchall()

def s3_key_video(owner, vid): return f"{owner}/{vid}/result.mp4"
def s3_key_vtt(owner, vid):   return f"{owner}/{vid}/transcription.vtt"

def s3_head(s3, key):
    try:
        return s3.head_object(Bucket=CFG["bucket"], Key=key)
    except Exception:
        return None

# ---------------- one tick (dry-run aware) ----------------
def tick():
    db = get_db()
    s3 = get_s3()
    cands = fetch_candidates(db)
    if not cands:
        log.info("no candidates"); return
    log.info("candidates: %d", len(cands))
    for v in cands:
        vid, owner = v["id"], v["ownerId"]
        head = s3_head(s3, s3_key_video(owner, vid))
        if not head:
            log.info("  [%s] result.mp4 not in S3 yet -> skip (upload in progress)", vid); continue
        size_mb = head["ContentLength"] / 1e6
        log.info("  [%s] owner=%s status=%s size=%.1fMB -> WOULD PROCESS%s",
                 vid, owner, v["transcriptionStatus"], size_mb,
                 " (dry-run)" if CFG["dry_run"] else "")
        if CFG["dry_run"]:
            continue
        try:
            process_video(db, s3, owner, vid)
        except Exception as e:
            log.exception("  [%s] processing failed: %s", vid, e)
            mark_status(db, vid, "ERROR")
    db.close()

# ---------------- Deepgram ----------------
def deepgram_transcribe(audio_bytes, content_type="video/mp4"):
    url = ("https://api.deepgram.com/v1/listen"
           "?model=nova-3&smart_format=true&utterances=true&punctuate=true&detect_language=true")
    req = urllib.request.Request(url, data=audio_bytes, method="POST", headers={
        "Authorization": "Token " + CFG["deepgram"],
        "Content-Type": content_type,
    })
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

def fmt_ts(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"

def _first_alt_text(res):
    chans = res.get("channels") or []
    if not chans:
        return ""
    alts = chans[0].get("alternatives") or []
    if not alts:
        return ""
    return (alts[0].get("transcript") or "").strip()

def build_vtt(dg):
    res = dg.get("results", {})
    utt = res.get("utterances") or []
    lines = ["WEBVTT", ""]
    if utt:
        for i, u in enumerate(utt, 1):
            lines += [str(i), f"{fmt_ts(u['start'])} --> {fmt_ts(u['end'])}", u["transcript"].strip(), ""]
        full_text = " ".join(u["transcript"].strip() for u in utt)
    else:
        txt = _first_alt_text(res)
        if not txt:
            return None, ""   # пусто -> NO_AUDIO
        dur = dg.get("metadata", {}).get("duration", 0) or 0
        lines += ["1", f"00:00:00.000 --> {fmt_ts(dur)}", txt, ""]
        full_text = txt
    return "\n".join(lines), full_text

# ---------------- OpenAI summary/chapters ----------------
def openai_summarize(transcript_vtt):
    prompt = (
        "You are given a video transcript (WebVTT). Produce a JSON object with:\n"
        '- "title": short video title (string, same language as transcript)\n'
        '- "summary": 2-4 sentence summary in markdown (same language)\n'
        '- "chapters": array of {"title": string, "start": number-seconds} for key moments '
        "(1-8 items, start>=0, ascending)\n"
        "Return ONLY valid JSON.\n\nTranscript:\n" + transcript_vtt
    )
    body = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, headers={
        "Authorization": "Bearer " + CFG["openai"], "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    return json.loads(out["choices"][0]["message"]["content"])

# ---------------- DB writes ----------------
def mark_status(db, vid, status):
    with db.cursor() as cur:
        cur.execute("UPDATE videos SET transcriptionStatus=%s WHERE id=%s", (status, vid))

def save_metadata(db, vid, ai):
    with db.cursor() as cur:
        cur.execute("SELECT metadata FROM videos WHERE id=%s", (vid,))
        row = cur.fetchone()
    meta = {}
    if row and row.get("metadata"):
        try:
            meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"])
        except Exception:
            meta = {}
    meta.update({
        "aiTitle": ai.get("title") or meta.get("aiTitle"),
        "summary": ai.get("summary") or meta.get("summary"),
        "chapters": ai.get("chapters") or meta.get("chapters"),
        "aiGenerationStatus": "COMPLETE",
    })
    with db.cursor() as cur:
        cur.execute("UPDATE videos SET metadata=%s WHERE id=%s",
                    (json.dumps(meta, ensure_ascii=False), vid))

# ---------------- process one video ----------------
def process_video(db, s3, owner, vid):
    # claim: пометить PROCESSING, чтобы другой тик не взял
    mark_status(db, vid, "PROCESSING")
    log.info("  [%s] downloading mp4...", vid)
    obj = s3.get_object(Bucket=CFG["bucket"], Key=s3_key_video(owner, vid))
    audio = obj["Body"].read()
    if len(audio) / 1e6 > CFG["max_mb"]:
        log.warning("  [%s] too big (%.0fMB) -> SKIPPED", vid, len(audio)/1e6)
        mark_status(db, vid, "SKIPPED"); return

    log.info("  [%s] deepgram...", vid)
    dg = deepgram_transcribe(audio)
    vtt, full_text = build_vtt(dg)
    if not vtt:
        log.info("  [%s] empty transcript -> NO_AUDIO", vid)
        mark_status(db, vid, "NO_AUDIO"); return

    s3.put_object(Bucket=CFG["bucket"], Key=s3_key_vtt(owner, vid),
                  Body=vtt.encode("utf-8"), ContentType="text/vtt")
    mark_status(db, vid, "COMPLETE")
    log.info("  [%s] transcript COMPLETE (%d chars)", vid, len(full_text))

    if CFG["enable_summary"]:
        try:
            log.info("  [%s] openai summary...", vid)
            ai = openai_summarize(vtt)
            save_metadata(db, vid, ai)
            log.info("  [%s] summary saved: title=%r chapters=%d",
                     vid, ai.get("title"), len(ai.get("chapters") or []))
        except Exception as e:
            log.exception("  [%s] summary failed (transcript still ok): %s", vid, e)

def main():
    log.info("cap-transcriber start | dry_run=%s summary=%s poll=%ss",
             CFG["dry_run"], CFG["enable_summary"], CFG["poll"])
    while True:
        try:
            tick()
        except Exception as e:
            log.exception("tick error: %s", e)
        time.sleep(CFG["poll"])

if __name__ == "__main__":
    if "--once" in sys.argv:
        tick()
    else:
        main()
