import yt_dlp
import whisper
import os
import subprocess
import re
import requests
from pathlib import Path

# =============================
# CONFIG
# =============================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "ministral-3:14b"
LLM_TIMEOUT = 600  # ลดจาก 300 กันค้าง

# =============================
# Whisper cache check
# =============================
def whisper_model_exists(model_name="medium"):
    cache_dir = Path.home() / ".cache" / "whisper"
    return (cache_dir / f"{model_name}.pt").exists()


# =============================
# 1. Download audio
# =============================
def download_audio(video_url, out_dir="downloads"):
    os.makedirs(out_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "320",
        }],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        filename = ydl.prepare_filename(info)

    return os.path.splitext(filename)[0] + ".wav"


# =============================
# 2. Remove silence
# =============================
def remove_silence(wav):
    out = wav.replace(".wav", "_nosilence.wav")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", wav,
        "-af",
        "silenceremove=start_periods=1:start_threshold=-40dB:"
        "stop_periods=1:stop_threshold=-40dB",
        out
    ], check=True)

    return out


# =============================
# 3. Separate vocals
# =============================
def separate_vocals(wav, out_dir="separated"):
    subprocess.run([
        "demucs", "-n", "htdemucs",
        wav, "-o", out_dir
    ], check=True)

    name = os.path.splitext(os.path.basename(wav))[0]
    return os.path.join(out_dir, "htdemucs", name, "vocals.wav")


# =============================
# 4. Clean whisper output
# =============================
def clean_whisper(text):
    text = re.sub(r"(เมื่อ){3,}", "เมื่อ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =============================
# 5. Whisper transcription
# =============================
def transcribe(vocals):
    if whisper_model_exists("medium"):
        print("✅ Whisper medium model found (skip download)")
    else:
        print("⬇️ Downloading Whisper medium model")

    model = whisper.load_model("medium")

    result = model.transcribe(
        vocals,
        language="th",
        task="transcribe",
        temperature=0,
        condition_on_previous_text=False,
        no_speech_threshold=0.4,
        fp16=False
    )

    return clean_whisper(result["text"])


# =============================
# 6. Detect bad lines
# =============================
def looks_bad(line):
    return (
        re.search(r"(เมื่อ){2,}", line) or
        re.search(r"\b(\w+)\s+\1\b", line) or
        len(line) > 120
    )


# =============================
# 7. Call Ollama safely
# =============================
def call_ollama(prompt):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": 0,
                "stream": False
            },
            timeout=LLM_TIMEOUT
        )
        return r.json()["response"].strip()
    except Exception as e:
        print("⚠️ LLM skipped:", e)
        return None


# =============================
# 8. LLM correction (SAFE)
# =============================
def llm_correct(text):
    lines = [l.strip() for l in re.split(r"[。\n]", text) if l.strip()]
    fixed = []

    for line in lines:
        if not looks_bad(line):
            fixed.append(line)
            continue

        prompt = f"""
แก้เฉพาะคำที่สะกดผิดจากการฟังเสียง
- ห้ามแต่งเพิ่ม
- ห้ามเปลี่ยนความหมาย
- ถ้าไม่แน่ใจ ให้ใช้ข้อความเดิม

ข้อความ:
{line}
"""

        out = call_ollama(prompt)

        if not out:
            fixed.append(line)
        elif len(out) > len(line) * 1.1:
            fixed.append(line)
        else:
            fixed.append(out)

    return "\n".join(fixed)


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    YOUTUBE_URL = "https://www.youtube.com/watch?v=yYnn549CIo4"

    audio = download_audio(YOUTUBE_URL)
    audio = remove_silence(audio)
    vocals = separate_vocals(audio)

    whisper_text = transcribe(vocals)
    print("\n--- WHISPER ---\n", whisper_text)

    final_lyrics = llm_correct(whisper_text)

    print("\n🎶 --- FINAL LYRICS (SAFE) ---\n")
    print(final_lyrics)
