"""
Test script for Qwen3-ASR API.

# 1. Start the server (in one terminal)
python run/api_server.py --model_path /cmy/cmy/Qwen3-ASR/qwen3-asr-finetuning-out-bak/checkpoint-326 --port 8000

# 2. Test it (in another terminal)
python run/test_api.py --audio /path/to/your/audio.wav

# With custom URL (e.g. remote server)
python run/test_api.py --url http://localhost:8000/transcribe --audio /cmy/after_catting/10h/Audio/06/S006T057E000N00165.wav


Usage:
    python test_api.py --url http://localhost:8000/transcribe --audio /path/to/audio.wav
"""

import argparse
import time
import sys


def test_single(url: str, audio_path: str, context: str = "", language: str = ""):
    import requests

    with open(audio_path, "rb") as f:
        files = {"audio": f}
        data = {}
        if context:
            data["context"] = context
        if language:
            data["language"] = language

        print(f"Sending: {audio_path}")
        print(f"URL:     {url}")
        t0 = time.perf_counter()
        resp = requests.post(url, files=files, data=data, timeout=120)
        elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    result = resp.json()
    print(f"\n--- Result ({elapsed:.2f}s) ---")
    print(f"Language: {result.get('language', 'unknown')}")
    print(f"Text:     {result.get('text', '')}")
    if "time_stamps" in result:
        print(f"Time stamps: available (forced aligner mode)")


def test_health(url: str):
    import requests
    resp = requests.get(url + "/health", timeout=5)
    print(f"Health: {resp.json()}")


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--audio", required=True, help="Path to audio file to transcribe")
    parser.add_argument("--context", default="", help="Optional context string")
    parser.add_argument("--language", default="", help="Optional language hint (e.g. Chinese)")
    args = parser.parse_args()

    health_url = args.url.rstrip("/") + "/health"
    try:
        import requests
        r = requests.get(health_url, timeout=5)
        print(f"Health: {r.json()}\n")
    except Exception as e:
        print(f"Warning: health check failed ({e}) — is the server running?\n")

    transcribe_url = args.url.rstrip("/") + "/transcribe"
    test_single(transcribe_url, args.audio, args.context, args.language)


if __name__ == "__main__":
    main()
