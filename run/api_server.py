"""
FastAPI wrapper for Qwen3-ASR.

Usage:
    python run/api_server.py --model_path ./Qwen3-ASR-1.7B --port 8000
    python run/api_server.py --model_path ./qwen3-asr-finetuning-out/checkpoint-400 --port 8000

Endpoints:
    POST /transcribe         — transcribe an audio file (multipart/form-data)
    POST /transcribe/stream  — SSE streaming transcription
    GET  /health             — health check
"""

import argparse
import io
import logging
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from qwen_asr import Qwen3ASRModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-ASR API", version="1.0.0")

# Module-level model (loaded once, reused across requests)
_model: Optional[Qwen3ASRModel] = None


def load_audio(file_bytes: bytes) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def transcription_to_dict(result) -> dict:
    out = {"language": result.language, "text": result.text}
    if result.time_stamps is not None:
        ts = result.time_stamps
        out["time_stamps"] = {
            "word-level": getattr(ts, "word_level", None),
            "char-level": getattr(ts, "char_level", None),
        }
    return out


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    context: Optional[str] = "",
    language: Optional[str] = None,
    return_time_stamps: bool = False,
) -> dict:
    """
    Transcribe a single audio file.

    Form fields:
        audio           — audio file (wav/mp3/flac, multipart)
        context         — optional context string (default: "")
        language        — optional language hint, e.g. "Chinese" (default: auto)
        return_time_stamps — if "true", include forced-aligner timestamps (default: false)
    """
    if _model is None:
        raise HTTPException(503, "model not loaded")

    try:
        wav_bytes = await audio.read()
        wav, sr = load_audio(wav_bytes)

        result = _model.transcribe(
            audio=(wav, sr),
            context=context or "",
            language=language,
            return_time_stamps=return_time_stamps,
        )[0]

        return transcription_to_dict(result)

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("transcribe error")
        raise HTTPException(500, str(e))


@app.post("/transcribe/batch")
async def transcribe_batch(
    audios: list[UploadFile] = File(...),
    contexts: Optional[str] = None,
    languages: Optional[str] = None,
) -> dict:
    """
    Batch transcribe multiple audio files in one request.

    Form fields:
        audios  — list of audio files (at least 1)
        contexts — JSON string list of context strings, one per audio (optional)
        languages — JSON string list of language hints, one per audio (optional)
    """
    if _model is None:
        raise HTTPException(503, "model not loaded")

    try:
        audio_data = []
        for f in audios:
            wav_bytes = await f.read()
            wav, sr = load_audio(wav_bytes)
            audio_data.append((wav, sr))

        ctx_list = None
        if contexts:
            import json as _json
            ctx_list = _json.loads(contexts)

        lang_list = None
        if languages:
            import json as _json
            lang_list = _json.loads(languages)

        results = _model.transcribe(
            audio=audio_data,
            context=ctx_list or "",
            language=lang_list,
        )

        return {"results": [transcription_to_dict(r) for r in results]}

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("batch transcribe error")
        raise HTTPException(500, str(e))


@app.get("/health")
async def health() -> dict:
    if _model is None:
        return {"status": "loading"}
    return {"status": "ok", "backend": _model.backend}


@app.on_event("startup")
def startup():
    pass  # Model is loaded via --model_path CLI arg before uvicorn starts


def create_app(model_path: str, port: int, max_new_tokens: int = 512) -> FastAPI:
    """
    Load model and configure the app.
    Called by CLI entry point below.
    """
    global _model
    logger.info(f"Loading model from {model_path} ...")
    _model = Qwen3ASRModel.from_pretrained(
        pretrained_model_name_or_path=model_path,
        max_new_tokens=max_new_tokens,
    )
    logger.info(f"Model loaded. Backend: {_model.backend}")
    return app


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR FastAPI server")
    parser.add_argument("--model_path", required=True, help="Path to Qwen3-ASR model")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max tokens to generate per transcript",
    )
    args = parser.parse_args()

    create_app(model_path=args.model_path, port=args.port, max_new_tokens=args.max_new_tokens)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
