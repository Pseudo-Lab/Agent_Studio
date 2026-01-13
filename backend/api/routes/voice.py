"""Voice (TTS/STT) endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()


@router.post("/stt/transcribe")
async def stt_transcribe(file: UploadFile = File(...)):
    """Transcribe audio to text."""
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    
    try:
        from kiosk_agent.voice import transcribe_audio_content
        text = transcribe_audio_content(content, language_code="ko-KR")
        return JSONResponse({"text": text})
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"STT module not available. Install google-cloud-speech: {e}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tts/audio/{filename}")
async def get_tts_audio(filename: str):
    """Serve TTS-generated audio files."""
    import os
    
    # Security: prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # TTS output directory from env or default
    env_dir = os.getenv("TTS_OUTPUT_DIR")
    if env_dir:
        tts_dir = Path(env_dir)
    else:
        project_root = Path(__file__).resolve().parents[3]
        tts_dir = project_root / "screenshots" / "tts_output"
    audio_path = tts_dir / filename
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")
    
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=3600"},
    )
