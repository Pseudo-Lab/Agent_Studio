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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tts/audio/{filename}")
async def get_tts_audio(filename: str):
    """Serve TTS-generated audio files."""
    import os
    
    # Security: prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # TTS output directory from env or default
    project_root = Path(__file__).resolve().parents[3]
    env_dir = os.getenv("TTS_OUTPUT_DIR")
    
    if env_dir:
        tts_dir = Path(env_dir)
        # If relative path, resolve against project root
        if not tts_dir.is_absolute():
            tts_dir = project_root / tts_dir
    else:
        tts_dir = project_root / "screenshots" / "tts_output"
        
    audio_path = tts_dir / filename
    
    # Fallback to legacy path (screenshots/tts_output) if file not found in main dir
    # This supports cases where files were generated in the old location
    if not audio_path.exists():
        legacy_dir = project_root / "screenshots" / "tts_output"
        legacy_path = legacy_dir / filename
        if legacy_path.exists():
            audio_path = legacy_path
            
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")
    
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=3600"},
    )
