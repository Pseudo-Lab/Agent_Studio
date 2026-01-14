"""Speech-to-Text module using Google Cloud Speech API."""

from __future__ import annotations

import base64
import os
import wave
from pathlib import Path
from typing import Optional, Union

import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request

from ..utils import get_logger

logger = get_logger(__name__)


def _get_credentials():
    """Get credentials from service account JSON file."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise ValueError(
            "GOOGLE_APPLICATION_CREDENTIALS 환경변수에 서비스 계정 JSON 파일 경로를 설정해주세요."
        )
    
    # Resolve relative path against project root
    path_obj = Path(credentials_path)
    if not path_obj.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        path_obj = project_root / path_obj
        
    if not path_obj.exists():
        # Try original path as fallback
        if Path(credentials_path).exists():
            path_obj = Path(credentials_path)
        else:
            raise FileNotFoundError(f"서비스 계정 파일을 찾을 수 없습니다: {path_obj} (orig: {credentials_path})")

    credentials = service_account.Credentials.from_service_account_file(
        str(path_obj),
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials


def _call_speech_api(
    audio_content: bytes,
    language_code: str,
    sample_rate_hz: int = None,
    encoding: str = "LINEAR16",
    audio_channel_count: int = 1,
) -> Optional[str]:
    """Call Google Speech-to-Text REST API."""
    credentials = _get_credentials()

    url = "https://speech.googleapis.com/v1/speech:recognize"
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }

    config = {
        "encoding": encoding,
        "languageCode": language_code,
        "audioChannelCount": audio_channel_count,
    }
    if sample_rate_hz:
        config["sampleRateHertz"] = sample_rate_hz

    payload = {
        "config": config,
        "audio": {
            "content": base64.b64encode(audio_content).decode("utf-8")
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        logger.error(f"API error: {response.status_code} - {response.text}")
        return None

    result = response.json()
    if "results" not in result or not result["results"]:
        logger.warning("No speech detected")
        return None

    transcripts = [r["alternatives"][0]["transcript"] for r in result["results"]]
    return " ".join(transcripts)


def transcribe_from_file(
    file_path: Union[str, Path],
    language_code: str = "ko-KR",
) -> Optional[str]:
    """
    Transcribes audio from a WAV file using Google Cloud Speech-to-Text REST API.

    Args:
        file_path: Path to the WAV file
        language_code: BCP-47 language code (default: Korean)

    Returns:
        Transcribed text or None if no speech detected
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    logger.info(f"Reading audio from file: {file_path}")

    # Read WAV file to get sample rate
    with wave.open(str(file_path), "rb") as wf:
        sample_rate_hz = wf.getframerate()
        channels = wf.getnchannels()
        logger.debug(f"Sample rate: {sample_rate_hz}Hz, channels: {channels}")

    # Read audio content
    with open(file_path, "rb") as f:
        audio_content = f.read()

    logger.info("Transcribing with Google STT REST API...")

    transcript = _call_speech_api(
        audio_content=audio_content,
        language_code=language_code,
        sample_rate_hz=sample_rate_hz,
        encoding="LINEAR16",
        audio_channel_count=channels,
    )

    if transcript:
        logger.info(f"Recognized: {transcript}")
    return transcript


def transcribe_audio_content(
    content: bytes,
    language_code: str = "ko-KR",
) -> Optional[str]:
    """
    Transcribes audio content (bytes) directly using Google Cloud Speech-to-Text REST API.
    Attempts WEBM_OPUS first, then falls back to LINEAR16.
    """
    logger.info("Transcribing with Google STT REST API...")

    # Try WEBM_OPUS first (common for web audio)
    transcript = _call_speech_api(
        audio_content=content,
        language_code=language_code,
        sample_rate_hz=48000,
        encoding="WEBM_OPUS",
        audio_channel_count=1,
    )

    if transcript is None:
        logger.debug("WEBM_OPUS failed, retrying with LINEAR16...")
        transcript = _call_speech_api(
            audio_content=content,
            language_code=language_code,
            encoding="LINEAR16",
            audio_channel_count=1,
        )

    if transcript:
        logger.info(f"Recognized: {transcript}")
    return transcript


def transcribe_from_microphone(
    language_code: str = "ko-KR",
    sample_rate_hz: int = 16000,
    timeout_seconds: float = 10.0,
) -> Optional[str]:
    """
    Records audio from the microphone and transcribes it using Google Cloud Speech-to-Text REST API.

    Args:
        language_code: BCP-47 language code (default: Korean)
        sample_rate_hz: Audio sample rate in Hz
        timeout_seconds: Maximum recording duration

    Returns:
        Transcribed text or None if no speech detected
    """
    try:
        import pyaudio
    except ImportError:
        raise ImportError("pyaudio is required. Install it with: pip install pyaudio")

    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    logger.info(f"Recording from microphone... ({timeout_seconds}s)")

    # Record audio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate_hz,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = []
    num_chunks = int(sample_rate_hz / CHUNK * timeout_seconds)

    for _ in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_content = b"".join(frames)
    logger.info("Recording complete. Transcribing with Google STT REST API...")

    transcript = _call_speech_api(
        audio_content=audio_content,
        language_code=language_code,
        sample_rate_hz=sample_rate_hz,
        encoding="LINEAR16",
        audio_channel_count=1,
    )

    if transcript:
        logger.info(f"Recognized: {transcript}")
    return transcript


def transcribe_streaming(
    language_code: str = "ko-KR",
    sample_rate_hz: int = 16000,
) -> Optional[str]:
    """
    Real-time streaming transcription from microphone.
    Note: REST API doesn't support true streaming, so this records in chunks
    and transcribes periodically.

    Args:
        language_code: BCP-47 language code (default: Korean)
        sample_rate_hz: Audio sample rate in Hz

    Returns:
        Final transcribed text or None if no speech detected
    """
    try:
        import pyaudio
    except ImportError:
        raise ImportError("pyaudio is required. Install it with: pip install pyaudio")

    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RECORD_SECONDS = 5  # Transcribe every 5 seconds

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate_hz,
        input=True,
        frames_per_buffer=CHUNK,
    )

    logger.info("Starting real-time speech recognition. Speak now... (Ctrl+C to stop)")

    final_transcript = []

    try:
        while True:
            frames = []
            num_chunks = int(sample_rate_hz / CHUNK * RECORD_SECONDS)

            for _ in range(num_chunks):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            audio_content = b"".join(frames)

            transcript = _call_speech_api(
                audio_content=audio_content,
                language_code=language_code,
                sample_rate_hz=sample_rate_hz,
                encoding="LINEAR16",
                audio_channel_count=1,
            )

            if transcript:
                logger.info(f"Recognized: {transcript}")
                final_transcript.append(transcript)

    except KeyboardInterrupt:
        logger.info("Speech recognition stopped")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    return " ".join(final_transcript) if final_transcript else None


if __name__ == "__main__":
    # Test the STT functionality
    result = transcribe_from_file("test.wav")
    if result:
        logger.info(f"Transcribed: {result}")
    else:
        logger.warning("No speech detected")
