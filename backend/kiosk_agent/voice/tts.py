"""
CosyVoice3 TTS integration module for Kiosk Agent.

Provides text-to-speech synthesis using Apple Silicon MLX-optimized CosyVoice3 model.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model

from ..utils import get_logger

logger = get_logger(__name__)


class CosyVoiceTTS:
    """
    Text-to-speech synthesizer using CosyVoice3 with MLX optimization.

    Features:
    - Korean language support with natural prosody
    - Reference audio cloning for voice consistency
    - Apple Silicon Metal acceleration
    - Automatic voice selection based on content
    """

    def __init__(
        self,
        model: str = "mlx-community/Fun-CosyVoice3-0.5B-2512-fp16",
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize CosyVoice TTS synthesizer.

        Args:
            model: Hugging Face model ID for CosyVoice3
            ref_audio_path: Path to reference audio file for voice cloning
            ref_text: Transcript of reference audio for better cloning
            output_dir: Directory to save generated audio files
        """
        self.model = model
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        # Use TTS_OUTPUT_DIR env or default to screenshots/tts_output
        if output_dir:
            self.output_dir = output_dir
        else:
            env_dir = os.getenv("TTS_OUTPUT_DIR")
            if env_dir:
                self.output_dir = Path(env_dir)
            else:
                self.output_dir = Path(__file__).resolve().parents[3] / "screenshots" / "tts_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # NOTE: generate_audio writes to CWD, and os.chdir() is process-global.
        # We serialize synth calls to minimize concurrency hazards.
        self._synthesis_lock = threading.Lock()
        self._cached_model = None
        self._model_loaded = False
        
        # Preload model at startup (서버 시작 시 로드)
        self._preload_model()

    def _preload_model(self):
        """Preload the TTS model to avoid cold-start latency."""
        try:
            logger.info(f"Preloading model: {self.model}...")
            # Disable HuggingFace Hub progress bars for cleaner logging
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            self._cached_model = load_model(model_path=self.model)
            self._model_loaded = True
            logger.info("Model preloaded successfully")
        except Exception as e:
            logger.warning(f"Model preload failed (will load on first use): {e}")
            self._cached_model = None
            self._model_loaded = False

    def synthesize(
        self,
        text: str,
        *,
        file_prefix: str = "agent_response",
        speed: float = 1.0,
        voice: Optional[str] = None,
        ref_audio_override: Optional[str] = None,
        ref_text_override: Optional[str] = None,
    ) -> Path:
        """
        Synthesize speech from text and return path to generated audio file.

        Args:
            text: Text to synthesize
            file_prefix: Prefix for output filename
            speed: Speech speed multiplier (1.0 = normal)
            voice: Optional voice ID override (defaults to auto-selection)
            ref_audio_override: Override default reference audio path
            ref_text_override: Override default reference text

        Returns:
            Path to generated WAV file

        Raises:
            RuntimeError: If synthesis fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text")

        # Use override or default reference
        ref_audio = ref_audio_override or self.ref_audio_path
        ref_text = ref_text_override or self.ref_text

        # Change to output directory for generate_audio (it saves in CWD).
        # Guarded by a lock because os.chdir() is global to the process.
        with self._synthesis_lock:
            original_cwd = Path.cwd()
            os.chdir(self.output_dir)
            try:
                # Ensure model is loaded
                if not self._model_loaded or self._cached_model is None:
                    logger.info("Loading model (first use)...")
                    self._cached_model = load_model(model_path=self.model)
                    self._model_loaded = True
                
                # Generate audio using pre-loaded model (no re-download)
                generate_audio(
                    text=text.strip(),
                    model=self._cached_model,  # Pass cached model, not path
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    file_prefix=file_prefix,
                    speed=speed,
                )

                # Find generated file (generate_audio appends _000, _001, etc.)
                pattern = f"{file_prefix}_*.wav"
                candidates = sorted(self.output_dir.glob(pattern))
                if not candidates:
                    raise RuntimeError(f"No audio file generated matching pattern: {pattern}")

                output_path = candidates[-1]  # Most recent file
                logger.info(f"Generated audio: {output_path} ({output_path.stat().st_size} bytes)")
                return output_path

            except Exception as e:
                raise RuntimeError(f"TTS synthesis failed: {e}") from e
            finally:
                os.chdir(original_cwd)

    def synthesize_to_bytes(self, text: str, **kwargs) -> bytes:
        """
        Synthesize speech and return audio data as bytes.

        Args:
            text: Text to synthesize
            **kwargs: Additional arguments passed to synthesize()

        Returns:
            WAV audio data as bytes
        """
        audio_path = self.synthesize(text, **kwargs)
        return audio_path.read_bytes()

    def cleanup_old_files(self, keep_last_n: int = 10):
        """
        Remove old generated audio files safely, keeping only the most recent N files.

        Args:
            keep_last_n: Number of recent files to keep
        """
        try:
            wav_files = sorted(
                self.output_dir.glob("*.wav"),
                key=lambda p: p.stat().st_mtime
            )
            if len(wav_files) <= keep_last_n:
                return

            for old_file in wav_files[:-keep_last_n]:
                try:
                    # Check if file still exists (race condition protection)
                    if old_file.exists():
                        old_file.unlink()
                        logger.debug(f"Cleaned up old audio: {old_file.name}")
                except PermissionError:
                    logger.warning(f"Cannot delete (in use): {old_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {old_file.name}: {e}")

        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def create_default_tts(agent_root: Optional[Path] = None) -> CosyVoiceTTS:
    """
    Create a TTS instance with default configuration for Kiosk Agent.

    Uses the reference audio file from the tts/ directory if available.
    Uses 안성재(Chef Ahn Sung-jae) voice reference by default.

    Args:
        agent_root: Root directory of kiosk_agent project

    Returns:
        Configured CosyVoiceTTS instance
    """
    if agent_root is None:
        # Try to auto-detect agent root from this module's location
        # backend/kiosk_agent/voice -> backend -> project_root
        agent_root = Path(__file__).resolve().parents[3]

    # 안성재 셰프 / 임짱 레퍼런스 오디오 경로 후보
    ref_audio_candidates = [
        agent_root / "ansungjae_51_62.wav",           # project root
        agent_root / "ansungjae_64_71.wav",
        agent_root / "imzzang_00_15.wav",             # 임짱 레퍼런스
        Path(__file__).resolve().parent / "ansungjae_51_62.wav",  # voice/
        Path(__file__).resolve().parent / "ansungjae_64_71.wav",
        agent_root / "tts" / "ansungjae_51_62.wav",               # tts/
        agent_root / "tts" / "ansungjae_64_71.wav",
    ]

    ref_audio_path = None
    for candidate in ref_audio_candidates:
        if candidate.exists():
            ref_audio_path = str(candidate)
            logger.info(f"Using reference audio: {ref_audio_path}")
            break

    # 안성재 셰프 레퍼런스 텍스트
    ref_text = (
        "얼음판 위에서 거르 걷고 있는 느낌? 근데 부정적인 느낌은 아니에요 "
        "모든 게 다 막 완벽하고 편안함을 느끼고 이렇다면은 더 잘못된 거라고 생각해요"
    )

    output_dir = agent_root / "screenshots" / "tts_output"

    return CosyVoiceTTS(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text if ref_audio_path else None,
        output_dir=output_dir,
    )
