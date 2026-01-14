"""Screenshot capture module for Android devices."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from ..config import ScreenshotConfig


@dataclass
class CapturedScreen:
    """In-memory representation of the captured screenshot."""

    image: Image.Image
    path: Optional[Path]

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.image.size


# Alias for backwards compatibility
ScreenshotResult = CapturedScreen


class AndroidScreenshotter:
    """Captures screenshots from a kiosk-attached Android device via adb."""

    _PNG_SIG = b"\x89PNG\r\n\x1a\n"

    def __init__(self, config: ScreenshotConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, *, save: bool = True) -> CapturedScreen:
        raw = self._capture_exec_out()
        raw = self._strip_to_png(raw)

        try:
            image = Image.open(BytesIO(raw)).convert("RGB")
        except Exception:
            # Retry once (transient capture glitches happen during UI transitions)
            try:
                raw_retry = self._strip_to_png(self._capture_exec_out())
                image = Image.open(BytesIO(raw_retry)).convert("RGB")
                raw = raw_retry
            except Exception as e:
                # Fallback: write screenshot to device and read back via `cat`
                try:
                    raw_fallback = self._strip_to_png(self._capture_via_device_file())
                    image = Image.open(BytesIO(raw_fallback)).convert("RGB")
                    raw = raw_fallback
                except Exception as e2:
                    debug_path = None
                    if save:
                        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
                        debug_path = self.config.output_dir / f"kiosk_screen_failed_{timestamp}.bin"
                        try:
                            debug_path.write_bytes(raw)
                        except Exception:
                            debug_path = None
                    raise RuntimeError(
                        f"Failed to decode screenshot: {e2}. "
                        f"Received {len(raw)} bytes. Device may be in an unsupported state."
                        + (f" (saved raw={debug_path})" if debug_path else "")
                    ) from e
        save_path = None
        if save:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
            save_path = self.config.output_dir / f"kiosk_screen_{timestamp}.png"
            
            image.save(save_path)
            self._cleanup_old_files()
        return CapturedScreen(image=image, path=save_path)

    def _cleanup_old_files(self) -> None:
        all_files = sorted(self.config.output_dir.glob("kiosk_screen_*.png"))
        if len(all_files) <= self.config.keep_last_n:
            return
        for path in all_files[:-self.config.keep_last_n]:
            path.unlink(missing_ok=True)

    def _base_adb_cmd(self) -> list[str]:
        cmd = [self.config.adb_path]
        if self.config.device_id:
            cmd += ["-s", self.config.device_id]
        return cmd

    def _capture_exec_out(self) -> bytes:
        cmd = self._base_adb_cmd() + ["exec-out", "screencap", "-p"]
        proc = subprocess.run(cmd, check=True, capture_output=True)
        raw = proc.stdout or b""
        if len(raw) < 100:
            raise RuntimeError(f"ADB returned empty/invalid screenshot data ({len(raw)} bytes).")
        return raw

    def _capture_via_device_file(self) -> bytes:
        """
        More robust fallback for devices that pollute exec-out stdout or occasionally corrupt streams.
        """
        tmp_path = "/sdcard/__kiosk_screen.png"
        base = self._base_adb_cmd()
        # Write on-device (stderr may contain warnings; stdout should be empty)
        subprocess.run(base + ["shell", "screencap", "-p", tmp_path], check=True, capture_output=True)
        # Read back raw bytes
        proc = subprocess.run(base + ["exec-out", "cat", tmp_path], check=True, capture_output=True)
        raw = proc.stdout or b""
        if len(raw) < 100:
            raise RuntimeError(f"ADB returned empty/invalid screenshot data via file ({len(raw)} bytes).")
        return raw

    def _strip_to_png(self, raw: bytes) -> bytes:
        """
        Some devices print warnings to stdout even when returning PNG bytes.
        We locate the PNG signature and keep bytes from that point.

        IMPORTANT: Do not apply newline/CRLF normalization here; it may corrupt binary PNG data.
        """
        sig_idx = raw.find(self._PNG_SIG)
        if sig_idx == -1:
            head = raw[:160]
            head_text = head.decode(errors="replace").replace("\n", "\\n")
            head_hex = head[:32].hex()
            raise RuntimeError(
                "Screenshot data does not contain a PNG header. "
                f"first32(hex)={head_hex} head(text)='{head_text}' bytes={len(raw)}"
            )
        return raw[sig_idx:]
