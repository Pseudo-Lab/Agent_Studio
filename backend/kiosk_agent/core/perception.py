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

    def __init__(self, config: ScreenshotConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, *, save: bool = True) -> CapturedScreen:
        cmd = [self.config.adb_path]
        
        if self.config.device_id:
            cmd += ["-s", self.config.device_id]
        cmd += ["exec-out", "screencap", "-p"]
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
        )
        
        # Check if we got valid data
        if not proc.stdout or len(proc.stdout) < 100:
            raise RuntimeError(
                f"ADB returned empty or invalid screenshot data ({len(proc.stdout) if proc.stdout else 0} bytes). "
                "Please check: 1) Device screen is on, 2) Device is unlocked, 3) Run 'adb shell screencap -p > test.png' manually to verify."
            )
        
        try:
            image = Image.open(BytesIO(proc.stdout)).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to decode screenshot: {e}. "
                f"Received {len(proc.stdout)} bytes. Device may be in an unsupported state."
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
