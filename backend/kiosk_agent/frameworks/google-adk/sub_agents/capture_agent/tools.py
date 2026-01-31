"""Screenshot capture tool for the ADK capture agent.

Reuses the existing AndroidScreenshotter from kiosk_agent.core to capture
screenshots via ADB, compresses them to JPEG, and stores in session state
for downstream agents (analysis_agent, status_agent) to consume.
"""

import io
import os

from PIL import Image
from google.genai import types
from google.adk.tools import ToolContext

from ...shared_libraries import Status

# ---------------------------------------------------------------------------
# Screenshot compression settings (configurable via environment variables)
# ---------------------------------------------------------------------------
_JPEG_QUALITY = int(os.getenv("SCREENSHOT_JPEG_QUALITY", "75"))
_MAX_DIM = int(os.getenv("SCREENSHOT_MAX_DIM", "1280"))


def _get_screenshotter():
    """Lazily create an AndroidScreenshotter from the existing kiosk_agent core.

    Lazy import avoids circular dependencies and allows the capture tool
    to be registered before ADB is actually available.
    """
    from .....core.perception import AndroidScreenshotter
    from .....config import ScreenshotConfig
    return AndroidScreenshotter(ScreenshotConfig())


def _compress_screenshot(image_path: str) -> bytes:
    """Compress a PNG screenshot to JPEG with optional downscaling.

    Args:
        image_path: Path to the original PNG screenshot file.

    Returns:
        JPEG-encoded bytes.

    The function:
    1. Resizes the image so that max(width, height) <= _MAX_DIM
    2. Converts RGBA/P mode to RGB (JPEG has no alpha channel)
    3. Encodes to JPEG with the configured quality level
    """
    img = Image.open(image_path)

    # Resize if exceeds max dimension
    if _MAX_DIM > 0:
        w, h = img.size
        if max(w, h) > _MAX_DIM:
            ratio = _MAX_DIM / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    # JPEG does not support alpha channel
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return buf.getvalue()


async def android_capture(tool_context: ToolContext) -> dict:
    """Capture a screenshot from the Android kiosk device via ADB.

    Workflow:
    1. ADB screencap -> PNG saved locally
    2. PNG -> JPEG compression (quality & resize configurable)
    3. JPEG bytes stored in ``tool_context.state["current_screenshot"]``
       as a ``types.Part`` so other agents can read it via load_screenshot()
    """
    print("TOOL CALLED: android_capture()")

    try:
        screenshotter = _get_screenshotter()
        screenshot = screenshotter.capture(save=True)

        # Compress PNG -> JPEG with resize
        jpeg_bytes = _compress_screenshot(str(screenshot.path))

        # Store compressed image in session state for downstream agents
        screenshot_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        tool_context.state["current_screenshot"] = screenshot_part

        png_size = os.path.getsize(screenshot.path)
        print(f"  Screenshot: {png_size:,} bytes (PNG) -> {len(jpeg_bytes):,} bytes (JPEG q={_JPEG_QUALITY})")

        return {
            "status": Status.SUCCESS.value,
            "output_path": str(screenshot.path),
            "message": f"Screenshot captured and compressed ({png_size:,} -> {len(jpeg_bytes):,} bytes)."
        }

    except Exception as e:
        return {
            "status": Status.FAIL.value,
            "error": str(e)
        }
