# utils/image_proc.py
import io
from pathlib import Path
from PIL import Image
from src.core import config
# ---------------------------------------------------------------------------
# Step 1 — Azure Document Intelligence OCR
# ---------------------------------------------------------------------------
_ADI_MAX_BYTES = 4 * 1024 * 1024  # Azure DI hard limit: 4 MB


def _to_jpeg_bytes(image_data: "Path | bytes", display_name: str = "") -> tuple[bytes, str]:
    """Return (image_bytes, content_type). Converts HEIC → JPEG and downscales images >4 MB.

    Accepts either a Path (read from disk) or raw bytes (in-memory).
    display_name is used for extension detection and error messages when bytes are passed.
    """
    import io

    # Normalise to (raw, ext, img_or_none)
    if isinstance(image_data, Path):
        ext = image_data.suffix.lower()
        name_for_error = image_data.name
        if ext == ".heic":
            try:
                import pillow_heif
                from PIL import Image
                pillow_heif.register_heif_opener()
                img = Image.open(image_data).convert("RGB")
            except ImportError:
                raise ImportError("HEIC support requires: pip install pillow pillow-heif")
            raw = None
        else:
            raw = image_data.read_bytes()
            img = None
    else:
        raw = image_data
        ext = Path(display_name).suffix.lower() if display_name else ".jpg"
        name_for_error = display_name or "image"
        if ext == ".heic":
            try:
                import pillow_heif
                from PIL import Image
                pillow_heif.register_heif_opener()
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except ImportError:
                raise ImportError("HEIC support requires: pip install pillow pillow-heif")
            raw = None
        else:
            img = None

    if img is None:
        if len(raw) <= _ADI_MAX_BYTES:
            content_type = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png",  ".webp": "image/webp",
                ".tiff": "image/tiff", ".tif": "image/tiff",
                ".bmp": "image/bmp",
            }.get(ext, "image/jpeg")
            return raw, content_type
        # Image is over 4 MB — need to downscale
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                f"{name_for_error} is {len(raw) // 1024 // 1024} MB (Azure DI limit: 4 MB). "
                "Install Pillow to auto-resize: pip install pillow"
            )
        img = Image.open(io.BytesIO(raw)).convert("RGB")

    # Downscale until the JPEG fits under 4 MB (quality 85 first, then halve dimensions)
    quality = 85
    scale = 1.0
    while True:
        w, h = int(img.width * scale), int(img.height * scale)
        resized = img.resize((w, h), Image.LANCZOS) if scale < 1.0 else img
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= _ADI_MAX_BYTES:
            return data, "image/jpeg"
        if quality > 60:
            quality -= 10
        else:
            scale *= 0.75
