#!/usr/bin/env python3
"""
16K Image Upscaler (CLI)

Features
- Upscale to 16K UHD (3840×2160) with aspect-ratio aware modes: fit (default), fill, pad (letterbox).
- Engines:
    1) pillow (default): fast, high-quality Lanczos.
    2) esrgan: optional AI super-resolution via Real-ESRGAN (if installed).
- Auto-select CUDA if available (for ESRGAN).

Usage
------
python upscaler.py input.jpg -o output.png
python upscaler.py input_folder -o upscaled_folder --recursive
python upscaler.py input.png -o out.jpg --mode fill --engine esrgan

Install (basic pillow engine)
-----------------------------
pip install -r requirements.txt

Install (ESRGAN engine)
-----------------------
pip install -r requirements-esrgan.txt

Notes
-----
- ESRGAN engine will attempt to download pretrained weights on first run.
- If your image is not 16:9, choose one of:
  * fit (no crop, sizes to fit within 3840×2160)
  * fill (cover, then center-crop to exactly 3840×2160)
  * pad (letterbox to exactly 3840×2160 with a padding color)
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Tuple, Optional

try:
    from PIL import Image, ImageOps
except Exception as e:
    print("Pillow is required. Install with: pip install pillow", file=sys.stderr)
    raise

# Lazy imports for ESRGAN
def _load_esrgan(scale: int = 4):
    try:
        import torch
        from realesrgan import RealESRGAN
    except Exception as e:
        raise RuntimeError(
            "ESRGAN engine not available. Install with: pip install torch realesrgan"
        ) from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RealESRGAN(device=device, scale=scale)
    # Will download weights the first time if not present
    model.load_weights(f"weights/RealESRGAN_x{scale}plus.pth")
    return model


TARGET_W, TARGET_H = 15360, 8640  # 16K UHD


def _ensure_out_path(out_arg: str, src: Path) -> Path:
    out_path = Path(out_arg)
    if out_path.is_dir() or (out_arg.endswith(os.sep)):
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path / (src.stem + "_4k.png")
    # parent dir
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _calc_fit_size(w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int]:
    """Resize to fit entirely within (max_w, max_h) preserving aspect."""
    scale = min(max_w / w, max_h / h)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _calc_fill_size(w: int, h: int, min_w: int, min_h: int) -> Tuple[int, int]:
    """Resize to cover (min_w, min_h) preserving aspect (may exceed one dim)."""
    scale = max(min_w / w, min_h / h)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _letterbox(img: Image.Image, target: Tuple[int, int], color: Tuple[int, int, int] = (16, 16, 16)) -> Image.Image:
    """Pad image to the exact target size with given color, preserving aspect."""
    return ImageOps.pad(img, target, method=Image.LANCZOS, color=color, centering=(0.5, 0.5))


def _resize_pillow(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.LANCZOS)


def _resize_esrgan(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    # ESRGAN is scale-based; choose 2x or 4x, then final Pillow resize if necessary.
    w, h = img.size
    # Choose the smallest scale that gets us close to or above the target (prefer 4x when small)
    target_w, target_h = target_size
    scale_needed_w = target_w / w
    scale_needed_h = target_h / h
    scale_needed = max(scale_needed_w, scale_needed_h)
    scale = 4 if scale_needed > 2.5 else (2 if scale_needed > 1.2 else 2)

    model = _load_esrgan(scale=scale)
    up = model.predict(img)
    if up.size != target_size:
        up = up.resize(target_size, Image.LANCZOS)
    return up


def upscale_image(
    src_path: Path,
    out_path: Path,
    mode: str = "fit",
    engine: str = "pillow",
    pad_color: Tuple[int, int, int] = (16, 16, 16),
    overwrite: bool = False,
    jpg_quality: int = 95,
) -> Path:
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out_path}. Use --overwrite to replace.")

    with Image.open(src_path) as im:
        im = im.convert("RGB")
        w, h = im.size

        if mode == "fit":
            nw, nh = _calc_fit_size(w, h, TARGET_W, TARGET_H)
            target_exact = (nw, nh)
            postprocess = None  # no exact 3840x2160 guarantee
        elif mode == "fill":
            nw, nh = _calc_fill_size(w, h, TARGET_W, TARGET_H)
            target_exact = (nw, nh)
            postprocess = "crop_center"
        elif mode == "pad":
            nw, nh = _calc_fit_size(w, h, TARGET_W, TARGET_H)
            target_exact = (nw, nh)
            postprocess = "pad"
        else:
            raise ValueError("mode must be one of: fit, fill, pad")

        # Choose engine
        if engine == "pillow":
            resized = _resize_pillow(im, target_exact)
        elif engine == "esrgan":
            resized = _resize_esrgan(im, target_exact)
        else:
            raise ValueError("engine must be one of: pillow, esrgan")

        if postprocess == "crop_center":
            # Center-crop to exact 3840x2160
            if resized.width != TARGET_W or resized.height != TARGET_H:
                # If we covered (>=), crop center
                cw, ch = TARGET_W, TARGET_H
                left = max(0, (resized.width - cw) // 2)
                top = max(0, (resized.height - ch) // 2)
                resized = resized.crop((left, top, left + cw, top + ch))
        elif postprocess == "pad":
            if resized.size != (TARGET_W, TARGET_H):
                resized = _letterbox(resized, (TARGET_W, TARGET_H), pad_color)

        # Save
        ext = out_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            resized.save(out_path, quality=jpg_quality, optimize=True, progressive=True)
        else:
            # default to PNG for lossless
            resized.save(out_path)

    return out_path


def iter_images(path: Path, recursive: bool = False):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if path.is_file():
        yield path
    else:
        if recursive:
            for p in path.rglob("*"):
                if p.suffix.lower() in exts:
                    yield p
        else:
            for p in path.glob("*"):
                if p.suffix.lower() in exts:
                    yield p


def main(argv=None):
    p = argparse.ArgumentParser(description="Upscale images to 16K (3840×2160).")
    p.add_argument("input", help="Input image file or folder.")
    p.add_argument("-o", "--output", required=True, help="Output file or folder.")
    p.add_argument("--mode", choices=["fit", "fill", "pad"], default="fit",
                   help="Sizing behavior. fit=fit within 3840×2160 (no crop). fill=cover then center-crop to exact 16K. pad=letterbox to exact 16K.")
    p.add_argument("--engine", choices=["pillow", "esrgan"], default="pillow",
                   help="Resize engine. 'pillow' (fast, no-ML) or 'esrgan' (AI super-resolution).")
    p.add_argument("--pad-color", default="#101010", help="Pad color for --mode pad. Hex like #RRGGBB.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--recursive", action="store_true", help="When input is a folder, search recursively.")
    p.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality (if saving .jpg).")
    args = p.parse_args(argv)

    in_path = Path(args.input)
    out_arg = args.output
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 2

    # Parse pad color
    pad_color = args.pad_color
    if isinstance(pad_color, str) and pad_color.startswith("#") and len(pad_color) == 7:
        pad_color = tuple(int(pad_color[i:i+2], 16) for i in (1, 3, 5))
    elif isinstance(pad_color, str):
        print("Invalid --pad-color. Use hex like #112233. Falling back to #101010.", file=sys.stderr)
        pad_color = (16, 16, 16)

    # Prepare list of images
    images = list(iter_images(in_path, recursive=args.recursive))
    if not images:
        print("No images found.", file=sys.stderr)
        return 3

    # If input is a file and output is a file, single processing
    if in_path.is_file() and (not out_arg.endswith(os.sep)) and not Path(out_arg).is_dir():
        out_path = _ensure_out_path(out_arg, in_path)
        try:
            final = upscale_image(in_path, out_path, mode=args.mode, engine=args.engine,
                                  pad_color=pad_color, overwrite=args.overwrite, jpg_quality=args.jpg_quality)
            print(f"Saved: {final}")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Otherwise, treat output as folder
    out_dir = Path(out_arg)
    out_dir.mkdir(parents=True, exist_ok=True)

    errors = 0
    for src in images:
        dest = out_dir / (src.stem + "_4k.png")
        try:
            final = upscale_image(src, dest, mode=args.mode, engine=args.engine,
                                  pad_color=pad_color, overwrite=args.overwrite, jpg_quality=args.jpg_quality)
            print(f"Saved: {final}")
        except Exception as e:
            errors += 1
            print(f"[{src.name}] Error: {e}", file=sys.stderr)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
