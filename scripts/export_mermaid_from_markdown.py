#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image


MERMAID_START = "```mermaid"
CODE_FENCE = "```"


def slugify(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "", text)
    return text[:80] or "figure"


def extract_blocks(markdown_path: Path):
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == MERMAID_START:
            start = i + 1
            i += 1
            body = []
            while i < len(lines) and lines[i].strip() != CODE_FENCE:
                body.append(lines[i])
                i += 1
            caption = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith("图"):
                caption = lines[j].strip()
            blocks.append(("\n".join(body).strip() + "\n", caption))
        i += 1
    return blocks


def ensure_mermaid_cli() -> list[str]:
    local = ["npx", "-y", "@mermaid-js/mermaid-cli"]
    return local


def export_block(cli_cmd: list[str], mmd_path: Path, png_path: Path, scale: int):
    cmd = cli_cmd + [
        "-i",
        str(mmd_path),
        "-o",
        str(png_path),
        "-s",
        str(scale),
        "-b",
        "white",
    ]
    subprocess.run(cmd, check=True)


def apply_png_dpi(png_path: Path, dpi: int):
    with Image.open(png_path) as img:
        img.save(png_path, dpi=(dpi, dpi))


def main():
    parser = argparse.ArgumentParser(
        description="Extract Mermaid blocks from Markdown and batch export PNGs."
    )
    parser.add_argument("markdown", type=str, help="Markdown file path")
    parser.add_argument(
        "--mmd-dir",
        type=str,
        default="mermaid-src",
        help="Directory for extracted .mmd files",
    )
    parser.add_argument(
        "--png-dir",
        type=str,
        default="mermaid-png",
        help="Directory for exported PNG files",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="PNG export scale passed to Mermaid CLI",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG dpi metadata to write after export",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract .mmd files without exporting PNG",
    )
    args = parser.parse_args()

    markdown_path = Path(args.markdown).expanduser().resolve()
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    blocks = extract_blocks(markdown_path)
    if not blocks:
        print(f"No Mermaid blocks found in {markdown_path}")
        return

    stem = markdown_path.stem
    mmd_root = Path(args.mmd_dir).expanduser().resolve() / stem
    png_root = Path(args.png_dir).expanduser().resolve() / stem
    mmd_root.mkdir(parents=True, exist_ok=True)
    if not args.extract_only:
        png_root.mkdir(parents=True, exist_ok=True)

    cli_cmd = ensure_mermaid_cli()

    for idx, (diagram, caption) in enumerate(blocks, start=1):
        suffix = slugify(caption) if caption else f"fig_{idx:02d}"
        base_name = f"{idx:02d}_{suffix}"
        mmd_path = mmd_root / f"{base_name}.mmd"
        mmd_path.write_text(diagram, encoding="utf-8")
        print(f"Extracted: {mmd_path}")

        if args.extract_only:
            continue

        png_path = png_root / f"{base_name}.png"
        try:
            export_block(cli_cmd, mmd_path, png_path, args.scale)
            apply_png_dpi(png_path, args.dpi)
            print(f"Exported:  {png_path}")
        except subprocess.CalledProcessError as exc:
            print(f"Failed to export {mmd_path.name}: {exc}", file=sys.stderr)
            print(
                "Tip: run again later, or first install Mermaid CLI with `npm i -g @mermaid-js/mermaid-cli`.",
                file=sys.stderr,
            )
            raise


if __name__ == "__main__":
    main()
