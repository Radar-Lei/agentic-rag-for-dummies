import os

import gradio as gr
from ui.css import custom_css
from ui.gradio_app import create_gradio_ui
from pathlib import Path


def _print_ocr_status() -> None:
    cv2_ok = False
    try:
        import cv2  # noqa: F401

        cv2_ok = True
    except Exception:
        cv2_ok = False

    tessdata_path = None
    tessdata_ok = False
    try:
        import pymupdf

        if hasattr(pymupdf, "get_tessdata"):
            tessdata_path = pymupdf.get_tessdata()
            tessdata_ok = bool(tessdata_path and os.path.isdir(tessdata_path))
    except Exception:
        tessdata_ok = False

    if cv2_ok and tessdata_ok:
        print(f"OCR 已启用（tessdata：{tessdata_path}）")
        return

    print("OCR 未启用。")
    if not cv2_ok:
        print("如需启用 OCR：`pip install opencv-python-headless`")
    if not tessdata_ok:
        print("macOS 启用 OCR：`brew install tesseract`（安装后重启应用）")


if __name__ == "__main__":
    _print_ocr_status()
    demo = create_gradio_ui()
    print("\n启动：智慧问答助手 ...")
    repo_root = Path(__file__).resolve().parent.parent
    favicon = repo_root / "assets" / "logo_replace.png"
    demo.launch(css=custom_css, favicon_path=favicon)
