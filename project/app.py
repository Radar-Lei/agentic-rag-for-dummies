import os

import gradio as gr
from ui.css import custom_css
from ui.gradio_app import create_gradio_ui


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
        print(f"OCR enabled (tessdata: {tessdata_path})")
        return

    print("OCR disabled.")
    if not cv2_ok:
        print("To enable OCR: `pip install opencv-python-headless`")
    if not tessdata_ok:
        print("To enable OCR on macOS: `brew install tesseract` (then restart the app)")


if __name__ == "__main__":
    _print_ocr_status()
    demo = create_gradio_ui()
    print("\n🚀 Launching RAG Assistant...")
    demo.launch(css=custom_css)
