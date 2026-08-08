"""
language_identifier
====================

Detects the language of raw text or an image, then (optionally)
generates real-word/sentence/paragraph or OCR-training-style images in
the detected script — all sourced from real corpora at runtime, with
no per-language data hardcoded into the package itself.

Public API:
    detect_language_from_text  -- identify a language from raw text
    discover_font_for_script   -- find a font that can render a script
"""

from .id_lang import detect_language_from_text, discover_font_for_script

__all__ = [
    "detect_language_from_text",
    "discover_font_for_script",
]

__version__ = "0.1.0"
