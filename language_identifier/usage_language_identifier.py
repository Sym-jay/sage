"""
Example usage of the language_identifier package.

Covers the two ways this package is meant to be used:
  1. As an importable library -- detect_language_from_text() for
     programmatic language identification.
  2. As an installed CLI tool -- gen-para / id-lang, once this package
     has been installed from its wheel (see readme.md).
"""

from language_identifier import detect_language_from_text
from language_identifier.id_lang import (
    get_tesseract_langs,
    get_langdetect_supported_codes,
    build_coverage_info,
    build_script_to_language_map,
)

# --- Library usage: detect a language from raw text -----------------

tess_langs       = get_tesseract_langs()
langdetect_codes = get_langdetect_supported_codes()
_, not_covered   = build_coverage_info(tess_langs, langdetect_codes)
script_map       = build_script_to_language_map(langdetect_codes)

sample_text = "ভাষা শনাক্তকরণ একটি গুরুত্বপূর্ণ কাজ।"
result = detect_language_from_text(sample_text, not_covered, script_map)

if result["language"] is not None:
    print(f"Detected: {result['language']} ({result['code']}) "
          f"— confidence {result['confidence']:.1%}")
else:
    print(f"Could not detect language: {result['warning']}")

# --- CLI usage (once installed from the built wheel) -----------------
#
#   id-lang --text "ভাষা শনাক্তকরণ একটি গুরুত্বপূর্ণ কাজ।"
#
#   gen-para --text "ভাষা শনাক্তকরণ একটি গুরুত্বপূর্ণ কাজ।" \
#            --words 5 --sentences 5 --paragraphs 5
#
#   gen-para --text "বাংলা" --replicate 5000   # OCR-style dataset
