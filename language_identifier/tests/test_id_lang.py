# tests/test_id_lang.py

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from language_identifier import id_lang


def test_discover_script_block_finds_full_bengali_block():
    # Bengali's Unicode block has internal gaps of unassigned
    # codepoints; the scan must tolerate them rather than stopping at
    # the first one (regression test for the bug that undercounted the
    # block down to 22 codepoints instead of ~127).
    start, end = id_lang._discover_script_block(0x0995)  # "KA"
    assert start <= 0x0995 <= end
    assert (end - start + 1) > 100


def test_build_script_to_language_map_keeps_multiple_devanagari_candidates():
    # Devanagari is shared by Hindi and Marathi; the map must return
    # BOTH as candidates for that script, not silently keep only one
    # (regression test for the Marathi-read-as-Hindi bug).
    script_map = id_lang.build_script_to_language_map({"hi", "mr", "ml", "ta"})
    devanagari_candidates = {code for code, _ in script_map.get("DEVANAGARI", [])}
    assert {"hi", "mr"}.issubset(devanagari_candidates)


def test_build_script_to_language_map_single_candidate_scripts():
    # A script with only one known language should still resolve to a
    # single, unambiguous candidate.
    script_map = id_lang.build_script_to_language_map({"ml", "ta"})
    assert len(script_map.get("MALAYALAM", [])) == 1


def test_detect_language_from_text_empty_input():
    result = id_lang.detect_language_from_text("", [], {})
    assert result["language"] is None
    assert result["confidence"] == 0.0


def test_detect_language_from_text_certain_script_match():
    script_map = {"MALAYALAM": [("ml", "Malayalam")]}
    result = id_lang.detect_language_from_text("മലയാളം", [], script_map)
    assert result["code"] == "ml"
    assert result["confidence"] == 1.0
