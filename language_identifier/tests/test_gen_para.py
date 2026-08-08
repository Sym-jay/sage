# tests/test_gen_para.py

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from language_identifier import gen_para


def test_strip_format_chars_removes_zero_width_joiner():
    # Regression test: real sentence text can contain invisible
    # shaping-control characters (ZWJ/ZWNJ) that this pipeline's
    # non-shaping renderer would otherwise draw as a stray "tofu" box.
    text_with_zwnj = "আম\u200cি বই"
    cleaned = gen_para._strip_format_chars(text_with_zwnj)
    assert "\u200c" not in cleaned
    assert cleaned == "আমি বই"


def test_strip_format_chars_leaves_normal_text_untouched():
    text = "This has no format characters."
    assert gen_para._strip_format_chars(text) == text


def test_is_renderable_true_when_coverage_missing():
    # If glyph coverage couldn't be loaded, nothing should be blocked --
    # fail open rather than reject all content.
    assert gen_para._is_renderable("anything", None) is True


def test_is_renderable_detects_missing_codepoint():
    covered = {ord(c) for c in "ABC "}
    assert gen_para._is_renderable("ABC", covered) is True
    assert gen_para._is_renderable("ABD", covered) is False


def test_filter_renderable_keeps_only_fully_covered_items():
    covered = {ord(c) for c in "CATDOG "}
    items = ["CAT", "DOG", "BIRD"]
    assert gen_para.filter_renderable(items, covered) == ["CAT", "DOG"]


def test_pick_word_returns_an_item_from_the_list():
    import random
    rng = random.Random(0)
    words = ["alpha", "beta", "gamma"]
    assert gen_para.pick_word(words, rng) in words


def test_user_cache_dir_respects_xdg_cache_home(monkeypatch, tmp_path):
    # Regression test: corpora caching previously lived next to the
    # installed module file, which breaks once that file is inside a
    # (often read-only) site-packages directory. It must now resolve to
    # a proper per-user cache location, honoring $XDG_CACHE_HOME.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    result = gen_para._user_cache_dir("language_identifier")
    assert result == tmp_path / "language_identifier"


def test_user_cache_dir_falls_back_without_xdg_cache_home(monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(gen_para.sys, "platform", "linux")
    result = gen_para._user_cache_dir("language_identifier")
    assert result == gen_para.Path.home() / ".cache" / "language_identifier"


def test_is_script_pure_accepts_cased_script_words():
    # Regression test: Unicode categorizes Latin/Cyrillic/Greek letters
    # as Lu (uppercase) / Ll (lowercase), not "Lo" (Letter, Other) --
    # the category caseless scripts like Malayalam/Tamil/Bengali use.
    # _VALID_CATS previously only included "Lo", which silently rejected
    # every word in any cased language (real symptom: hermitdave's
    # en_50k.txt downloads successfully but load_word_list() returns
    # zero words).
    assert gen_para._is_script_pure("hello", "en") is True
    assert gen_para._is_script_pure("test", "en") is True


def test_is_script_pure_still_accepts_caseless_script_words():
    # Must remain true for the scripts this was already working for.
    assert gen_para._is_script_pure("മലയാളം", "ml") is True
    assert gen_para._is_script_pure("বাংলা", "bn") is True


def test_safe_dirname_lowercases_and_replaces_spaces():
    assert gen_para._safe_dirname("Malayalam") == "malayalam"
    assert gen_para._safe_dirname("Chinese (Simplified)") == "chinese_simplified"


def test_safe_dirname_falls_back_for_empty_input():
    assert gen_para._safe_dirname("") == "unknown_language"


def test_discover_renderable_fonts_excludes_unverifiable_fonts(monkeypatch):
    # Regression test: found via a live end-to-end run where fc-list
    # reported legacy Type1 (.pfb) fonts as language-matching, but
    # fontTools can't parse them to verify real coverage (raises
    # TTLibError). load_font_coverage() correctly returns None for
    # those (can't verify), but discover_renderable_fonts() must NOT
    # treat that as "trust it" the way single-font fallback paths do --
    # unlike a lone chosen font, there's no reason to include an
    # unverified candidate when other, verifiable fonts exist.
    monkeypatch.setattr(gen_para, "discover_all_fonts_for_script",
                         lambda iso2_code: ["/fake/good.ttf", "/fake/bad.pfb"])
    def fake_coverage(path):
        return {ord(c) for c in "Hi"} if path.endswith(".ttf") else None
    monkeypatch.setattr(gen_para, "load_font_coverage", fake_coverage)
    result = gen_para.discover_renderable_fonts("en", "Hi")
    assert result == ["/fake/good.ttf"]


def test_build_font_coverage_pool_excludes_unverifiable_fonts(monkeypatch):
    # The pool-builder that --augment mode reuses across many words
    # must have the same "exclude, don't trust" behavior as
    # discover_renderable_fonts -- this is what --augment's efficiency
    # gain (build once, filter many times) must not silently weaken.
    monkeypatch.setattr(gen_para, "discover_all_fonts_for_script",
                         lambda iso2_code: ["/fake/good.ttf", "/fake/bad.pfb"])
    def fake_coverage(path):
        return {ord(c) for c in "abc"} if path.endswith(".ttf") else None
    monkeypatch.setattr(gen_para, "load_font_coverage", fake_coverage)
    pool = gen_para.build_font_coverage_pool("en")
    assert pool == {"/fake/good.ttf": {ord(c) for c in "abc"}}


def test_fonts_for_text_filters_pool_per_word():
    pool = {
        "/fake/latin_only.ttf": {ord(c) for c in "abc"},
        "/fake/full_coverage.ttf": {ord(c) for c in "abcxyz"},
    }
    # Only one font in the pool covers 'xyz'
    assert gen_para.fonts_for_text("xyz", pool) == ["/fake/full_coverage.ttf"]
    # Both fonts cover 'ab'
    assert set(gen_para.fonts_for_text("ab", pool)) == {
        "/fake/latin_only.ttf", "/fake/full_coverage.ttf"
    }
