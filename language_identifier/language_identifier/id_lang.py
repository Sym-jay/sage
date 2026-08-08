"""
Language Identification Tool
==============================
Given text or an image, identifies the language.
No hardcoded language data anywhere.
"""
import sys
import os
import argparse
import unicodedata
import subprocess
import langcodes
import langdetect
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import pytesseract
from PIL import Image
DetectorFactory.seed = 0
# langcodes.Language.display_name()/.script_name() -- used throughout
# this file to turn an ISO code into a human-readable name/script --
# lazily require the separate `language_data` package as of recent
# langcodes versions. Without it, those calls raise ModuleNotFoundError,
# which previously got swallowed by generic exception handlers and
# surfaced as a misleading "no font found for this script" error with
# no indication the real problem was a missing Python dependency.
# Checking here, once, makes that failure mode loud and correctly
# diagnosed instead. The primary fix is declaring `langcodes[data]` as
# an actual dependency (see setup.py / requirements.txt); this is the
# defense-in-depth check for environments where that was skipped.
try:
    import language_data  # noqa: F401  (presence check only)
    LANGUAGE_DATA_AVAILABLE = True
except ImportError:
    LANGUAGE_DATA_AVAILABLE = False
# One representative codepoint per ISO 15924 script tag.
# Used only to bridge langcodes' tag ("Taml") to unicodedata's block name
# ("TAMIL"). These are Unicode Standard block assignments — structural
# metadata, not language data.
_ISO15924_SAMPLE = {
    "Latn": 0x0041, "Deva": 0x0915, "Beng": 0x0995, "Guru": 0x0A15,
    "Gujr": 0x0A95, "Orya": 0x0B15, "Taml": 0x0B95, "Telu": 0x0C15,
    "Knda": 0x0C95, "Mlym": 0x0D15, "Sinh": 0x0D9A,
}

def _discover_script_block(sample_cp: int, max_gap: int = 24) -> tuple[int, int]:
    """
    Determines the practical Unicode block range around sample_cp by
    scanning outward with unicodedata.name() until the block-name prefix
    changes. This derives block boundaries purely from Unicode Character
    Database metadata at runtime — no per-script range table is
    hardcoded, the same technique _is_script_pure already uses to
    identify a single character's script, just applied in both
    directions to find the block's edges.

    Real Unicode blocks — especially Indic scripts — contain internal
    gaps of unassigned codepoints (Bengali's ~128-codepoint block alone
    has roughly a dozen scattered unassigned slots). A naive scan that
    stops at the very first unassigned codepoint badly undercounts the
    block: for Bengali this found only 22 of ~90 actual letters/marks,
    which made a font with 100% real coverage look untrustworthy. This
    scan tolerates runs of up to `max_gap` consecutive unassigned
    codepoints before concluding the block has genuinely ended, then
    trims any trailing unassigned overshoot back to the last real hit.
    """
    def _name_prefix(cp):
        try:
            return unicodedata.name(chr(cp)).split()[0]
        except ValueError:
            return None  # unassigned codepoint
    script_name = _name_prefix(sample_cp)
    if script_name is None:
        return sample_cp, sample_cp

    start, gap = sample_cp, 0
    while start > 0:
        prefix = _name_prefix(start - 1)
        if prefix is None:
            gap += 1
            if gap > max_gap:
                break
            start -= 1
            continue
        if prefix != script_name:
            break
        gap = 0
        start -= 1
    while start < sample_cp and _name_prefix(start) is None:
        start += 1  # trim trailing unassigned overshoot

    end, gap = sample_cp, 0
    while end < 0x10FFFF:
        prefix = _name_prefix(end + 1)
        if prefix is None:
            gap += 1
            if gap > max_gap:
                break
            end += 1
            continue
        if prefix != script_name:
            break
        gap = 0
        end += 1
    while end > sample_cp and _name_prefix(end) is None:
        end -= 1  # trim trailing unassigned overshoot

    return start, end

def _glyph_has_ink(font, glyph_name: str) -> bool:
    """
    Returns True if the named glyph actually has drawable outline data
    (contours, or components referencing other glyphs with outlines),
    rather than being visually blank.

    This matters because glyph index 0 (.notdef) is not the only way a
    font can fail to really support a character: some fonts map a
    codepoint to a real, non-zero glyph index that is nonetheless an
    empty placeholder — passing a "does this glyph exist" check while
    still rendering as an invisible box. Checking for actual contour
    data is what catches that.
    """
    try:
        if "glyf" in font:
            glyph = font["glyf"][glyph_name]
            if glyph.isComposite():
                return len(glyph.components) > 0
            return getattr(glyph, "numberOfContours", 0) != 0
        if "CFF " in font:
            cff = font["CFF "].cff
            charstrings = cff[cff.fontNames[0]].CharStrings
            # A charstring that only contains "endchar" (a handful of
            # bytes) draws nothing; real ink needs real path operators.
            return len(charstrings[glyph_name].bytecode) > 4
    except Exception:
        pass
    return True  # can't verify this glyph specifically — don't block on it

def _font_cmap_codepoints(font_path: str) -> set[int] | None:
    """
    Returns the set of Unicode codepoints font_path can ACTUALLY render
    a visible glyph for, or None if the font can't be inspected.

    Uses fontTools' getBestCmap(), which applies the same subtable
    priority order real text renderers (FreeType/HarfBuzz) use to pick
    "the" cmap — rather than naively merging every cmap subtable in the
    font. Some fonts carry extra subtables (legacy Mac tables, symbol
    encodings) whose codepoint->glyph mappings aren't what actually gets
    used for standard Unicode text, so merging them all can report false
    coverage. Each candidate glyph is then checked for real ink via
    _glyph_has_ink, catching non-.notdef placeholder glyphs too.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        print("  [WARNING] fontTools is not installed — cannot verify real "
              "glyph coverage during font discovery.")
        print("            Fix: pip install fonttools --break-system-packages")
        return None
    try:
        font = TTFont(font_path, lazy=True, fontNumber=0)
        best = font.getBestCmap()
        if not best:
            # No "preferred" Unicode cmap found by the standard priority
            # rules — fall back to merging raw subtables rather than
            # reporting zero coverage outright.
            best = {}
            for table in font["cmap"].tables:
                best.update(table.cmap)
        covered = set()
        for cp, glyph_name in best.items():
            if font.getGlyphID(glyph_name) == 0:
                continue
            if not _glyph_has_ink(font, glyph_name):
                continue
            covered.add(cp)
        return covered
    except Exception as e:
        print(f"  [WARNING] Could not inspect font '{font_path}' "
              f"({type(e).__name__}: {e}) — treating its coverage as unknown.")
        return None

def _script_coverage_ratio(covered_codepoints: set[int],
                            block_start: int, block_end: int) -> float:
    """
    Fraction of letter/mark codepoints in [block_start, block_end] that
    are present in covered_codepoints. Checking the WHOLE practical
    block — not just one sample letter — is what catches fonts that
    have a base consonant or two but are missing most of a script's
    actual vowel signs, conjunct formers, and digits. A font that passes
    a single-codepoint spot check can still fail almost every real word.
    """
    total = hit = 0
    for cp in range(block_start, block_end + 1):
        try:
            if unicodedata.category(chr(cp)) not in _VALID_CATS:
                continue
        except ValueError:
            continue
        total += 1
        if cp in covered_codepoints:
            hit += 1
    return (hit / total) if total else 1.0
# Unicode General Categories that count as "a letter or combining mark"
# (same definition gen_para.py's corpus filters use). Covers ALL letter
# categories -- Lu/Ll/Lt for bicameral scripts (Latin, Cyrillic, Greek),
# Lo for caseless scripts (Malayalam, Tamil, Bengali, Devanagari) -- not
# just "Lo" alone, which would silently exclude every cased-script
# character from font block-coverage estimates.
_VALID_CATS = {"Lu", "Ll", "Lt", "Lo", "Mn", "Mc"}

def discover_font_for_script(iso2_code: str) -> str:
    """
    Discovers the best available vector font for the given language at
    runtime using fc-match, THEN scores every candidate by how much of
    the script's actual Unicode block it covers (see
    _script_coverage_ratio) rather than trusting fc-match's first
    answer. A single-codepoint spot check is not sufficient: a font can
    contain one common base letter of a script "by accident" while
    genuinely lacking most of what real words need (vowel signs,
    conjuncts, digits), producing near-total "tofu" in output that a
    one-letter check would have let through.
    Strategy:
      1. Get the script name from langcodes (e.g. "ml" -> "Malayalam")
      2. Query fc-match for "NotoSans<ScriptName>", ":lang=<iso2>", and
         "unifont" — collect whichever of these fontconfig can resolve
      3. Score each resolved font's coverage of the script's whole
         Unicode block; return the highest-scoring one that clears a
         high coverage bar (95%)
      4. If none clears that bar, return the best-covering candidate
         found anyway (content selection in main.py adds a second,
         per-item safety net so nothing unrenderable is actually used)
    Nothing is hardcoded — font path and script range are always
    derived at runtime.
    """
    try:
        lang        = langcodes.get(iso2_code).maximize()
        script_full = lang.script_name()          # e.g. "Malayalam"
        sample_cp   = _ISO15924_SAMPLE.get(lang.script)
        block       = _discover_script_block(sample_cp) if sample_cp is not None else None
        noto_name   = f"NotoSans{script_full.replace(' ', '')}"

        best_path, best_ratio = None, -1.0
        seen_paths = set()
        for query in [noto_name, f":lang={iso2_code}", "unifont"]:
            result = subprocess.run(
                ["fc-match", "--format=%{file}", query],
                capture_output=True, text=True
            )
            path = result.stdout.strip()
            if not path or not os.path.exists(path) or path in seen_paths:
                continue
            seen_paths.add(path)
            if block is None:
                print(f"  fc-match {query!r} -> {path}  "
                      f"(script block unknown — trusting fontconfig as-is)")
                return path
            covered = _font_cmap_codepoints(path)
            if covered is None:
                print(f"  fc-match {query!r} -> {path}  "
                      f"(coverage unverifiable — trusting fontconfig as-is)")
                return path
            ratio = _script_coverage_ratio(covered, *block)
            print(f"  fc-match {query!r} -> {path}  "
                  f"(covers {ratio:.1%} of the {script_full} block)")
            if ratio >= 0.95:
                return path
            if ratio > best_ratio:
                best_path, best_ratio = path, ratio
        if best_path is not None and best_ratio > 0:
            print(f"  No candidate reached 95% coverage — using the best "
                  f"one found ({best_ratio:.1%}): {best_path}")
            return best_path
    except Exception as e:
        print(f"  [WARNING] Font discovery hit an unexpected error "
              f"({type(e).__name__}: {e}).")
    sys.exit(
        f"[ERROR] No installed font actually covers the script needed for "
        f"'{iso2_code}'.\n"
        "        fontconfig likely returned a fallback font with no glyphs "
        "for this script (this\n"
        "        produces invisible \"tofu\" boxes instead of visible text "
        "if not caught).\n"
        "        Install a font covering this script, e.g. the full Noto "
        "family:\n"
        "          sudo pacman -S noto-fonts noto-fonts-extra   (Arch)\n"
        "          sudo apt install fonts-noto-core fonts-noto-extra   "
        "(Debian/Ubuntu)\n"
        "        Then run: fc-cache -fv"
    )
def discover_all_fonts_for_script(iso2_code: str) -> list[str]:
    """
    Returns every distinct font file installed on the system that
    fontconfig considers a match for iso2_code's language, via
    `fc-list :lang=<iso2>` — fontconfig's own "which fonts support this
    language" query, rather than the single best-match guess
    discover_font_for_script makes.

    Used when VARIETY across many different fonts is the goal (e.g.
    generating diverse OCR training images of one fixed word), rather
    than picking the single best font for rendering prose. Callers
    still need to verify each candidate can render their exact target
    text — this only reflects fontconfig's own language-coverage
    metadata, which (as seen with discover_font_for_script) isn't
    always accurate on its own.
    """
    try:
        result = subprocess.run(
            ["fc-list", f":lang={iso2_code}", "--format", "%{file}\n"],
            capture_output=True, text=True
        )
        paths = {p.strip() for p in result.stdout.splitlines() if p.strip()}
        return sorted(p for p in paths if os.path.exists(p))
    except Exception as e:
        print(f"  [WARNING] fc-list failed to enumerate fonts for "
              f"'{iso2_code}' ({type(e).__name__}: {e}).")
        return []
def get_tesseract_langs() -> list[str]:
    try:
        raw = pytesseract.get_languages()
    except pytesseract.TesseractNotFoundError:
        sys.exit(
            "[ERROR] Tesseract not found.\n"
            "        Install: sudo pacman -S tesseract\n"
            "        Then: sudo pacman -S tesseract-data-eng tesseract-data-mal ..."
        )
    return [code for code in raw if langcodes.tag_is_valid(code)]

def get_langdetect_supported_codes() -> set[str]:
    profile_dir = os.path.join(os.path.dirname(langdetect.__file__), "profiles")
    return set(os.listdir(profile_dir))

def build_coverage_info(tess_langs, langdetect_codes):
    iso_to_name, not_covered = {}, []
    for tess_code in tess_langs:
        try:
            lang = langcodes.get(tess_code)
            iso2, name = lang.language, lang.display_name()
        except Exception:
            iso2, name = tess_code, tess_code
        iso_to_name[iso2] = name
        if iso2 not in langdetect_codes:
            not_covered.append(name)
    return iso_to_name, not_covered

def build_script_to_language_map(langdetect_codes):
    """
    Returns dict: UD/Unicode script block name -> list of (iso2, name)
    candidates.

    Candidates are derived from every language langdetect ships a
    profile for — NOT from installed tesseract packs. Language
    identification runs on raw Unicode text, whether it came from
    --text or was already pulled out of an image by OCR, so the
    candidate set it reasons over must not depend on which OCR data
    packs happen to be installed on this particular machine. Tying this
    map to tesseract's installed packs was the root cause of Marathi
    being misidentified as Hindi: if the 'mar' tesseract pack wasn't
    installed, Marathi never entered the candidate set for Devanagari,
    so Devanagari text always fell through to the sole remaining
    Devanagari candidate (Hindi) at a false 100% confidence — regardless
    of what script the actual input text used.

    A script maps to a SINGLE candidate when only one langdetect-known
    language uses it (e.g. Malayalam script -> just "ml") — in that case
    the script alone identifies the language with certainty.

    A script can also map to MULTIPLE candidates, because several
    languages legitimately share one script — most notably Devanagari,
    used by Hindi, Marathi, and Nepali alike. In that case script
    identity alone is NOT enough to tell them apart; detect_language_
    from_text() falls back to a statistical model restricted to just
    that candidate set. This is why the map returns a list rather than
    a single tuple: a 1:1 script-to-language assumption silently breaks
    the moment two same-script languages both need to be supported.
    """
    result = {}
    for code in langdetect_codes:
        try:
            lang      = langcodes.get(code)
            iso2      = lang.language
            name      = lang.display_name()
            iso15924  = lang.maximize().script
            cp        = _ISO15924_SAMPLE.get(iso15924)
            if cp is None:
                continue
            ud_script = unicodedata.name(chr(cp)).split()[0]
            bucket    = result.setdefault(ud_script, [])
            if not any(existing_iso2 == iso2 for existing_iso2, _ in bucket):
                bucket.append((iso2, name))
        except Exception:
            continue
    return result

def resolve_language_name(code):
    try:
        return langcodes.get(code).display_name()
    except Exception:
        return code

def extract_text_from_image(image_path, tess_lang_string):
    try:
        img = Image.open(image_path)
    except Exception as e:
        sys.exit(f"[ERROR] Could not open image: {e}")
    try:
        text = pytesseract.image_to_string(img, lang=tess_lang_string)
    except pytesseract.TesseractError as e:
        sys.exit(f"[ERROR] OCR failed: {e}")
    return text.strip()

def get_dominant_script(text):
    counts = {}
    for ch in text:
        if not ch.strip() or not ch.isalpha():
            continue
        try:
            script = unicodedata.name(ch).split()[0]
            counts[script] = counts.get(script, 0) + 1
        except ValueError:
            continue
    return max(counts, key=counts.get) if counts else None

def detect_language_from_text(text, not_covered_by_langdetect, script_map):
    text = text.strip()
    if not text:
        return {"language": None, "code": None, "confidence": 0.0,
                "candidates": [], "warning": "No text to analyze."}
    dominant_script      = get_dominant_script(text)
    candidates_for_script = script_map.get(dominant_script, [])
    # Certain case: exactly one installed language uses this script.
    if len(candidates_for_script) == 1:
        iso2, name = candidates_for_script[0]
        return {"language": name, "code": iso2, "confidence": 1.0,
                "candidates": [(name, iso2, 1.0)], "warning": ""}
    # Ambiguous or uncovered case: either several installed languages
    # share this script (e.g. Devanagari -> Hindi + Marathi) and script
    # identity alone can't disambiguate, or no installed tesseract
    # language claims this script at all. Either way, fall back to
    # langdetect's per-language statistical model.
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return {"language": None, "code": None, "confidence": 0.0,
                "candidates": [],
                "warning": "Could not detect language — text may be too short."}
    # If the script narrowed things down to a known candidate set,
    # prefer the highest-probability langdetect result that's actually
    # in that set — this is what correctly separates Hindi from Marathi
    # even though both are Devanagari script.
    if candidates_for_script:
        allowed  = {iso2 for iso2, _ in candidates_for_script}
        narrowed = [c for c in candidates if c.lang in allowed]
        if narrowed:
            candidates = narrowed
    best = candidates[0]
    name = resolve_language_name(best.lang)
    warning = ""
    if best.prob < 0.75 and not_covered_by_langdetect:
        missing = ", ".join(not_covered_by_langdetect)
        warning = (f"Low confidence ({best.prob:.0%}). "
                   f"Languages not in langdetect: {missing}.")
    return {
        "language": name, "code": best.lang,
        "confidence": round(best.prob, 4),
        "candidates": [(resolve_language_name(c.lang), c.lang, round(c.prob, 4))
                       for c in candidates],
        "warning": warning,
    }

def main():
    parser = argparse.ArgumentParser(description="Identify language of text or image.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",  metavar="TEXT")
    group.add_argument("--image", metavar="PATH")
    args = parser.parse_args()
    if not LANGUAGE_DATA_AVAILABLE:
        sys.exit(
            "[ERROR] The 'language_data' package is required but not "
            "installed.\n"
            "        langcodes needs it to resolve language/script names "
            "(display_name(), script_name()) --\n"
            "        without it, language and font detection fail with "
            "misleading errors elsewhere.\n"
            "        Install it, then re-run:\n"
            "          pip install language_data"
        )
    tess_langs       = get_tesseract_langs()
    langdetect_codes = get_langdetect_supported_codes()
    _, not_covered   = build_coverage_info(tess_langs, langdetect_codes)
    script_map       = build_script_to_language_map(langdetect_codes)
    tess_lang_string = "+".join(tess_langs)
    print(f"Tesseract packs : {len(tess_langs)}")
    print(f"langdetect      : {len(langdetect_codes)} languages")
    if not_covered:
        print(f"Coverage gap    : {', '.join(not_covered)}")
    print()
    text = (extract_text_from_image(args.image, tess_lang_string)
            if args.image else args.text)
    result = detect_language_from_text(text, not_covered, script_map)
    print("─" * 50)
    if result["language"] is None:
        print(f"Result  : Could not identify\nReason  : {result['warning']}")
    else:
        print(f"Language   : {result['language']} ({result['code']})")
        print(f"Confidence : {result['confidence']:.1%}")
        if len(result["candidates"]) > 1:
            for name, code, prob in result["candidates"][1:4]:
                print(f"  {name} ({code}) — {prob:.1%}")
        if result["warning"]:
            print(f"Warning : {result['warning']}")
    print("─" * 50)
if __name__ == "__main__":
    main()
