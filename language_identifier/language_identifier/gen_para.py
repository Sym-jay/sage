"""
Pipeline: Predict Language -> Generate Word / Sentence / Paragraph Images
===========================================================================
HOW MEANINGFUL CONTENT IS GENERATED (no API, no hardcoded word lists)
-----------------------------------------------------------------------
  WORDS   : Sourced from a FREQUENCY WORD LIST (hermitdave/FrequencyWords
            on GitHub). This gives the most COMMONLY USED words in the
            detected language — real, everyday words. Only words appearing
            at least 10 times in real text are kept, filtered to the
            correct Unicode script block (no English, no fragments).
            If a language has no list on hermitdave (e.g. Marathi), a
            word-frequency list is derived instead from the token stream
            of the same Universal Dependencies treebank used for
            sentences below — every token FORM in a UD treebank is a
            real, attested word, so this needs no separate dictionary
            source and no hardcoded per-language data.
  SENTENCES / PARAGRAPHS : Sourced from the Universal Dependencies (UD)
            treebank for the detected language — real, grammatically correct
            sentences annotated by linguists for NLP research. These are
            actual sentences with meaning, not synthetic combinations.
            Paragraphs are formed by grouping consecutive real sentences.
  Both sources are downloaded ONCE per language, cached in ./corpora/,
  and reused on all subsequent runs without re-downloading.
  The download URLs are CONSTRUCTED from the language code at runtime
  via langcodes — nothing is hardcoded per language.
"""
import sys
import os
import csv
import argparse
import random
import subprocess
import unicodedata
import urllib.request
import urllib.error
import socket
from collections import Counter
from pathlib import Path
import langcodes
from PIL import Image, ImageDraw, ImageFont
from language_identifier.id_lang import (
    discover_font_for_script,
    discover_all_fonts_for_script,
    get_tesseract_langs,
    get_langdetect_supported_codes,
    build_coverage_info,
    build_script_to_language_map,
    detect_language_from_text,
    extract_text_from_image,
    LANGUAGE_DATA_AVAILABLE,
)
# fontTools powers every render-safety check in this file (see the
# RENDER SAFETY section below): it's what proves a chosen font can
# ACTUALLY draw the specific words/sentences about to be rendered,
# rather than just trusting fontconfig's font-matching guess. Checking
# its availability once, up front, and failing loudly if it's missing
# is deliberate — every one of those checks silently no-ops without it
# (falls back to "trust the input"), which would make font-coverage
# bugs (invisible "tofu" boxes) reappear with no error message at all
# to explain why. A loud failure here is far more useful than a silent
# one three steps later.
try:
    import fontTools  # noqa: F401  (presence check only)
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False
# ─────────────────────────────────────────────────────────────────────────────
# TUNING CONSTANTS  (rendering / content behaviour — not language data)
# ─────────────────────────────────────────────────────────────────────────────
def _user_cache_dir(app_name: str) -> Path:
    """
    Returns a writable, per-user cache directory for disposable,
    re-downloadable data — following the XDG Base Directory spec on
    Linux ($XDG_CACHE_HOME, or ~/.cache as the standard fallback), with
    the platform-conventional equivalents on macOS and Windows.

    This exists because CORPORA_DIR previously resolved relative to
    this module's OWN file location (Path(__file__).parent). That only
    worked by coincidence when running the script directly from a repo
    checkout: once installed from a wheel, the package typically lands
    in site-packages, which is often read-only and is never where a
    user expects a tool's cache to live regardless. Nothing about which
    directory to use is hardcoded per-language — this is a general
    packaging concern, resolved the same way for every language.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / app_name / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / app_name
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return base / app_name
CORPORA_DIR   = _user_cache_dir("language_identifier") / "corpora"
# Generated images are a deliverable the user is actively looking for,
# not disposable cache data — so unlike CORPORA_DIR, this defaults to
# relative to the CURRENT WORKING DIRECTORY (wherever the user runs
# `gen-para` from), matching the "look in ./output/images" convention
# most CLI tools use for their actual output. It's only a DEFAULT: the
# --output-dir flag (see main()) lets the person running the tool pick
# any location instead, so this constant is never the final word on
# where files land.
DEFAULT_OUT_DIR = Path.cwd() / "output" / "images"
def _safe_dirname(name: str) -> str:
    """
    Turns a language display name (e.g. "Malayalam", "Chinese
    (Simplified)") into a safe, lowercase directory-name component:
    lowercased, whitespace collapsed to underscores, and anything
    outside [a-z0-9_-] stripped. Used to organize generated output into
    one subfolder per detected language — derived from whatever name
    langcodes resolves at runtime, not a hardcoded per-language mapping.
    """
    slug = name.strip().lower().replace(" ", "_")
    slug = "".join(ch for ch in slug if ch.isalnum() or ch in "_-")
    return slug or "unknown_language"
PAD              = 30
MAX_IMG_WIDTH    = 1024
LINE_SPACING     = 10
MIN_WORD_FREQ    = 10       # minimum corpus frequency to keep a word
MIN_WORD_CHARS   = 3        # minimum character length for words
MAX_WORD_CHARS   = 15       # maximum character length for words
MIN_WORD_POOL    = 20       # if fewer words clear MIN_WORD_FREQ, relax the floor
UD_FALLBACK_FILES = 6       # max UD treebank files to merge for the word fallback
PARA_SIZE        = 3        # number of sentences per paragraph
WORD_FONT_RANGE  = (48, 90)
SENT_FONT_RANGE  = (26, 40)
PARA_BODY_RANGE  = (18, 26)
PARA_SUB_EXTRA   = (8, 14)
PARA_TTL_EXTRA   = (12, 20)
DOWNLOAD_TIMEOUT = 30       # seconds
# --replicate mode: many varied renders of ONE fixed word/letter (e.g.
# for an OCR/training dataset), rather than many DIFFERENT corpus items.
REPLICATE_FONT_SIZE_RANGE   = (48, 140)
REPLICATE_ROTATION_DEG_RANGE = (-8, 8)
REPLICATE_PADDING_RANGE     = (30, 70)
REPLICATE_BG_LUMINANCE_RANGE = (235, 255)  # near-white background jitter
REPLICATE_FG_LUMINANCE_RANGE = (0, 45)     # near-black text jitter
# Unicode General Categories that count as "a letter or combining mark"
# (Unicode Standard). Deliberately covers ALL letter categories, not
# just "Lo": caseless scripts (Malayalam, Tamil, Bengali, Devanagari,
# ...) categorize their letters as "Lo" (Letter, Other), but bicameral
# scripts (Latin, Cyrillic, Greek, ...) use "Lu"/"Ll" (upper/lowercase)
# instead -- "Lo" alone silently rejects every word in any cased
# language. "Lt" (titlecase, e.g. Dž-style digraphs) is included for
# the same completeness reason. Mn/Mc cover combining diacritics and
# Indic vowel signs. This is the full, correct definition of "letter or
# mark" across scripts, not a per-script special case.
_VALID_CATS = {"Lu", "Ll", "Lt", "Lo", "Mn", "Mc"}
# ─────────────────────────────────────────────────────────────────────────────
# CORPUS URL CONSTRUCTION  (built from language code, not hardcoded)
# ─────────────────────────────────────────────────────────────────────────────
def get_word_frequency_urls(iso2_code: str) -> list[str]:
    """
    Returns candidate URLs for the frequency word list for iso2_code.
    hermitdave/FrequencyWords names files differently across languages
    (some have "_50k.txt", others only "_full.txt") — both are tried.
    The language code in the URL is taken directly from iso2_code —
    no per-language filename is hardcoded.
    """
    base = (f"https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
            f"master/content/2018/{iso2_code}")
    return [f"{base}/{iso2_code}_50k.txt", f"{base}/{iso2_code}_full.txt"]

def _get_ud_repo_names() -> list[str]:
    """
    Returns every "UD_*" treebank repo name under the UniversalDependencies
    GitHub org, fetched once and cached to disk (corpora/_ud_repo_index.json).

    This exists to sidestep a real, non-obvious mismatch: UD repo names
    use whatever English exonym the linguistics community settled on for
    a language (e.g. the repo is "UD_Bengali-BRU"), which does not always
    equal langcodes' CLDR-preferred display name for the same ISO code
    (bn's CLDR display name is "Bangla", not "Bengali"). Searching GitHub
    for a repo matching langcodes' display name can therefore silently
    miss a treebank that does exist. Listing every repo once and
    resolving each repo's OWN name back to an ISO code (done in
    discover_ud_treebank_url) works regardless of which exonym either
    side happens to prefer, without hardcoding any language-specific
    alias table.
    """
    import json
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    cache = CORPORA_DIR / "_ud_repo_index.json"
    if cache.exists() and cache.stat().st_size > 0:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    names, page = [], 1
    while True:
        url = (f"https://api.github.com/orgs/UniversalDependencies/repos"
               f"?per_page=100&page={page}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                batch = json.loads(r.read())
        except Exception as e:
            # A common real-world cause: GitHub's unauthenticated API
            # allows only 60 requests/hour, easily exhausted across a
            # busy session (this pipeline makes several other GitHub
            # API calls per language). When that happens mid-pagination,
            # `names` so far is necessarily incomplete for later-
            # alphabetical repos — discover_ud_treebank_url()'s further
            # fallback layers still work, but may land on a smaller or
            # less standard treebank than the best available one. This
            # is reported rather than silently swallowed so that's
            # visible when it happens, instead of looking identical to
            # "this language genuinely has no treebank."
            print(f"  [WARNING] UD repo listing stopped at page {page} "
                  f"({type(e).__name__}: {e}) — likely GitHub API rate "
                  f"limiting. Falling back to per-language search; this "
                  f"may miss the largest/preferred treebank for some "
                  f"languages.")
            break
        if not batch:
            break
        names.extend(repo["name"] for repo in batch
                     if repo.get("name", "").startswith("UD_"))
        if len(batch) < 100:
            break
        page += 1
    if names:
        cache.write_text(json.dumps(names), encoding="utf-8")
    return names

def _get_ud_language_index() -> dict:
    """
    Returns dict: normalized ISO 639 code -> the exact English language
    name UD uses in its repo names, read from UD's OWN authoritative
    source: docs-automation/codes_and_flags.yaml. Fetched once and
    cached to disk (corpora/_ud_language_index.json).

    UD's own tooling (udlib.pm's get_language_hash, used when generating
    every "UD_<LanguageName>-<ProjectCode>" repo) reads this exact file
    to go from ISO code to the language name baked into repo names. It
    is a documented one-to-one mapping ("There is a one-to-one mapping
    between language names and ISO 639 codes" per UD's own new-language
    checklist). Using it directly cannot disagree with the repo names
    themselves — unlike guessing from langcodes' CLDR display name,
    which for bn resolves to "Bangla" while the actual repo is
    "UD_Bengali-BRU".
    """
    import json, re
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    cache = CORPORA_DIR / "_ud_language_index.json"
    if cache.exists() and cache.stat().st_size > 0:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    url  = ("https://raw.githubusercontent.com/UniversalDependencies/"
            "docs-automation/master/codes_and_flags.yaml")
    dest = CORPORA_DIR / "_codes_and_flags.yaml"
    index = {}
    if _download(url, dest):
        try:
            text = dest.read_text(encoding="utf-8")
            # codes_and_flags.yaml is a flat mapping: top-level keys are
            # English language names, each with an indented "lcode: xx"
            # field among others. Parsed directly (no PyYAML dependency)
            # since the structure is this narrow and well-documented.
            current_name = None
            for line in text.splitlines():
                if not line.strip() or line.strip().startswith("#"):
                    continue
                if not line.startswith((" ", "\t")) and line.rstrip().endswith(":"):
                    current_name = line.rstrip()[:-1].strip().strip('"').strip("'")
                elif current_name and re.match(r"[ \t]+lcode:\s*\S+", line):
                    raw_code = line.split(":", 1)[1].strip().strip('"').strip("'")
                    try:
                        norm_code = langcodes.get(raw_code).language
                    except Exception:
                        norm_code = raw_code
                    index[norm_code] = current_name
        except Exception:
            index = {}
    dest.unlink(missing_ok=True)
    if index:
        cache.write_text(json.dumps(index), encoding="utf-8")
    return index

def discover_ud_treebank_url(iso2_code: str) -> list[tuple[str, str]]:
    """
    Discovers the Universal Dependencies treebank repository for
    iso2_code's language AT RUNTIME — no repo name or filename is
    hardcoded per language.
    UD treebank repos are named "UD_<LanguageName>-<ProjectCode>" where
    the project code (e.g. "TTB" for Tamil, "GujTB" for Gujarati) is
    assigned arbitrarily by the UD project per treebank and cannot be
    derived from the language name — it must be discovered.

    Strategy, most to least reliable:
      1. Look up iso2_code in UD's own authoritative name index
         (_get_ud_language_index / codes_and_flags.yaml) and match the
         repo index (_get_ud_repo_names) against that exact name.
      2. If the code isn't in that index, fall back to resolving each
         repo's own <LanguageName> segment back to an ISO code via
         langcodes.find() — still repo-index-driven, just using a
         different name resolver.
      3. GitHub's repository search API using langcodes' CLDR display
         name — used only if the repo index couldn't be fetched at all.
      4. Common UD project codes tried directly (no API, no rate limit)
         against every language-name spelling seen so far. This list is
         a set of project-naming conventions reused across UD for many
         languages, not data about any specific language.
    """
    import json

    repo_index = _get_ud_repo_names()

    def _repos_named(target_lang_segment: str) -> list[str]:
        target = target_lang_segment.replace(" ", "_")
        return [r for r in repo_index
                if "-" in r[3:] and r[3:].rsplit("-", 1)[0] == target]

    matched_repos = []
    ud_name = _get_ud_language_index().get(iso2_code)
    if ud_name:
        matched_repos = _repos_named(ud_name)

    if not matched_repos:
        for repo_name in repo_index:
            body = repo_name[3:]  # strip leading "UD_"
            if "-" not in body:
                continue
            lang_part, _project_code = body.rsplit("-", 1)
            try:
                resolved = langcodes.find(lang_part.replace("_", " "))
                if resolved.language == iso2_code:
                    matched_repos.append(repo_name)
            except LookupError:
                continue

    if not matched_repos:
        reason = ("the UD repo index was empty (see the rate-limiting "
                   "warning above)" if not repo_index else
                   "no repo in the index matched this language")
        print(f"  Repo-index lookup found nothing for '{iso2_code}' "
              f"({reason}); trying GitHub search next.")

    def _list_conllu_files(repo_name: str) -> list[tuple[str, str]]:
        try:
            url = (f"https://api.github.com/repos/UniversalDependencies/"
                   f"{repo_name}/contents/")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                files = json.loads(r.read())
            return [(f["download_url"], f["name"]) for f in files
                    if f["name"].endswith(".conllu")]
        except Exception:
            return []

    candidates = []
    for repo_name in matched_repos:
        candidates.extend(_list_conllu_files(repo_name))
    if candidates:
        candidates.sort(key=lambda x: (
            0 if "test" in x[1] else 1 if "dev" in x[1] else 2
        ))
        return candidates

    # Secondary: GitHub search API using langcodes' display name.
    lang_name = langcodes.get(iso2_code).display_name()
    repo_name = None
    try:
        query = f"org:UniversalDependencies+UD_{lang_name.replace(' ', '_')}"
        url   = f"https://api.github.com/search/repositories?q={query}"
        req   = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        items = data.get("items", [])
        if items:
            repo_name = items[0]["name"]
    except Exception:
        pass
    if repo_name:
        candidates.extend(_list_conllu_files(repo_name))
    if candidates:
        candidates.sort(key=lambda x: (
            0 if "test" in x[1] else 1 if "dev" in x[1] else 2
        ))
        return candidates

    # Last resort: common UD project codes tried directly against every
    # language-name spelling encountered (CLDR display name, plus any
    # repo-index language segments already known to resolve here).
    fallback_codes = ["TTB", "UFAL", "GSD", "PUD", "TRG", "MWTT", "VLSP",
                       "HDTB", "BRU", "BHTB", "GujTB", "CS", "YCU"]
    name_variants = {lang_name.replace(" ", "_")}
    for repo_name in matched_repos:
        body = repo_name[3:]
        if "-" in body:
            name_variants.add(body.rsplit("-", 1)[0])
    results = []
    for lang_underscored in name_variants:
        for code in fallback_codes:
            repo = f"UD_{lang_underscored}-{code}"
            for split in ["test", "dev"]:
                fname = f"{iso2_code}_{code.lower()}-ud-{split}.conllu"
                url = (f"https://raw.githubusercontent.com/UniversalDependencies/"
                       f"{repo}/master/{fname}")
                results.append((url, f"{repo}/{fname}"))
    return results
# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _download(url: str, dest: Path) -> bool:
    """
    Downloads url to dest with chunked reads, live progress bar,
    and a timeout on every read. Returns True on success.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            total      = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        bar = ("█" * int(pct // 5)
                               + "░" * (20 - int(pct // 5)))
                        print(f"\r    [{bar}] {pct:5.1f}%"
                              f"  {downloaded//1024}KB / {total//1024}KB",
                              end="", flush=True)
        print()
        return dest.exists() and dest.stat().st_size > 0
    except (urllib.error.HTTPError, urllib.error.URLError,
            socket.timeout, OSError) as e:
        print(f"\n    ✗ {e}")
        if dest.exists():
            dest.unlink()
        return False
# ─────────────────────────────────────────────────────────────────────────────
# WORD CORPUS — frequency word list
# ─────────────────────────────────────────────────────────────────────────────
def _is_script_pure(word: str, iso2_code: str) -> bool:
    """
    Returns True if every character in word belongs to the Unicode block
    for iso2_code's script AND is a letter/combining mark.
    Derived at runtime from langcodes + unicodedata — no block range hardcoded.
    """
    try:
        iso15924    = langcodes.get(iso2_code).maximize().script
        # Get one representative codepoint for this script via langcodes
        # then check its Unicode block range from unicodedata
        from language_identifier.id_lang import _ISO15924_SAMPLE
        sample_cp   = _ISO15924_SAMPLE.get(iso15924)
        if sample_cp is None:
            return True   # can't verify — allow through
        sample_name = unicodedata.name(chr(sample_cp))
        script_name = sample_name.split()[0]   # e.g. "MALAYALAM"
    except Exception:
        return True
    for ch in word:
        cat = unicodedata.category(ch)
        if cat not in _VALID_CATS:
            return False
        try:
            ch_script = unicodedata.name(ch).split()[0]
            if ch_script != script_name:
                return False
        except ValueError:
            return False
    return True

def _build_word_list_from_ud(iso2_code: str, dest: Path) -> bool:
    """
    Fallback word source for languages that hermitdave/FrequencyWords has
    no list for (e.g. Marathi, as of the 2018 OpenSubtitles batch).
    Derives a real word-frequency list directly from the token stream of
    the same Universal Dependencies treebank used for sentences: every
    FORM field in a CoNLL-U file is a real, attested word in that
    language, annotated by linguists — not a synthetic or hardcoded word.
    Multiple treebank files are merged (not just one split) to build a
    reasonably sized vocabulary, since UD treebanks are much smaller than
    the OpenSubtitles-derived hermitdave lists.
    Writes results to `dest` in the same "word<space>freq" format the
    hermitdave lists use, so load_word_list()'s parsing is unchanged.
    """
    candidates = discover_ud_treebank_url(iso2_code)
    if not candidates:
        return False
    counts      = Counter()
    files_used  = 0
    tmp         = CORPORA_DIR / f"{iso2_code}_wordsrc.conllu"
    for url, label in candidates:
        if files_used >= UD_FALLBACK_FILES:
            break
        print(f"  Trying UD token source: {label}...")
        if not _download(url, tmp):
            continue
        content = tmp.read_text(encoding="utf-8")
        tmp.unlink(missing_ok=True)
        for line in content.splitlines():
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue  # skip multiword-token ranges and empty nodes
            form = cols[1].strip()
            if form:
                counts[form] += 1
        files_used += 1
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    if not counts:
        return False
    lines = [f"{w} {c}" for w, c in counts.most_common()]
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ Derived {len(lines)} candidate words from "
          f"{files_used} UD treebank file(s)")
    return True

def load_word_list(iso2_code: str) -> list[str]:
    """
    Downloads (once) and returns a list of the most common real words
    in iso2_code's language. Primary source is the hermitdave frequency
    word list; if that language has no list there, a word-frequency
    list is derived from the UD treebank's own tokens instead (see
    _build_word_list_from_ud). Either way, words are filtered to the
    correct script and a minimum corpus frequency.
    """
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    dest = CORPORA_DIR / f"{iso2_code}_words.txt"
    if not (dest.exists() and dest.stat().st_size > 0):
        urls = get_word_frequency_urls(iso2_code)
        downloaded = False
        for url in urls:
            print(f"  Trying word list: {url}")
            if _download(url, dest):
                downloaded = True
                break
        if not downloaded:
            print(f"  No frequency list found on hermitdave/FrequencyWords "
                  f"for '{iso2_code}'.")
            print("  Deriving a word-frequency list from the UD treebank "
                  "token stream instead...")
            if not _build_word_list_from_ud(iso2_code, dest):
                sys.exit(
                    f"[ERROR] Could not obtain a word list for '{iso2_code}' "
                    "from either hermitdave/FrequencyWords or the UD "
                    "treebank.\n        Check your internet connection."
                )

    def _filter(min_freq: int) -> list[str]:
        out = []
        with open(dest, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) != 2:
                    continue
                word, freq_str = parts
                try:
                    freq = int(freq_str)
                except ValueError:
                    continue
                if freq < min_freq:
                    continue
                if not (MIN_WORD_CHARS <= len(word) <= MAX_WORD_CHARS):
                    continue
                if not _is_script_pure(word, iso2_code):
                    continue
                out.append(word)
        return out

    words = _filter(MIN_WORD_FREQ)
    # UD-derived fallback corpora are far smaller than the OpenSubtitles-
    # based hermitdave lists, so few (or no) words may clear the same
    # frequency floor. Progressively relax the floor — never the script
    # or length filters — until a usable pool exists. hermitdave-backed
    # languages (ml, ta, hi, bn) already clear MIN_WORD_FREQ on the first
    # pass, so this loop never runs for them and their output is unchanged.
    floor = MIN_WORD_FREQ
    while len(words) < MIN_WORD_POOL and floor > 1:
        floor = max(1, floor // 2)
        words = _filter(floor)
    return words
# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE CORPUS — real annotated sentences from Universal Dependencies
# ─────────────────────────────────────────────────────────────────────────────
def _strip_format_chars(text: str) -> str:
    """
    Removes Unicode format-control characters (General Category "Cf" —
    e.g. zero-width joiner/non-joiner, directional marks) from text.

    These exist to guide complex-script SHAPING ENGINES — e.g. telling
    one which of two adjacent letters should merge into a conjunct
    ligature. This pipeline draws text with PIL/FreeType directly, with
    no shaping engine in between, so these controls are never consumed
    the way they're meant to be. Left in, they're instead looked up in
    the font's cmap like any other character; since fonts frequently
    don't map an explicit glyph for them at all, the result is a visible
    "tofu" box in the middle of otherwise-correct text instead of no
    visible effect. They carry no information once shaping is out of
    the picture, so removing them doesn't change the readable text —
    only prevents a literal box from standing in for an invisible
    control character.

    The word list never needs this explicitly: _is_script_pure already
    requires every character in a word to be a letter or combining mark
    (categories Lo/Mn/Mc), which excludes Cf characters as a side
    effect. Raw UD sentence text has no such filter, so this is applied
    directly to sentences here.
    """
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")

def _parse_conllu_sentences(content: str) -> list[str]:
    """
    Extracts the '# text = ...' lines from a CoNLL-U file.
    These are the actual full sentences written by linguists.
    """
    sentences = []
    for line in content.splitlines():
        if line.startswith("# text = "):
            sent = _strip_format_chars(line[9:].strip())
            if sent:
                sentences.append(sent)
    return sentences

def load_sentence_list(iso2_code: str) -> list[str]:
    """
    Downloads (once) and returns a list of real sentences in iso2_code's
    language from the Universal Dependencies treebank.

    Format-character cleaning (_strip_format_chars) is applied here on
    EVERY load — including cache hits — not just at parse time. A
    sentence cache written before that cleaning existed would otherwise
    keep serving un-stripped text forever, since the cache-hit path
    never re-parses anything. Re-cleaning on every read makes an old
    cache self-heal the first time it's loaded, with no manual cache
    deletion required.
    """
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    dest = CORPORA_DIR / f"{iso2_code}_sentences.txt"
    if dest.exists() and dest.stat().st_size > 0:
        cached = dest.read_text(encoding="utf-8").splitlines()
        return [s for s in (_strip_format_chars(line) for line in cached) if s]
    candidates = discover_ud_treebank_url(iso2_code)
    for url, label in candidates:
        print(f"  Trying sentence corpus: {label}...")
        raw_dest = CORPORA_DIR / f"{iso2_code}_raw.conllu"
        if _download(url, raw_dest):
            content   = raw_dest.read_text(encoding="utf-8")
            sentences = _parse_conllu_sentences(content)
            raw_dest.unlink()
            if sentences:
                dest.write_text("\n".join(sentences), encoding="utf-8")
                print(f"  ✓ {len(sentences)} real sentences loaded")
                return sentences
    sys.exit(
        f"[ERROR] Could not load sentence corpus for '{iso2_code}'.\n"
        "        Check your internet connection.\n"
        f"        Tried: {[u for u,_ in candidates]}"
    )
# ─────────────────────────────────────────────────────────────────────────────
# RENDER SAFETY  (guarantee no unrenderable "tofu" glyphs reach an image)
# ─────────────────────────────────────────────────────────────────────────────
# Unicode General Categories that NEVER need a visible glyph, so they're
# skipped by the renderability check below. Deliberately narrow (just
# whitespace and invisible format/control characters) rather than a
# broad allow-list of "the categories we expect text to use" — a script's
# own punctuation (e.g. the Devanagari/Indic danda "।" used to end
# Gujarati and Bengali sentences) is category "Po", not a letter, but it
# absolutely still needs a real glyph or it renders as a stray box.
_SKIP_RENDER_CHECK_CATS = {"Zs", "Zl", "Zp", "Cc", "Cf"}

def _glyph_has_ink(font, glyph_name: str) -> bool:
    """
    Returns True if the named glyph actually has drawable outline data,
    rather than being visually blank. Glyph index 0 (.notdef) is not the
    only way a font can fail to really support a character: some fonts
    map a codepoint to a real, non-zero glyph index that is nonetheless
    an empty placeholder. Checking for actual contour/component data is
    what catches that case too.
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
            return len(charstrings[glyph_name].bytecode) > 4
    except Exception:
        pass
    return True  # can't verify this glyph specifically — don't block on it

def load_font_coverage(font_path: str) -> set[int] | None:
    """
    Returns the set of Unicode codepoints font_path can ACTUALLY render
    a visible glyph for, loaded once per run. This is the ground truth
    used to filter corpus content before rendering: font DISCOVERY
    (discover_font_for_script) only ever estimates coverage from a
    sampled script block, which is necessarily a heuristic. This
    function inspects the exact font that was actually chosen.

    Uses getBestCmap(), which applies the same subtable priority order
    real text renderers (FreeType/HarfBuzz/PIL) use to pick "the" cmap,
    rather than naively merging every subtable a font carries — some
    fonts include extra subtables (legacy Mac tables, symbol encodings)
    that don't reflect what actually gets used for standard Unicode
    text, which can make a raw merge report false coverage. Each
    candidate glyph is then checked for real ink via _glyph_has_ink.

    Returns None if the font can't be inspected (e.g. fontTools missing
    or an unusual font format); callers should then skip filtering
    rather than block on it. This is printed loudly rather than swallowed
    silently — a silent fallback here would mean the render-safety
    guarantee this whole function exists for just quietly stops applying,
    with no signal that "tofu" boxes might now reach the output.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        print("  [WARNING] fontTools is not installed — cannot verify "
              "glyph coverage for the chosen font.")
        print("            Render-safety filtering is DISABLED: "
              "unrenderable characters may appear as boxes.")
        print("            Fix: pip install fonttools --break-system-packages")
        return None
    try:
        font = TTFont(font_path, lazy=True, fontNumber=0)
        best = font.getBestCmap()
        if not best:
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
        print(f"  [WARNING] Could not inspect font '{font_path}' for glyph "
              f"coverage ({type(e).__name__}: {e}).")
        print("            Render-safety filtering is DISABLED for this "
              "run: unrenderable characters may appear as boxes.")
        return None

def _is_renderable(text: str, covered: set[int] | None) -> bool:
    """
    True if every character in `text` that needs a visible glyph — i.e.
    everything except whitespace/format/control characters, see
    _SKIP_RENDER_CHECK_CATS — is present in `covered`. This deliberately
    includes punctuation: a script's own sentence-ending marks need
    glyphs just as much as its letters do. If `covered` is None
    (coverage couldn't be loaded), everything is treated as renderable
    rather than blocking all content.
    """
    if covered is None:
        return True
    for ch in text:
        if unicodedata.category(ch) in _SKIP_RENDER_CHECK_CATS:
            continue
        if ord(ch) not in covered:
            return False
    return True

def filter_renderable(items: list[str], covered: set[int] | None) -> list[str]:
    """
    Keeps only the items the actually-chosen font can render in full.
    This is the hard guarantee behind "no boxes in the output": rather
    than trusting font discovery's script-level coverage estimate, every
    individual word/sentence is checked against the font's real cmap
    before it ever becomes eligible for selection.
    """
    if covered is None:
        return items
    return [item for item in items if _is_renderable(item, covered)]
def build_font_coverage_pool(iso2_code: str) -> dict[str, set[int]]:
    """
    Returns {font_path: covered_codepoints} for every font fontconfig
    reports as supporting iso2_code's language AND that fontTools could
    actually verify (see discover_renderable_fonts's docstring for why
    unverifiable fonts are excluded, not trusted).

    Computed once and reused across MANY different pieces of text — the
    expensive part of discover_renderable_fonts is enumerating and
    inspecting every candidate font, which doesn't need repeating for
    every word when generating a large, textually-diverse dataset (see
    --augment mode). discover_renderable_fonts itself is now a thin
    wrapper around this plus fonts_for_text, for the single-fixed-text
    --replicate case where building a whole pool for one lookup isn't
    worth a separate code path.
    """
    pool = {}
    for path in discover_all_fonts_for_script(iso2_code):
        covered = load_font_coverage(path)
        if covered is not None:
            pool[path] = covered
    return pool
def fonts_for_text(text: str, pool: dict[str, set[int]]) -> list[str]:
    """Filters a pre-built font coverage pool down to fonts that can render `text` in full."""
    return [path for path, covered in pool.items() if _is_renderable(text, covered)]
def discover_renderable_fonts(iso2_code: str, text: str) -> list[str]:
    """
    Returns every installed font that can render `text` in full — used
    by --replicate mode, where visual VARIETY (many different fonts)
    across renders of one fixed word/letter is the goal, unlike the
    single best font discover_font_for_script picks for prose.

    Combines fc-list's language-candidate list (discover_all_fonts_for_
    script) with the exact same glyph-coverage machinery already used
    to keep tofu out of the corpus-based modes (load_font_coverage +
    _is_renderable) — but checked against the literal target string
    itself, which is the most precise possible filter: fontconfig's
    "supports this language" metadata is a guess (as seen with
    discover_font_for_script's own fallback logic), the actual cmap of
    the actual font is not.

    Unlike the single-font content-filtering path elsewhere (where
    "couldn't verify this font" fails OPEN, since there's only one
    chosen font and blocking everything would be worse), a font that
    can't be inspected here is EXCLUDED rather than trusted blindly.
    fc-list can report legacy PostScript Type1 (.pfb) fonts as
    supporting a language; fontTools can't parse those to verify real
    coverage (raises TTLibError), and PIL/FreeType may still render
    them for SOME text without actually having full coverage for it —
    since this function is choosing among many candidates, not falling
    back on a single one, there's no good reason to include a font we
    couldn't actually verify when other, verifiable fonts exist.
    """
    return fonts_for_text(text, build_font_coverage_pool(iso2_code))
# ─────────────────────────────────────────────────────────────────────────────
# CONTENT SELECTION  (random samples from real corpora)
# ─────────────────────────────────────────────────────────────────────────────
def pick_word(words: list[str], rng: random.Random) -> str:
    return rng.choice(words)

def pick_sentence(sentences: list[str], rng: random.Random) -> str:
    return rng.choice(sentences)

def pick_paragraph(sentences: list[str], rng: random.Random) -> dict:
    """
    A paragraph = title (1 sentence, used as heading) + subtitle
    (1 shorter sentence) + body (PARA_SIZE consecutive sentences).
    All sourced from the real sentence corpus.
    """
    # Shuffle a pool to get variety
    pool = sentences[:]
    rng.shuffle(pool)
    # Title: pick a short sentence (≤ 5 words) if available, else any
    short = [s for s in pool if len(s.split()) <= 5]
    title = rng.choice(short) if short else rng.choice(pool)
    # Subtitle: a sentence different from title
    remaining = [s for s in pool if s != title]
    subtitle  = rng.choice(remaining) if remaining else rng.choice(pool)
    # Body: PARA_SIZE consecutive sentences from a random position
    idx   = rng.randint(0, max(0, len(sentences) - PARA_SIZE))
    body_sents = sentences[idx : idx + PARA_SIZE]
    body  = " ".join(body_sents)
    return {"title": title, "subtitle": subtitle, "body": body}
# ─────────────────────────────────────────────────────────────────────────────
# RENDERING  (vector font, no pixel breaking)
# ─────────────────────────────────────────────────────────────────────────────
def _wrap(text: str, font, max_w: int, draw) -> list[str]:
    words, lines, cur = text.split(" "), [], []
    for w in words:
        trial = " ".join(cur + [w])
        if draw.textbbox((0, 0), trial, font=font)[2] > max_w and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]

def _lh(font, draw) -> int:
    """
    Returns the true line height for this font using its own metrics
    (ascent + descent), not a text-measurement probe.
    Using draw.textbbox("Ag") as a probe was a bug: Tamil/Malayalam
    glyphs with above-base or below-base vowel signs are taller than
    Latin letters in the same font, so 'Ag' underestimated the height
    by ~5-12px and caused multi-line text to overlap.
    font.getmetrics() returns the correct typographic ascent + descent
    for the font regardless of what script is being rendered.
    """
    ascent, descent = font.getmetrics()
    return ascent + descent + LINE_SPACING

def render_word(text: str, font_size: int,
                font_path: str, out: Path) -> tuple[int, int]:
    font  = ImageFont.truetype(font_path, font_size)
    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=font)
    w     = max(bbox[2] - bbox[0] + PAD * 2, 150)
    h     = max(bbox[3] - bbox[1] + PAD * 2, 80)
    img   = Image.new("RGB", (w, h), "white")
    draw  = ImageDraw.Draw(img)
    draw.text((PAD - bbox[0], PAD - bbox[1]), text, font=font, fill="black")
    img.save(str(out), "PNG")
    return w, h

def render_sentence(text: str, font_size: int,
                    font_path: str, out: Path) -> tuple[int, int]:
    font  = ImageFont.truetype(font_path, font_size)
    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    lines = _wrap(text, font, MAX_IMG_WIDTH - PAD * 2, draw)
    lh    = _lh(font, draw)
    w     = min(
        max(draw.textbbox((0, 0), l, font=font)[2] for l in lines) + PAD * 2,
        MAX_IMG_WIDTH
    )
    h     = max(len(lines) * lh + PAD * 2, 80)
    img   = Image.new("RGB", (w, h), "white")
    draw  = ImageDraw.Draw(img)
    y = PAD
    for line in lines:
        draw.text((PAD, y), line, font=font, fill="black")
        y += lh
    img.save(str(out), "PNG")
    return w, h

def render_paragraph(parts: dict, font_path: str,
                      rng: random.Random, out: Path) -> dict:
    body_sz = rng.randint(*PARA_BODY_RANGE)
    sub_sz  = body_sz + rng.randint(*PARA_SUB_EXTRA)
    ttl_sz  = sub_sz  + rng.randint(*PARA_TTL_EXTRA)
    f_t = ImageFont.truetype(font_path, ttl_sz)
    f_s = ImageFont.truetype(font_path, sub_sz)
    f_b = ImageFont.truetype(font_path, body_sz)
    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    inner = MAX_IMG_WIDTH - PAD * 2
    t_lines = _wrap(parts["title"],    f_t, inner, draw)
    s_lines = _wrap(parts["subtitle"], f_s, inner, draw)
    b_lines = _wrap(parts["body"],     f_b, inner, draw)
    lh_t, lh_s, lh_b = _lh(f_t, draw), _lh(f_s, draw), _lh(f_b, draw)
    img_h = (PAD * 4
             + len(t_lines) * lh_t
             + len(s_lines) * lh_s
             + len(b_lines) * lh_b)
    img   = Image.new("RGB", (MAX_IMG_WIDTH, img_h), "white")
    draw  = ImageDraw.Draw(img)
    y = PAD
    for l in t_lines:
        draw.text((PAD, y), l, font=f_t, fill="black"); y += lh_t
    y += PAD // 2
    for l in s_lines:
        draw.text((PAD, y), l, font=f_s, fill="black"); y += lh_s
    y += PAD // 2
    for l in b_lines:
        draw.text((PAD, y), l, font=f_b, fill="black"); y += lh_b
    img.save(str(out), "PNG")
    return {"title_pt": ttl_sz, "subtitle_pt": sub_sz, "body_pt": body_sz,
            "width": MAX_IMG_WIDTH, "height": img_h}
def render_replicate_image(text: str, font_path: str, out: Path,
                            rng: random.Random) -> dict:
    """
    Renders one randomized variant of `text` (a single fixed word or
    letter) for --replicate mode: random font size, small rotation,
    padding, and near-white/near-black contrast jitter, using whichever
    font is passed in — the caller cycles through discover_renderable_
    fonts' results for variety across the whole batch.

    Text is drawn onto a separate transparent layer and rotated there,
    THEN composited onto an opaque background — rotating directly on an
    opaque canvas would either crop corners or introduce visible empty
    triangles at the canvas edges; compositing after rotation avoids
    both.
    """
    size  = rng.randint(*REPLICATE_FONT_SIZE_RANGE)
    angle = rng.uniform(*REPLICATE_ROTATION_DEG_RANGE)
    pad   = rng.randint(*REPLICATE_PADDING_RANGE)
    bg    = rng.randint(*REPLICATE_BG_LUMINANCE_RANGE)
    fg    = rng.randint(*REPLICATE_FG_LUMINANCE_RANGE)

    font  = ImageFont.truetype(font_path, size)
    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    layer = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(
        (pad - bbox[0], pad - bbox[1]), text, font=font,
        fill=(fg, fg, fg, 255)
    )
    rotated = layer.rotate(angle, expand=True, resample=Image.BICUBIC)

    canvas = Image.new("RGB", rotated.size, (bg, bg, bg))
    canvas.paste(rotated, (0, 0), rotated)
    canvas.save(str(out), "PNG")
    return {"width": canvas.width, "height": canvas.height,
            "font_size_pt": size, "rotation_deg": round(angle, 2)}
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Detect language, then generate real-word and real-sentence "
                    "images in that language."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",  metavar="TEXT")
    group.add_argument("--image", metavar="PATH")
    parser.add_argument("--words",      type=int, default=5)
    parser.add_argument("--sentences",  type=int, default=5)
    parser.add_argument("--paragraphs", type=int, default=5)
    parser.add_argument(
        "--replicate", type=int, default=0, metavar="N",
        help="Generate N randomized image variants of the exact --text "
             "string (a single word or letter) instead of sampling "
             "words/sentences/paragraphs from a corpus — e.g. for "
             "building an OCR/training dataset. Varies font, size, "
             "rotation, and contrast per image; writes a manifest.csv "
             "alongside the images."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, metavar="PATH",
        help="Directory to write generated images into (a subfolder "
             "named after the detected language is created inside it). "
             "Defaults to ./output/images relative to wherever this "
             "command is run from."
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Apply --replicate-style visual augmentation (font "
             "variety, rotation, contrast jitter) to the --words "
             "corpus-sampled word images too, and write a manifest.csv "
             "alongside them. Combines real word diversity (many "
             "different corpus words) with visual diversity (many "
             "renders per word) — the combination an OCR training "
             "dataset actually needs, which neither plain --words nor "
             "--replicate alone provides on its own. Has no effect on "
             "--sentences/--paragraphs, which stay single-font as "
             "before, or on --replicate mode itself."
    )
    args = parser.parse_args()
    # ── Step 0: render-safety verification requires fontTools ─────────
    # This pipeline's entire guarantee against invisible "tofu" boxes in
    # output depends on being able to inspect a font's real glyph
    # coverage (see the RENDER SAFETY section). Without fontTools, every
    # check that depends on it silently no-ops and trusts its input
    # instead — which reintroduces exactly the bug this pipeline exists
    # to prevent, with no error message pointing at why. Checking here,
    # before any downloads start, turns that silent failure mode into
    # an immediate, actionable one.
    if not FONTTOOLS_AVAILABLE:
        sys.exit(
            "[ERROR] The 'fontTools' package is required but not installed.\n"
            "        Without it, this pipeline cannot verify that a chosen "
            "font actually has\n"
            "        real glyphs for the words/sentences it's about to "
            "render — the exact\n"
            "        check that prevents invisible \"tofu\" boxes in "
            "output.\n"
            "        Install it, then re-run:\n"
            "          pip install fonttools --break-system-packages"
        )
    # langcodes' display_name()/script_name() -- used throughout font
    # discovery and treebank discovery -- lazily require the separate
    # 'language_data' package. Missing it previously surfaced as a
    # misleading "no font found for this script" error for EVERY
    # language, not just one, with no hint the real problem was a
    # missing dependency. Checking here catches that at the source.
    if not LANGUAGE_DATA_AVAILABLE:
        sys.exit(
            "[ERROR] The 'language_data' package is required but not "
            "installed.\n"
            "        langcodes needs it to resolve language/script names "
            "-- without it, font\n"
            "        discovery and treebank discovery fail with misleading "
            "errors that look\n"
            "        unrelated to the real cause.\n"
            "        Install it, then re-run:\n"
            "          pip install language_data"
        )
    # ── Step 1: detect language ───────────────────────────────────────
    tess_langs       = get_tesseract_langs()
    langdetect_codes = get_langdetect_supported_codes()
    _, not_covered   = build_coverage_info(tess_langs, langdetect_codes)
    script_map       = build_script_to_language_map(langdetect_codes)
    tess_lang_str    = "+".join(tess_langs)
    if args.image:
        print(f"Running OCR on  : {args.image}")
        input_text = extract_text_from_image(args.image, tess_lang_str)
        if not input_text:
            sys.exit("[ERROR] No text extracted from image.")
        print(f"Extracted text  : {input_text}")
    else:
        input_text = args.text
    result = detect_language_from_text(input_text, not_covered, script_map)
    if result["language"] is None:
        sys.exit(f"[ERROR] Could not detect language: {result['warning']}")
    lang_code = result["code"]
    lang_name = result["language"]
    print(f"Detected        : {lang_name} ({lang_code})")
    print(f"Confidence      : {result['confidence']:.1%}")
    # ── Resolve where output goes: caller's choice, or the default ─────
    # --output-dir lets the person running this tool decide where
    # generated files land, rather than it always being CWD-relative.
    # Every run's images are organized into a subfolder named after the
    # DETECTED language (e.g. "malayalam"), not the raw ISO code, so
    # output from different languages never mixes in one flat folder.
    base_out_dir = (Path(args.output_dir).expanduser().resolve()
                     if args.output_dir else DEFAULT_OUT_DIR)
    lang_dir = base_out_dir / _safe_dirname(lang_name)
    lang_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder   : {lang_dir}/\n")
    # ── Replicate mode: many varied renders of ONE fixed word/letter ───
    # Diverges from the corpus-based flow entirely: no word list or
    # sentence corpus is needed since the text is already given. Font
    # selection also differs on purpose — discover_renderable_fonts
    # returns EVERY usable font (for variety across the dataset), not
    # discover_font_for_script's single best pick for prose.
    if args.replicate > 0:
        print(f"\nDiscovering fonts that can render '{input_text}' in "
              f"full...")
        font_paths = discover_renderable_fonts(lang_code, input_text)
        if not font_paths:
            sys.exit(
                f"[ERROR] No installed font can fully render "
                f"'{input_text}'.\n"
                "        Install a font covering this script and re-run "
                "— e.g. sudo pacman -S noto-fonts-extra, or\n"
                "        sudo apt install fonts-noto-extra."
            )
        preview = ", ".join(os.path.basename(p) for p in font_paths[:5])
        more    = ", ..." if len(font_paths) > 5 else ""
        print(f"  ✓ {len(font_paths)} usable font(s) found: {preview}{more}\n")

        replicate_dir = lang_dir / "replicate"
        replicate_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = replicate_dir / "manifest.csv"
        rng = random.Random()

        print(f"Generating {args.replicate} image(s) of '{input_text}':")
        report_every = max(1, args.replicate // 20)
        with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
            writer = csv.writer(mf)
            writer.writerow(["filename", "label", "font", "width", "height",
                              "font_size_pt", "rotation_deg"])
            for i in range(1, args.replicate + 1):
                font_path = rng.choice(font_paths)
                fname = f"{i:06d}.png"
                meta = render_replicate_image(
                    input_text, font_path, replicate_dir / fname, rng
                )
                writer.writerow([
                    fname, input_text, os.path.basename(font_path),
                    meta["width"], meta["height"],
                    meta["font_size_pt"], meta["rotation_deg"],
                ])
                if i % report_every == 0 or i == args.replicate:
                    pct = i / args.replicate * 100
                    print(f"\r  [{i}/{args.replicate}]  {pct:5.1f}%",
                          end="", flush=True)
        print(f"\n\nDone — {args.replicate} images + manifest.csv saved "
              f"to: {replicate_dir}/")
        return
    # ── Step 2: discover font ─────────────────────────────────────────
    font_path = discover_font_for_script(lang_code)
    print(f"Font            : {font_path}\n")
    # ── Step 3: load real words and real sentences ────────────────────
    print("Loading word frequency list...")
    words = load_word_list(lang_code)
    if not words:
        sys.exit("[ERROR] No words loaded from frequency list.")
    print(f"  ✓ {len(words)} real common words loaded\n")
    print("Loading real sentence corpus...")
    sentences = load_sentence_list(lang_code)
    if not sentences:
        sys.exit("[ERROR] No sentences loaded from corpus.")
    print(f"  ✓ {len(sentences)} real grammatical sentences loaded\n")
    # ── Step 3b: keep only content the chosen font can fully render ────
    # Font discovery already ranks candidates by estimated script
    # coverage, but that's a heuristic based on a sampled block. This is
    # the hard guarantee: check the ACTUAL font against the ACTUAL text,
    # so no unrenderable "tofu" glyph can reach a final image.
    coverage = load_font_coverage(font_path)
    if coverage is None:
        print("Render-safety filtering was skipped (see warning above) — "
              "output may still contain unrenderable characters.\n")
    words_r     = filter_renderable(words, coverage)
    sentences_r = filter_renderable(sentences, coverage)
    if coverage is not None:
        dropped_w = len(words) - len(words_r)
        dropped_s = len(sentences) - len(sentences_r)
        if dropped_w or dropped_s:
            print(f"Font coverage check: dropped {dropped_w} word(s) and "
                  f"{dropped_s} sentence(s) the font '{font_path}' can't "
                  f"fully render.")
    if not words_r:
        sys.exit(
            f"[ERROR] The chosen font ('{font_path}') cannot fully render "
            f"any of the {len(words)} loaded words for '{lang_code}'.\n"
            "        Install a font with real coverage for this script "
            "(see the font-discovery error message format)\n"
            "        and re-run — e.g. sudo pacman -S noto-fonts-extra, or "
            "sudo apt install fonts-noto-extra."
        )
    if not sentences_r:
        sys.exit(
            f"[ERROR] The chosen font ('{font_path}') cannot fully render "
            f"any of the {len(sentences)} loaded sentences for "
            f"'{lang_code}'.\n"
            "        Install a font with real coverage for this script "
            "and re-run."
        )
    words, sentences = words_r, sentences_r
    print(f"  ✓ {len(words)} words / {len(sentences)} sentences confirmed "
          f"fully renderable by the chosen font\n")
    rng = random.Random()
    # ── Step 4a: word images ──────────────────────────────────────────
    if args.augment:
        # Build the font pool once, up front -- this is the expensive
        # part (enumerating + inspecting every candidate font), and
        # doing it once and reusing it across many different words is
        # what makes this practical for training-dataset-scale counts,
        # instead of re-running font discovery per word.
        print("Building font pool for augmented word generation...")
        font_pool = build_font_coverage_pool(lang_code)
        # The single font already chosen above is guaranteed to render
        # every word in `words` (that's what the filtering above just
        # confirmed) -- keeping it in the pool guarantees fonts_for_text
        # below is never empty, even for a word no OTHER pooled font
        # happens to fully cover.
        if coverage is not None:
            font_pool.setdefault(font_path, coverage)
        print(f"  ✓ {len(font_pool)} font(s) available for augmentation\n")
        manifest_path = lang_dir / "manifest.csv"
        print(f"Generating {args.words} augmented word image(s):")
        report_every = max(1, args.words // 20)
        with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
            writer = csv.writer(mf)
            writer.writerow(["filename", "label", "font", "width", "height",
                              "font_size_pt", "rotation_deg"])
            for i in range(1, args.words + 1):
                word = pick_word(words, rng)
                candidate_fonts = fonts_for_text(word, font_pool)
                chosen_font = rng.choice(candidate_fonts)
                fname = f"word_{i:06d}.png"
                meta = render_replicate_image(
                    word, chosen_font, lang_dir / fname, rng
                )
                writer.writerow([
                    fname, word, os.path.basename(chosen_font),
                    meta["width"], meta["height"],
                    meta["font_size_pt"], meta["rotation_deg"],
                ])
                if i % report_every == 0 or i == args.words:
                    pct = i / args.words * 100
                    print(f"\r  [{i}/{args.words}]  {pct:5.1f}%",
                          end="", flush=True)
        print(f"\n  ✓ manifest.csv written to {manifest_path}\n")
    else:
        print(f"Generating {args.words} word image(s):")
        for i in range(1, args.words + 1):
            word  = pick_word(words, rng)
            size  = rng.randint(*WORD_FONT_RANGE)
            fname = f"word_{i:02d}.png"
            w, h  = render_word(word, size, font_path, lang_dir / fname)
            print(f"  {i:2d}. [{size:2d}pt] {word:<25}  {fname}  ({w}×{h}px)")
    # ── Step 4b: sentence images ──────────────────────────────────────
    print(f"\nGenerating {args.sentences} sentence image(s):")
    for i in range(1, args.sentences + 1):
        sent  = pick_sentence(sentences, rng)
        size  = rng.randint(*SENT_FONT_RANGE)
        fname = f"sentence_{i:02d}.png"
        w, h  = render_sentence(sent, size, font_path, lang_dir / fname)
        print(f"  {i:2d}. [{size:2d}pt] {sent[:55]:<55}  {fname}")
    # ── Step 4c: paragraph images ─────────────────────────────────────
    print(f"\nGenerating {args.paragraphs} paragraph image(s):")
    for i in range(1, args.paragraphs + 1):
        parts = pick_paragraph(sentences, rng)
        fname = f"paragraph_{i:02d}.png"
        meta  = render_paragraph(parts, font_path, rng, lang_dir / fname)
        print(f"  {i:2d}. title=[{meta['title_pt']}pt] "
              f"subtitle=[{meta['subtitle_pt']}pt] "
              f"body=[{meta['body_pt']}pt]  {fname}")
        print(f"      Title    : {parts['title']}")
        print(f"      Subtitle : {parts['subtitle']}")
        print(f"      Body     : {parts['body'][:70]}...")
    total = args.words + args.sentences + args.paragraphs
    print(f"\nDone — {total} images saved to: {lang_dir}/")
if __name__ == "__main__":
    main()
