# Introduction

`language_identifier` detects the language of raw text or an image, then
generates real-word, real-sentence, and real-paragraph images in the
detected script — sourced live from public frequency-word and treebank
corpora, with no per-language word lists or sentence data hardcoded
into the package. It can also generate large batches of randomized
image variants of a single fixed word or letter (`--replicate`), for
building OCR/training datasets.

-----

## Features

- Detects language from plain text or from an image (via OCR).
- Sources real common words from frequency word lists, and real
  grammatical sentences from Universal Dependencies treebanks,
  discovered and downloaded at runtime per detected language.
- Verifies the chosen font can actually render the selected content
  before generating any image, to avoid missing-glyph ("tofu") output.
- `--replicate N` mode: generates N visually varied renders (font,
  size, rotation, contrast) of one fixed word/letter, with a
  `manifest.csv` labeling each image — for OCR/training datasets.

-----

## Installation

```bash
# create a virtual environment
python3 -m venv venv
source venv/bin/activate

# build the wheel (requires setup.py locally -- see note below)
pip install wheel build
python -m build --wheel

# install the built wheel
pip install dist/language_identifier-0.1.0-py3-none-any.whl
```

> **Note:** `setup.py` is required to build the wheel but is not
> committed to version control (see `.gitignore`) — keep a local copy
> alongside this readme.

-----

## Usage

### As an installed CLI

```bash
id-lang --text "ভাষা শনাক্তকরণ একটি গুরুত্বপূর্ণ কাজ।"

gen-para --text "ভাষা শনাক্তকরণ একটি গুরুত্বপূর্ণ কাজ।" \
         --words 5 --sentences 5 --paragraphs 5

# choose where output goes instead of the default ./output/images
gen-para --text "বাংলা" --words 5 --output-dir ~/datasets/bengali

# OCR/training dataset, one fixed word/letter, many visual variants:
gen-para --text "বাংলা" --replicate 5000

# OCR/training dataset, MANY different real words, each with visual
# variety too (font/rotation/contrast) -- writes manifest.csv:
gen-para --text "தமிழ்" --words 5000 --sentences 0 --paragraphs 0 --augment
```

### As an importable library

```python
from language_identifier import detect_language_from_text
from language_identifier.id_lang import (
    get_tesseract_langs,
    get_langdetect_supported_codes,
    build_coverage_info,
    build_script_to_language_map,
)

tess_langs       = get_tesseract_langs()
langdetect_codes = get_langdetect_supported_codes()
_, not_covered   = build_coverage_info(tess_langs, langdetect_codes)
script_map       = build_script_to_language_map(langdetect_codes)

result = detect_language_from_text("বাংলা", not_covered, script_map)
print(result["language"], result["code"], result["confidence"])
```

See `usage_language_identifier.py` for a complete runnable example.

-----

## Output format

By default, `gen-para` writes images to `./output/images/<language>/`
(relative to wherever you run the command) — a separate subfolder per
*detected* language (e.g. `output/images/malayalam/`,
`output/images/bengali/`), so output from different runs never mixes
in one flat folder. Pass `--output-dir PATH` to choose a different base
location entirely; the per-language subfolder is still created inside
whatever directory you point it at.

```
output/images/
├── malayalam/
│   ├── word_01.png
│   ├── sentence_01.png
│   └── paragraph_01.png
└── bengali/
    ├── word_01.png
    └── replicate/
        ├── 000001.png
        ├── 000002.png
        └── manifest.csv
```

`--replicate` mode writes its images into a `replicate/` subfolder
inside the language folder, alongside a `manifest.csv` with columns:
`filename, label, font, width, height, font_size_pt, rotation_deg`.

Downloaded word-frequency lists and sentence corpora are cached
per-user (not inside the installed package, and not affected by
`--output-dir`) so they're only downloaded once and reused across runs
regardless of where the package is installed:

- Linux: `$XDG_CACHE_HOME/language_identifier/corpora`, or
  `~/.cache/language_identifier/corpora` if that variable isn't set
- macOS: `~/Library/Caches/language_identifier/corpora`
- Windows: `%LOCALAPPDATA%\language_identifier\Cache\corpora`

-----

## Languages

Nothing about which languages this supports is hardcoded — `gen-para`
works with any language for which, at runtime, it can find: (1) a
word-frequency list on hermitdave/FrequencyWords *or* a Universal
Dependencies treebank to derive one from, (2) a UD treebank for
sentences, and (3) an installed font with real glyph coverage for that
script. New languages work automatically as those public sources add
them — no code changes needed.

The following have been explicitly tested end-to-end (language
detection → word/sentence/paragraph generation → verified non-tofu
output) over the course of this project:

| Language | Code | Script     |
|----------|------|------------|
| Malayalam| ml   | Malayalam  |
| Tamil    | ta   | Tamil      |
| Bengali  | bn   | Bengali    |
| Hindi    | hi   | Devanagari |
| Marathi  | mr   | Devanagari |
| Punjabi  | pa   | Gurmukhi   |
| Gujarati | gu   | Gujarati   |
| English  | en   | Latin      |
| Estonian | et   | Latin      |

Any other language is expected to work the same way, provided the
three runtime requirements above are met — if you hit a language that
doesn't, it's most likely a missing font on the system running it,
which `gen-para` will report clearly rather than silently producing
broken output.

-----

## Development

```bash
pip install -r requirements.txt
pytest
```
