from setuptools import setup, find_packages

setup(
    name="language-identifier",
    version="0.1.0",
    description=(
        "Detects the language of text/images and generates real-word, "
        "real-sentence, and OCR-training-style images in the detected "
        "script, sourced from live corpora with no per-language data "
        "hardcoded."
    ),
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.10",
    install_requires=[
        "langcodes[data]",
        "langdetect",
        "pytesseract",
        "Pillow",
        "fonttools",
    ],
    entry_points={
        "console_scripts": [
            "id-lang=language_identifier.id_lang:main",
            "gen-para=language_identifier.gen_para:main",
        ],
    },
)
