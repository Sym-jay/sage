"""
prepare_dataset.py
===================
Combines the per-language manifest.csv files produced by
`gen-para ... --augment` (one per language subfolder under
output/images/) into a single, unified manifest ready for HuggingFace
`datasets` — with a train/validation/test split done PER LANGUAGE, not
globally.

Why per-language splitting matters: languages here have wildly
different sample counts (e.g. Gujarati's ~23 underlying words vs.
Malayalam's 15,000). A single global random split would let
high-volume languages dominate both the training data AND the
evaluation set, silently hiding how well low-volume languages actually
learned. Splitting within each language first guarantees every
language gets its own fair train/val/test slice.

No language list is hardcoded — every language subfolder containing a
manifest.csv under --images-dir is discovered and included
automatically, so this keeps working as more languages are added.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --languages malayalam,tamil,hindi
    python prepare_dataset.py --images-dir output/images --output dataset_manifest.csv
"""
import argparse
import csv
import random
from pathlib import Path


def find_language_manifests(images_root: Path) -> dict[str, Path]:
    """
    Returns {language_folder_name: path_to_manifest.csv} for every
    subfolder of images_root that has one. Discovered dynamically from
    whatever folders are actually present — nothing language-specific
    is hardcoded here.
    """
    result = {}
    for lang_dir in sorted(images_root.iterdir()):
        if not lang_dir.is_dir():
            continue
        manifest = lang_dir / "manifest.csv"
        if manifest.exists():
            result[lang_dir.name] = manifest
    return result


def load_rows(language: str, manifest_path: Path) -> list[dict]:
    """
    Reads one language's manifest.csv (written by --augment/--replicate
    mode) and returns rows with an absolute image path, so the combined
    manifest works regardless of where it's later read from.
    """
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = (manifest_path.parent / row["filename"]).resolve()
            rows.append({
                "image_path": str(image_path),
                "label": row["label"],
                "language": language,
                "font": row.get("font", ""),
            })
    return rows


def split_rows(rows: list[dict], train_frac: float, val_frac: float,
               seed: int) -> list[dict]:
    """
    Assigns a 'split' field (train/validation/test) to each row,
    shuffled and split independently WITHIN each language — see the
    module docstring for why this matters more than it might look like
    it does.
    """
    rng = random.Random(seed)
    by_language: dict[str, list[dict]] = {}
    for row in rows:
        by_language.setdefault(row["language"], []).append(row)
    for lang_rows in by_language.values():
        rng.shuffle(lang_rows)
        n = len(lang_rows)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        for i, row in enumerate(lang_rows):
            if i < n_train:
                row["split"] = "train"
            elif i < n_train + n_val:
                row["split"] = "validation"
            else:
                row["split"] = "test"
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-language manifest.csv files into one "
                    "unified, per-language-stratified train/val/test "
                    "manifest ready for HuggingFace `datasets`."
    )
    parser.add_argument(
        "--images-dir", type=str, default="output/images",
        help="Directory containing one subfolder per language, each "
             "with a manifest.csv (default: output/images)"
    )
    parser.add_argument(
        "--output", type=str, default="dataset_manifest.csv",
        help="Where to write the combined manifest CSV"
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--languages", type=str, default=None,
        help="Comma-separated language folder names to include "
             "(default: every language folder found with a manifest.csv)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_root = Path(args.images_dir)
    if not images_root.exists():
        raise SystemExit(f"[ERROR] {images_root} does not exist.")

    manifests = find_language_manifests(images_root)
    if args.languages:
        wanted = {l.strip() for l in args.languages.split(",")}
        missing = wanted - set(manifests)
        if missing:
            print(f"[WARNING] Requested language(s) not found (no "
                  f"manifest.csv): {', '.join(sorted(missing))}")
        manifests = {k: v for k, v in manifests.items() if k in wanted}

    if not manifests:
        raise SystemExit(
            f"[ERROR] No manifest.csv files found under {images_root}. "
            "Did you run gen-para with --augment or --replicate first?"
        )

    print(f"Found {len(manifests)} language(s): {', '.join(sorted(manifests))}\n")

    all_rows = []
    for language, manifest_path in manifests.items():
        rows = load_rows(language, manifest_path)
        print(f"  {language:12s} {len(rows):6d} images")
        all_rows.extend(rows)

    if args.train_frac + args.val_frac >= 1.0:
        raise SystemExit(
            f"[ERROR] --train-frac ({args.train_frac}) + --val-frac "
            f"({args.val_frac}) must leave room for a test split."
        )

    all_rows = split_rows(all_rows, args.train_frac, args.val_frac, args.seed)

    out_path = Path(args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "label", "language", "font", "split"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nTotal: {len(all_rows)} images across {len(manifests)} language(s)")
    for split in ["train", "validation", "test"]:
        count = sum(1 for r in all_rows if r["split"] == split)
        print(f"  {split:12s} {count:6d}")

    print(f"\nPer-language breakdown by split:")
    for language in sorted(manifests):
        counts = {s: 0 for s in ["train", "validation", "test"]}
        for r in all_rows:
            if r["language"] == language:
                counts[r["split"]] += 1
        print(f"  {language:12s} train={counts['train']:5d}  "
              f"val={counts['validation']:5d}  test={counts['test']:5d}")

    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
