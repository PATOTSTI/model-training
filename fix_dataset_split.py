import os
import random
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"

SPLITS = {
    "train": DATASET / "train",
    "valid": DATASET / "valid",
    "test":  DATASET / "test",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
# TEST_RATIO  = remaining 10%

# ── Helpers ────────────────────────────────────────────────────────────────────
def ensure_dirs() -> None:
    for split_dir in SPLITS.values():
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)


def collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path | None]]:
    """Return (image_path, label_path_or_None) pairs from the given directories."""
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = labels_dir / (img.stem + ".txt")
        pairs.append((img, lbl if lbl.exists() else None))
    return pairs


def move_pair(img: Path, lbl: Path | None, dest_split: str) -> None:
    dest_img = SPLITS[dest_split] / "images" / img.name
    shutil.move(str(img), str(dest_img))
    if lbl is not None:
        dest_lbl = SPLITS[dest_split] / "labels" / lbl.name
        shutil.move(str(lbl), str(dest_lbl))


def count_images(split: str) -> int:
    folder = SPLITS[split] / "images"
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def fix_yaml() -> None:
    yaml_path = DATASET / "data.yaml"
    content = (
        "train: train/images\n"
        "val:   valid/images\n"
        "test:  test/images\n"
        "\n"
        "nc: 1\n"
        "names: ['oil-spill']\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"data.yaml updated at: {yaml_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("  Oil Spill Dataset — Train / Valid / Test Split")
    print("=" * 55)

    ensure_dirs()

    # ── Step 1: Collect all image+label pairs from train ─────────────────────
    train_img_dir = SPLITS["train"] / "images"
    train_lbl_dir = SPLITS["train"] / "labels"

    print(f"\nScanning: {train_img_dir}")
    pairs = collect_pairs(train_img_dir, train_lbl_dir)

    if not pairs:
        print("[ERROR] No images found in train/images/. Nothing to split.")
        return

    total = len(pairs)
    print(f"Total image-label pairs found: {total}")

    # ── Step 2: Shuffle and calculate split sizes ─────────────────────────────
    random.seed(42)          # reproducible splits
    random.shuffle(pairs)

    n_valid = max(1, round(total * VALID_RATIO))
    n_test  = max(1, round(total * (1.0 - TRAIN_RATIO - VALID_RATIO)))
    # train keeps the remainder (no files are touched for train — they stay)

    valid_pairs = pairs[:n_valid]
    test_pairs  = pairs[n_valid : n_valid + n_test]
    # train_pairs = pairs[n_valid + n_test:]  — already in the right place

    print(f"\nSplit plan:")
    print(f"  Train  : {total - n_valid - n_test} pairs  (stay in place)")
    print(f"  Valid  : {n_valid} pairs  (will be moved)")
    print(f"  Test   : {n_test} pairs  (will be moved)")

    # ── Step 3: Move valid pairs ──────────────────────────────────────────────
    print("\nMoving valid pairs …")
    for img, lbl in valid_pairs:
        move_pair(img, lbl, "valid")
    print(f"  Moved {len(valid_pairs)} pairs to valid/")

    # ── Step 4: Move test pairs ───────────────────────────────────────────────
    print("Moving test pairs …")
    for img, lbl in test_pairs:
        move_pair(img, lbl, "test")
    print(f"  Moved {len(test_pairs)} pairs to test/")

    # ── Step 5: Print summary ─────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("  Split Summary")
    print("─" * 40)
    print(f"  Train images : {count_images('train')}")
    print(f"  Valid images : {count_images('valid')}")
    print(f"  Test  images : {count_images('test')}")
    print("─" * 40)

    # ── Step 6: Fix data.yaml ─────────────────────────────────────────────────
    print("\nFixing data.yaml …")
    fix_yaml()

    print("\nDataset split complete. Ready to train!")


if __name__ == "__main__":
    main()
