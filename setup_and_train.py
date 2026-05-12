import os
import sys
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
DATASET  = ROOT / "dataset"
RUNS_DIR = ROOT / "runs"

SPLIT_MAP = {
    "train":      DATASET / "train",
    "valid":      DATASET / "valid",
    "validation": DATASET / "valid",
    "test":       DATASET / "test",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DATASET_FOLDERS = [
    DATASET / "train" / "images",
    DATASET / "train" / "labels",
    DATASET / "valid" / "images",
    DATASET / "valid" / "labels",
    DATASET / "test"  / "images",
    DATASET / "test"  / "labels",
]

# ── V3 config — change these if you want to tweak ─────────────────────────────
RESUME_WEIGHTS = r"F:\model-training\runs\2026-05-12_20-26-07\oil_detection_v2\weights\last.pt"
                        # Leave as None to start fresh from yolov8m.pt
RUN_NAME       = "oil_detection_v3"

# ─────────────────────────────────────────────────────────────────────────────
def banner(text: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)


# ── Step 1: Get zip path ──────────────────────────────────────────────────────
def get_zip_path() -> Path:
    if len(sys.argv) > 1:
        zip_path = Path(sys.argv[1])
    else:
        raw = input("Enter the full path to your Roboflow dataset zip file:\n> ").strip().strip('"')
        zip_path = Path(raw)

    if not zip_path.exists():
        print(f"[ERROR] File not found: {zip_path}")
        sys.exit(1)
    if zip_path.suffix.lower() != ".zip":
        print(f"[ERROR] Expected a .zip file, got: {zip_path.suffix}")
        sys.exit(1)

    print(f"[OK] Zip file found: {zip_path}")
    return zip_path


# ── Step 2: Extract and map folders ───────────────────────────────────────────
def extract_and_map(zip_path: Path) -> None:
    banner("STEP 2 — Extracting dataset")

    tmp_dir = ROOT / "_zip_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    print(f"Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)
    print("Extraction complete.")

    children = list(tmp_dir.iterdir())
    if len(children) == 1 and children[0].is_dir():
        effective_root = children[0]
        print(f"Detected single wrapper folder: '{effective_root.name}' — unwrapping.")
    else:
        effective_root = tmp_dir

    mapped_splits: set[str] = set()
    yaml_src: Path | None = None

    for item in effective_root.rglob("*"):
        if item.is_dir():
            key = item.name.lower()
            if key in SPLIT_MAP:
                dest_split = SPLIT_MAP[key]
                print(f"  Mapping '{item.relative_to(tmp_dir)}' → {dest_split.relative_to(ROOT)}")
                if dest_split.exists():
                    shutil.rmtree(dest_split)
                shutil.copytree(item, dest_split)
                mapped_splits.add(key.replace("validation", "valid"))
                continue

        if item.is_file() and item.name.lower() == "data.yaml" and yaml_src is None:
            yaml_src = item

    if yaml_src:
        dest_yaml = DATASET / "data.yaml"
        shutil.copy2(yaml_src, dest_yaml)
        print(f"  Copied data.yaml → {dest_yaml.relative_to(ROOT)}")
    else:
        print("  [WARN] No data.yaml found in zip — will create one from scratch.")

    shutil.rmtree(tmp_dir)

    missing = {"train", "valid", "test"} - mapped_splits
    if missing:
        print(f"\n[WARN] The following splits were not found in the zip: {missing}")
        print("       Make sure the zip contains 'train', 'valid'/'validation', and 'test' folders.")

    print("\nFolder mapping done.")


# ── Step 3: Fix data.yaml ──────────────────────────────────────────────────────
def fix_data_yaml() -> Path:
    banner("STEP 3 — Writing data.yaml")

    yaml_path = DATASET / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"train: train/images\n"
        f"val:   valid/images\n"
        f"test:  test/images\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['Oil-spill']\n"   # ← fixed capitalisation to match Roboflow
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"data.yaml written to: {yaml_path}")
    print(content)
    return yaml_path


# ── Step 4: Verify images ──────────────────────────────────────────────────────
def verify_dataset() -> bool:
    banner("STEP 4 — Verifying dataset")

    counts = {
        "train": count_images(DATASET / "train" / "images"),
        "valid": count_images(DATASET / "valid" / "images"),
        "test":  count_images(DATASET / "test"  / "images"),
    }

    print(f"  Train images found : {counts['train']}")
    print(f"  Valid images found : {counts['valid']}")
    print(f"  Test  images found : {counts['test']}")

    ok = True
    for split, n in counts.items():
        if n == 0:
            print(f"  [WARN] No images in '{split}/images' — check your zip structure.")
            ok = False

    if ok:
        print("\n[OK] All splits have images. Ready to train.")
    else:
        answer = input("\nSome splits are empty. Continue anyway? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(1)

    return ok


# ── Step 5 & 6: Train and report ───────────────────────────────────────────────
def train(yaml_path: Path, dated_dir: Path, run_name: str) -> None:
    banner("STEP 5 — Starting YOLOv8 Training")

    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("[ERROR] ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    gpu_available = torch.cuda.is_available()
    gpu_name      = torch.cuda.get_device_name(0) if gpu_available else None

    print("Device detection:")
    if gpu_available:
        print(f"  [GPU] {gpu_name} — CUDA ready")
    else:
        print("  [GPU] Not available (PyTorch has no CUDA support in this install)")
    print("  [CPU] Always available — much slower (2–6 hours for 100 epochs)")
    print()

    if gpu_available:
        choice = input("Select device — GPU (g) or CPU (c)? [g/c]: ").strip().lower()
        if choice == "c":
            device = "cpu"
            print("  Using CPU.")
        else:
            device = 0
            print(f"  Using GPU: {gpu_name}")
    else:
        print("  GPU is not available from this PyTorch build.")
        print("  To enable GPU, exit and run:")
        print("    pip uninstall torch torchvision torchaudio -y")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        print()
        print("  (c) Continue on CPU  — slow but works")
        print("  (x) Exit now         — fix PyTorch CUDA first (recommended)")
        print()
        choice = input("Your choice [c/x]: ").strip().lower()
        if choice == "x":
            print("\nExiting. Re-run this script after reinstalling PyTorch with CUDA support.")
            sys.exit(0)
        device = "cpu"
        print("  Continuing with CPU.")

    print()

    # ── Load model — resume from last.pt if available, else start fresh ───────
    if RESUME_WEIGHTS:
        resume_path = Path(RESUME_WEIGHTS)
        if resume_path.exists():
            print(f"Resuming from previous weights: {resume_path}")
            model_source = str(resume_path)
        else:
            print(f"[WARN] RESUME_WEIGHTS path not found: {resume_path}")
            print("       Falling back to fresh yolov8m.pt weights.")
            model_source = "yolov8m.pt"
    else:
        print("Loading YOLOv8m base model (fresh start) …")
        model_source = "yolov8m.pt"

    from ultralytics import YOLO
    model = YOLO(model_source)

    print("Launching training with the following config:")
    print(f"  data       : {yaml_path}")
    print(f"  epochs     : 150")
    print(f"  imgsz      : 640")
    print(f"  batch      : 16")
    print(f"  device     : {device}")
    print(f"  workers    : 2")
    print(f"  cache      : False")
    print(f"  patience   : 75")
    print(f"  optimizer  : AdamW")
    print(f"  lr0        : 0.0005")
    print(f"  name       : {run_name}")
    print()

    results = model.train(
        data=str(yaml_path),
        epochs=250,            # ← fresh start with bigger dataset needs more epochs
        imgsz=640,
        batch=16,
        device=device,
        workers=2,             # ← kept low — safe for your RAM
        cache=False,           # ← kept off — safe for your RAM
        patience=50,           # ← back to 50, fresh training doesn't need 75
        optimizer='AdamW',
        lr0=0.001,             # ← back to 0.001, fresh start needs higher lr
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        project=str(dated_dir),
        name=run_name,
        exist_ok=True,
    )

    # ── Step 6: Report results ────────────────────────────────────────────────
    banner("STEP 6 — Training Complete")

    save_dir = Path(results.save_dir)
    best_pt  = save_dir / "weights" / "best.pt"

    print(f"  Best model saved to : {best_pt}")

    map50 = None
    try:
        rd = results.results_dict
        for key in ("metrics/mAP50(B)", "metrics/mAP_50(B)", "metrics/mAP50"):
            if key in rd:
                map50 = rd[key]
                break
    except Exception:
        pass

    if map50 is not None:
        print(f"  Final mAP50         : {map50:.4f}  ({map50 * 100:.2f}%)")
    else:
        print(f"  Final mAP50         : (see runs/detect/{run_name}/results.csv for metrics)")

    print()
    print("=" * 60)
    print("  Training is complete! Your best model is ready to use.")
    print("=" * 60)

    if best_pt.exists():
        print(f"\nTo run predictions:\n"
              f"  cd scripts\n"
              f"  python predict.py <path_to_image>")
    else:
        print(f"\n[WARN] best.pt not found at expected path: {best_pt}")
        print("       Check the runs folder manually.")

    rename_best_model(save_dir)


# ── Step 7: Read results.csv, rename best.pt, print final summary ─────────────
def rename_best_model(save_dir: Path) -> None:
    banner("STEP 7 — Renaming Model & Final Summary")

    weights_dir = save_dir / "weights"
    best_pt     = weights_dir / "best.pt"
    results_csv = save_dir / "results.csv"

    if not best_pt.exists():
        print(f"[ERROR] best.pt not found at: {best_pt}")
        print("        Check the runs folder manually.")
        return

    if not results_csv.exists():
        print(f"[ERROR] results.csv not found at: {results_csv}")
        print("        Cannot rename model without metric values.")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    last_row = df.iloc[-1]

    map50 = None
    for col in ("metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP_50(B)"):
        if col in df.columns:
            map50 = float(last_row[col])
            break

    if map50 is None:
        print("[ERROR] Could not find a mAP50 column in results.csv.")
        print(f"        Available columns: {list(df.columns)}")
        return

    map50_95 = None
    for col in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP_50-95(B)"):
        if col in df.columns:
            map50_95 = float(last_row[col])
            break

    if map50_95 is None:
        print("[ERROR] Could not find a mAP50-95 column in results.csv.")
        print(f"        Available columns: {list(df.columns)}")
        return

    map50_pct    = round(map50    * 100, 2)
    map50_95_pct = round(map50_95 * 100, 2)

    new_name = f"best_mAP50-{map50_pct}%_mAP50-95-{map50_95_pct}%.pt"
    new_path = weights_dir / new_name
    os.rename(best_pt, new_path)

    bar = "=" * 52
    print(bar)
    print("  TRAINING COMPLETE")
    print(bar)
    print(f"  Model accuracy (mAP50)     : {map50_pct}%")
    print(f"  Model accuracy (mAP50-95)  : {map50_95_pct}%")
    print(f"  Best model saved as        : {new_name}")
    print(f"  Full path                  : {new_path}")
    print(bar)


# ── Cleanup ───────────────────────────────────────────────────────────────────
def _clear_folder_contents(folder: Path) -> int:
    if not folder.exists():
        return 0
    removed = 0
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        else:
            shutil.rmtree(item)
        removed += 1
    return removed


def clear_dataset_folders() -> None:
    banner("STEP 0 — Dataset Cleanup Check")

    nonempty = [f for f in DATASET_FOLDERS if f.exists() and any(f.iterdir())]

    if not nonempty:
        print("Dataset folders are empty. No cleanup needed.")
        return

    print("The following dataset folders already contain files:")
    for folder in nonempty:
        n = sum(1 for _ in folder.iterdir())
        print(f"  {folder.relative_to(ROOT)}  ({n} file(s))")
    print()

    answer = input("Old dataset detected. Do you want to clear it before loading new data? (y/n): ").strip().lower()

    if answer == "y":
        print("\nClearing dataset folders …")
        total = 0
        for folder in DATASET_FOLDERS:
            removed = _clear_folder_contents(folder)
            if removed:
                print(f"  Cleared {removed} file(s) from {folder.relative_to(ROOT)}")
            total += removed
        print(f"  Total files removed: {total}")

        if RUNS_DIR.exists() and any(RUNS_DIR.iterdir()):
            clear_runs = input("\nDo you also want to clear previous training runs? (y/n): ").strip().lower()
            if clear_runs == "y":
                removed = _clear_folder_contents(RUNS_DIR)
                print(f"  Cleared {removed} item(s) from runs/")
            else:
                print("  Keeping existing training runs.")

        print("\nOld dataset cleared. Loading new zip...")
    else:
        print()
        print("WARNING: New images will be mixed with old dataset images.")
        print("         This may affect training accuracy.")


# ── Date-stamped run directory ─────────────────────────────────────────────────
def make_run_info() -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dated_dir = RUNS_DIR / timestamp
    dated_dir.mkdir(parents=True, exist_ok=True)
    return dated_dir, RUN_NAME


# ── Reset ──────────────────────────────────────────────────────────────────────
def reset_training_runs() -> None:
    banner("RESET — Clearing Previous Training Runs")

    if not RUNS_DIR.exists() or not any(RUNS_DIR.iterdir()):
        print("No previous training runs found. Nothing to reset.")
        return

    runs = sorted(RUNS_DIR.iterdir())
    print(f"Found {len(runs)} training run folder(s):")
    for r in runs:
        print(f"  {r.name}")
    print()

    answer = input("Delete ALL training runs listed above? [y/N]: ").strip().lower()
    if answer == "y":
        shutil.rmtree(RUNS_DIR)
        RUNS_DIR.mkdir()
        print("[OK] All training runs deleted. Ready to start fresh.")
    else:
        print("Reset cancelled — existing runs kept.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--reset" in sys.argv:
        reset_training_runs()
        sys.exit(0)

    banner("YOLOv8 Oil Spill Detection — Setup & Train (V3)")
    print(f"Project root : {ROOT}")
    print(f"Dataset dir  : {DATASET}")
    print()

    dated_dir, run_name = make_run_info()
    print(f"Results will be saved to: {dated_dir.relative_to(ROOT)} / detect / {run_name}\n")

    already_extracted = input("Do you already have the dataset extracted? (y/n): ").strip().lower()

    if already_extracted == "y":
        print("Skipping extraction, using existing dataset...")
        yaml_path = fix_data_yaml()
        verify_dataset()
        train(yaml_path, dated_dir, run_name)
    else:
        clear_dataset_folders()
        zip_path  = get_zip_path()
        extract_and_map(zip_path)
        yaml_path = fix_data_yaml()
        verify_dataset()
        train(yaml_path, dated_dir, run_name)