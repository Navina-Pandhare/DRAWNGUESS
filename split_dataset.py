import os, shutil, random
from pathlib import Path

# CONFIG
SRC_ROOT = Path("dataset_original")      # e.g., dataset_raw/cat/*.png, dataset_raw/dog/*.jpg
DST_ROOT = Path("dataset")          # will create dataset/train|val|test/<class>/
TRAIN_PCT, VAL_PCT, TEST_PCT = 0.80, 0.10, 0.10
SEED = 42
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

random.seed(SEED)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def split_and_copy():
    classes = [d.name for d in SRC_ROOT.iterdir() if d.is_dir()]
    print("Classes:", classes)

    for split in ["train", "val", "test"]:
        for cls in classes:
            ensure_dir(DST_ROOT / split / cls)

    for cls in classes:
        src_dir = SRC_ROOT / cls
        files = [p for p in src_dir.iterdir() if p.suffix.lower() in ALLOWED_EXT and p.is_file()]
        files.sort()                       # deterministic order before shuffling
        random.shuffle(files)              # shuffle for randomness

        n = len(files)
        n_train = int(n * TRAIN_PCT)
        n_val   = int(n * VAL_PCT)
        n_test  = n - n_train - n_val      # remainder goes to test to avoid rounding issues

        train_files = files[:n_train]
        val_files   = files[n_train:n_train+n_val]
        test_files  = files[n_train+n_val:]

        print(f"{cls}: total={n}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

        for f in train_files:
            shutil.copy2(f, DST_ROOT / "train" / cls / f.name)
        for f in val_files:
            shutil.copy2(f, DST_ROOT / "val"   / cls / f.name)
        for f in test_files:
            shutil.copy2(f, DST_ROOT / "test"  / cls / f.name)

if __name__ == "__main__":
    split_and_copy()
    print("✅ Split complete. Check the 'dataset' folder.")
