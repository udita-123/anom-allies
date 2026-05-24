"""
tools/prepare_ucsd.py — Convert UCSD Ped1 or Ped2 .tif sequences to .jpg frames.

Usage:
    python tools/prepare_ucsd.py --dataset ucsd_ped2
    python tools/prepare_ucsd.py --dataset ucsd_ped1
"""

import cv2
import argparse
from pathlib import Path


def convert_split(raw_dir: Path, out_dir: Path, split: str) -> None:
    split_dir = raw_dir / split
    out_split = out_dir / split.lower()

    if not split_dir.exists():
        print(f"[SKIP] {split_dir} not found")
        return

    sequences = sorted([d for d in split_dir.iterdir()
                        if d.is_dir() and not d.name.endswith("_gt")])

    print(f"\n── {split} split: {len(sequences)} sequences ──")

    for seq in sequences:
        out_seq = out_split / seq.name
        out_seq.mkdir(parents=True, exist_ok=True)

        images = sorted(seq.glob("*.tif"))
        if not images:
            print(f"  [SKIP] {seq.name} — no .tif files")
            continue

        print(f"  {seq.name}: {len(images)} frames")

        for i, img_path in enumerate(images, start=1):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"    [WARN] Failed to read {img_path.name}")
                continue
            img = cv2.resize(img, (256, 256))
            cv2.imwrite(str(out_seq / f"{i:04d}.jpg"), img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ucsd_ped2",
                        choices=["ucsd_ped1", "ucsd_ped2"],
                        help="Which UCSD dataset to prepare")
    args = parser.parse_args()

    raw_dir = Path("data/raw") / args.dataset
    out_dir = Path("data/frames") / args.dataset

    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} not found.")
        return

    print(f"Dataset : {args.dataset}")
    print(f"Source  : {raw_dir}")
    print(f"Output  : {out_dir}")

    convert_split(raw_dir, out_dir, "Train")
    convert_split(raw_dir, out_dir, "Test")

    train_frames = list((out_dir / "train").rglob("*.jpg"))
    test_frames  = list((out_dir / "test").rglob("*.jpg"))
    print(f"\nDONE — {len(train_frames)} train frames, {len(test_frames)} test frames")


if __name__ == "__main__":
    main()
