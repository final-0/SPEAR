"""Create the Spring val split by MOVING the first frames of every train sequence into val/."""
import os
import shutil
import sys

SPRING_ROOT = os.environ["SPRING_ROOT"]
TRAIN_DIR = os.path.join(SPRING_ROOT, "train")
VAL_DIR = os.path.join(SPRING_ROOT, "val")
N_VAL_FRAMES = 3

# A second run would move further train frames and rmtree the moved ones; run exactly once.
if os.path.exists(VAL_DIR):
    sys.exit(f"{VAL_DIR} already exists; refusing to re-split (run this script only once on a fresh Spring download)")

sequences = sorted(os.listdir(TRAIN_DIR))
print(f"Train sequences: {len(sequences)}")

for seq in sequences:
    seq_train = os.path.join(TRAIN_DIR, seq)
    seq_val = os.path.join(VAL_DIR, seq)
    os.makedirs(seq_val, exist_ok=True)

    for subdir in sorted(os.listdir(seq_train)):
        src_dir = os.path.join(seq_train, subdir)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(seq_val, subdir)
        os.makedirs(dst_dir, exist_ok=True)

        files = sorted(os.listdir(src_dir))
        if "frame_" in subdir:
            n_move = N_VAL_FRAMES + 1  # need 4 frames for 3 pairs
        else:
            n_move = N_VAL_FRAMES

        for f in files[:n_move]:
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f)
            shutil.copy2(src, dst)  # copy to val
            os.remove(src)          # remove from train
        
        remaining = len(os.listdir(src_dir))
        print(f"  {seq}/{subdir}: moved {n_move}, remaining {remaining}")

print(f"\nDone. Val at {VAL_DIR}")
