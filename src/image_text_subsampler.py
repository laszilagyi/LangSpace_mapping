import os
import sys
import shutil
from pathlib import Path

def subsample_images_and_traj(root_dir, sub):
    root_dir = Path(root_dir)
    input_img_dir = root_dir / "results_"
    output_img_dir = root_dir / "results"
    output_img_dir.mkdir(parents=True, exist_ok=True)

    # Get all frame and depth images, sorted
    frame_images = sorted(input_img_dir.glob("frame00*.jpg"))
    depth_images = sorted(input_img_dir.glob("depth00*.png"))

    # Sanity check
    if len(frame_images) != len(depth_images):
        print(f"Warning: Number of frames ({len(frame_images)}) and depths ({len(depth_images)}) do not match.")

    num_copied = 0
    for i in range(0, min(len(frame_images), len(depth_images)), sub):
        shutil.copy(frame_images[i], output_img_dir / frame_images[i].name)
        shutil.copy(depth_images[i], output_img_dir / depth_images[i].name)
        num_copied += 1

    print(f"Copied every {sub}-th image. Total copied: {num_copied}")

    # Handle trajectory file
    traj_in_path = root_dir / "traj_.txt"
    traj_out_path = root_dir / "traj.txt"

    if traj_in_path.exists():
        with traj_in_path.open("r") as f_in:
            lines = f_in.readlines()

        with traj_out_path.open("w") as f_out:
            for i in range(0, len(lines), sub):
                f_out.write(lines[i])

        print(f"Subsampled trajectory saved to: {traj_out_path}")
    else:
        print(f"Warning: {traj_in_path} does not exist, skipping trajectory subsampling.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python subsample_images.py <root_folder> <sub>")
        sys.exit(1)

    root_folder = sys.argv[1]
    #output_folder = sys.argv[2]  # Not used in this new logic but kept for compatibility
    sub = int(sys.argv[2])

    subsample_images_and_traj(root_folder, sub)
