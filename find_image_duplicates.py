import os
import shutil
from pathlib import Path
from PIL import Image
import imagehash

# Define paths
ROOT = Path(__file__).parent.resolve()
TESTING_DIR = ROOT / 'Testing'
TRAINING_DIR = ROOT / 'Training'
DUPLICATES_DIR = ROOT / 'duplicates'

# Supported image extensions
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

# Create duplicates directory if it doesn't exist
DUPLICATES_DIR.mkdir(exist_ok=True)

def all_image_files(*folders):
    for folder in folders:
        for dirpath, _, filenames in os.walk(folder):
            for fname in filenames:
                if Path(fname).suffix.lower() in IMG_EXTS:
                    yield Path(dirpath) / fname

def main():
    hash_dict = {}
    files = list(all_image_files(TESTING_DIR, TRAINING_DIR))
    print(f"Scanning {len(files)} images...")
    
    # Calculate hashes for all images
    for img_path in files:
        try:
            with Image.open(img_path) as img:
                img_hash = imagehash.phash(img)
        except Exception as e:
            print(f"[ERROR] Could not process {img_path}: {e}")
            continue
        hash_str = str(img_hash)
        if hash_str in hash_dict:
            hash_dict[hash_str].append(img_path)
        else:
            hash_dict[hash_str] = [img_path]

    # Move duplicates
    moved = 0
    for hash_val, paths in hash_dict.items():
        if len(paths) > 1:
            # Keep the first, move the rest
            for dup_path in paths[1:]:
                rel_path = dup_path.relative_to(ROOT)
                dest_path = DUPLICATES_DIR / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[DUPLICATE] Moving {dup_path} -> {dest_path}")
                shutil.move(str(dup_path), str(dest_path))
                moved += 1
    print(f"Done. {moved} duplicates moved to '{DUPLICATES_DIR}'.")

if __name__ == "__main__":
    main() 