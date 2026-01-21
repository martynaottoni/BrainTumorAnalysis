import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def image_files(folder, img_exts={'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}):
    for dirpath, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in img_exts:
                yield Path(dirpath) / f

def local_normalize(img):
    """
    Lokalna normalizacja dla każdego obrazu osobno
    - Z-score normalization z lokalnymi statystykami
    - Output: float32 w zakresie [0,1] (lepsze dla ML)
    """
    # 1. Z-score normalization z lokalnymi statystykami
    img_norm = (img - img.mean()) / (img.std() + 1e-8)
    
    # 2. Clipping do rozsądnego zakresu (2.5 sigma)
    img_norm = np.clip(img_norm, -2.5, 2.5)
    
    # 3. Skalowanie do [0,1] dla ML (float32)
    img_final = ((img_norm + 2.5) / 5).astype(np.float32)
    
    return img_final

def save_normalized(norm_img, img_path, src_root, dst_root):
    # Znajdź, od którego folderu (Training/Testing/Validation) zaczyna się ścieżka
    folders = ['Training', 'Testing', 'Validation']
    for folder in folders:
        try:
            idx = img_path.parts.index(folder)
            rel_path = Path(*img_path.parts[idx:])  # np. Training/glioma/plik.jpg
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Nie znaleziono folderu bazowego w ścieżce: {img_path}")
    
    out_dir = dst_root / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / img_path.name
    
    # Zapisujemy jako JPG (konwertujemy float32 [0,1] do uint8 [0,255])
    img_uint8 = (norm_img * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), img_uint8)

def main():
    ROOT = Path(__file__).parent.resolve()
    SRC_ROOT = ROOT / 'Preprocessing' / 'debug_step3'  # Źródło: pliki po preprocessing'u
    DST_ROOT = ROOT / 'Normalization'  # Cel: znormalizowane pliki
    
    print("Rozpoczynam lokalną normalizację...")
    print("Każdy obraz będzie znormalizowany używając własnych statystyk")
    print(f"Źródło: {SRC_ROOT}")
    print(f"Cel: {DST_ROOT}")
    
    folders = ['Training', 'Testing', 'Validation']
    for folder in folders:
        folder_path = SRC_ROOT / folder
        if not folder_path.exists():
            print(f"Folder {folder_path} nie istnieje, pomijam...")
            continue
            
        print(f"\nPrzetwarzam folder: {folder}")
        for img_path in tqdm(list(image_files(folder_path)), desc=f"Normalizuję {folder}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'Nie można wczytać {img_path}')
                continue
                
            norm_img = local_normalize(img)
            save_normalized(norm_img, img_path, SRC_ROOT, DST_ROOT)
    
    print("\nNormalizacja zakończona!")
    print(f"Znormalizowane obrazy zapisane w: {DST_ROOT}")
    print("Pliki z debug_step3 pozostają niezmienione.")

if __name__ == "__main__":
    main() 