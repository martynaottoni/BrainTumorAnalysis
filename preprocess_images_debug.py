import os
from pathlib import Path
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage import measure
import argparse
from skimage.segmentation import clear_border
import torch
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.resolve()
FOLDERS = ['Training', 'Testing', 'Validation']  # Dodano Validation
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

# Folder główny dla preprocessingu
PREPROCESSING_ROOT = ROOT / 'Preprocessing'

DEBUG_STEPS = {
    1: 'debug_step1',
    2: 'debug_step2',
    3: 'debug_step3',
}

STEP_NAMES = {
    1: 'Median Blur',
    2: 'Anizotropowa dyfuzja (TV)',
    3: 'CLAHE',
}

# Funkcja do pobierania plików z danego folderu bazowego

def image_files(folder):
    for dirpath, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                yield Path(dirpath) / f

def save_step(img, img_path, step):
    # Znajdź, od którego folderu (Training/Testing/Validation) zaczyna się ścieżka
    for folder in FOLDERS:
        try:
            idx = img_path.parts.index(folder)
            rel_path = Path(*img_path.parts[idx:])  # np. Training/glioma/plik.jpg
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Nie znaleziono folderu bazowego w ścieżce: {img_path}")
    debug_dir = PREPROCESSING_ROOT / DEBUG_STEPS[step] / rel_path.parent
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / img_path.name
    cv2.imwrite(str(out_path), img)

# Każdy krok jako osobna funkcja

def step1_median(img):
    return cv2.medianBlur(img, 3)

def step2_tv(img):
    img_tv = denoise_tv_chambolle(img, weight=0.001, channel_axis=-1)
    return (img_tv * 255).astype(np.uint8)

def step3_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[...,0] = clahe.apply(lab[...,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


STEP_FUNCS = {
    1: step1_median,
    2: step2_tv,
    3: step3_clahe,
}

# Funkcja główna do uruchamiania wybranego kroku

def run_step(step):
    print(f"Wykonuję krok {step}: {STEP_NAMES[step]}")
    # Ustal folder wejściowy
    if step == 1:
        input_folders = [ROOT / f for f in FOLDERS]
    else:
        input_folders = [PREPROCESSING_ROOT / DEBUG_STEPS[step-1] / f for f in FOLDERS]
    for folder in input_folders:
        for img_path in image_files(folder):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Nie można wczytać {img_path}")
                    continue
                out_img = STEP_FUNCS[step](img)
                save_step(out_img, img_path, step)
            except Exception as e:
                print(f"Błąd przy {img_path}: {e}")
    print(f'Krok {step} zakończony. Wyniki w folderze {PREPROCESSING_ROOT / DEBUG_STEPS[step]}')

def show_processing_grid(example_tuples, base_dir=PREPROCESSING_ROOT):
    """
    Wyświetla siatkę 3x4: 3 obrazy (wiersze), 4 wersje (kolumny): oryginał, median blur, TV, CLAHE
    example_tuples: lista krotek (class_name, file_name), np. [('glioma', 'Te-gl_0010.jpg'), ...]
    base_dir: katalog Preprocessing
    """
    titles = ["Oryginał", "Median Blur", "TV", "CLAHE"]
    n_rows = len(example_tuples)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    
    for row_idx, (class_name, example_name) in enumerate(example_tuples):
        step_dirs = [
            base_dir / "debug_step1" / "Testing" / class_name / example_name,
            base_dir / "debug_step2" / "Testing" / class_name / example_name,
            base_dir / "debug_step3" / "Testing" / class_name / example_name,
        ]
        imgs = []
        for p in step_dirs:
            img = cv2.imread(str(p))
            if img is None:
                print(f"Nie można wczytać {p}")
                imgs.append(np.zeros((64,64,3), dtype=np.uint8))
            else:
                imgs.append(img)
        # Oryginał to pierwszy obraz z debug_step1
        original = imgs[0]
        median_blur = imgs[0]
        tv = imgs[1]
        clahe = imgs[2]
        images = [original, median_blur, tv, clahe]
        for col_idx, (im, title) in enumerate(zip(images, titles)):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if row_idx == 0:
                ax.set_title(title, fontsize=16)
            ax.axis('off')
    
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uruchamianie wybranego kroku preprocessingu lub tylko wizualizacji.")
    parser.add_argument('--step', type=int, choices=[1,2,3], help='Numer kroku do wykonania (1-3)')
    parser.add_argument('--show-grid', action='store_true', help='Wyświetl tylko siatkę porównawczą, bez preprocessingu')
    args = parser.parse_args()

    if args.show_grid:
        example_tuples = [
            ("glioma", "Te-gl_0014.jpg"),
            ("meningioma", "Te-me_0029.jpg"),
            ("pituitary", "Te-pi_0015.jpg"),
        ]
        show_processing_grid(example_tuples)
    elif args.step:
        run_step(args.step)
    else:
        print("Podaj --step [1-3] lub --show-grid") 