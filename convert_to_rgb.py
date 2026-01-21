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

def convert_to_rgb(img_gray):
    """
    Konwertuje grayscale do RGB (3 kanały)
    """
    # Metoda 1: Replikacja kanałów (najprostsza)
    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
    return img_rgb

def resize_image(img, target_size=(224, 224)):
    """
    Zmienia rozmiar obrazu do target_size
    """
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def save_rgb_image(img_rgb, img_path, src_root, dst_root):
    """
    Zapisuje RGB obraz jako JPG
    """
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
    
    # Zapis jako JPG (float32 [0,1] konwertujemy do uint8 [0,255] dla zapisu)
    img_uint8 = (img_rgb * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), img_uint8)

def main():
    SRC_ROOT = Path(__file__).parent.resolve() / 'Normalization'
    DST_ROOT = Path(__file__).parent.resolve() / 'RGB_224x224'
    
    print("Konwertuję grayscale do RGB i zmieniam rozmiar do 224x224...")
    print(f"Źródło: {SRC_ROOT}")
    print(f"Cel: {DST_ROOT}")
    
    folders = ['Training', 'Testing', 'Validation']  # Dodano Validation
    for folder in folders:
        folder_path = SRC_ROOT / folder
        if not folder_path.exists():
            print(f"Folder {folder_path} nie istnieje, pomijam...")
            continue
            
        print(f"\nPrzetwarzam folder: {folder}")
        for img_path in tqdm(list(image_files(folder_path)), desc=f"Konwertuję {folder}"):
            # Wczytaj .jpg i konwertuj do [0,1]
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f'Nie można wczytać {img_path}')
                continue
            
            # Konwertuj do float32 [0,1]
            img_gray = img_gray.astype(np.float32) / 255.0
            
            # Zmień rozmiar do 224x224
            img_resized = resize_image(img_gray, (224, 224))
            
            # Konwertuj do RGB (3 kanały)
            img_rgb = convert_to_rgb(img_resized)
            
            # Sprawdź wymiary
            if img_rgb.shape != (224, 224, 3):
                print(f"Błąd wymiarów dla {img_path}: {img_rgb.shape}")
                continue
            
            # Zapisz
            save_rgb_image(img_rgb, img_path, SRC_ROOT, DST_ROOT)
    
    print("\nKonwersja zakończona!")
    print(f"RGB obrazy 224x224 zapisane w: {DST_ROOT}")
    print("Przetwarzanie: float32 [0,1] w pamięci")
    print("Zapis: JPG (uint8 [0,255], shape: 224x224x3)")

if __name__ == "__main__":
    main()