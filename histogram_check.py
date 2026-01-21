import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ścieżki do obrazów przed i po normalizacji
BEFORE_ROOT = Path(__file__).parent.resolve() / 'Preprocessing' / 'debug_step3'
AFTER_ROOT = Path(__file__).parent.resolve() / 'Normalization'

FOLDERS = ['Training', 'Testing', 'Validation']  # Dodano Validation
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

def image_files(folder):
    for dirpath, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                yield Path(dirpath) / f

def plot_comparison_histograms(example_names, class_name="glioma"):
    """
    Wyświetla histogramy porównujące obrazy przed i po normalizacji
    dla kilku przykładowych obrazów w siatce
    """
    n_examples = len(example_names)
    fig, axes = plt.subplots(n_examples, 2, figsize=(16, 7 * n_examples))  # Zmieniono na 2 kolumny
    
    for idx, example_name in enumerate(example_names):
        # Ścieżki do obrazów
        before_path = BEFORE_ROOT / "Testing" / class_name / example_name
        after_path = AFTER_ROOT / "Testing" / class_name / example_name
        
        # Wczytaj obrazy
        before_img = cv2.imread(str(before_path), cv2.IMREAD_GRAYSCALE)
        after_img = cv2.imread(str(after_path), cv2.IMREAD_GRAYSCALE)
        
        if before_img is None or after_img is None:
            print(f"Nie można wczytać obrazów dla {example_name}")
            continue
            
        # Przygotuj osie
        if n_examples == 1:
            ax_before = axes[0]
            ax_after = axes[1]
        else:
            ax_before = axes[idx, 0]
            ax_after = axes[idx, 1]
        
        # Histogram przed normalizacją
        ax_before.hist(before_img.ravel(), bins=50, range=(0, 255), 
                      color='red', alpha=0.7, label='Przed normalizacją')
        ax_before.set_title(f'Przed - {example_name}', fontsize=12)
        ax_before.set_xlabel('Wartość piksela')
        ax_before.set_ylabel('Liczba pikseli')
        ax_before.legend()
        ax_before.grid(True, alpha=0.3)
        
        # Histogram po normalizacji
        ax_after.hist(after_img.ravel(), bins=50, range=(0, 255), 
                     color='blue', alpha=0.7, label='Po normalizacji')
        ax_after.set_title(f'Po - {example_name}', fontsize=12)
        ax_after.set_xlabel('Wartość piksela')
        ax_after.set_ylabel('Liczba pikseli')
        ax_after.legend()
        ax_after.grid(True, alpha=0.3)
    
    plt.suptitle(f'Porównanie histogramów - klasa: {class_name}', fontsize=16)
    plt.subplots_adjust(wspace=0.4, hspace=0.6, top=0.9, bottom=0.1)
    plt.show()

def plot_statistics_comparison():
    """
    Wyświetla statystyki porównujące rozkład wartości pikseli przed i po normalizacji
    dla wszystkich klas
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, class_name in enumerate(CLASSES):
        before_stats = []
        after_stats = []
        
        # Zbierz statystyki dla wszystkich obrazów w klasie
        before_dir = BEFORE_ROOT / "Testing" / class_name
        after_dir = AFTER_ROOT / "Testing" / class_name
        
        if not before_dir.exists() or not after_dir.exists():
            print(f"Foldery nie istnieją dla klasy {class_name}")
            continue
        
        for img_path in image_files(before_dir):
            before_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if before_img is None:
                continue
                
            # Znajdź odpowiadający obraz po normalizacji
            rel_path = img_path.relative_to(before_dir)
            after_path = after_dir / rel_path
            
            if after_path.exists():
                after_img = cv2.imread(str(after_path), cv2.IMREAD_GRAYSCALE)
                if after_img is not None:
                    before_stats.append(before_img.mean())
                    after_stats.append(after_img.mean())
        
        if before_stats and after_stats:
            # Wykres porównawczy
            axes[idx].scatter(before_stats, after_stats, alpha=0.6, s=20)
            axes[idx].plot([0, 255], [0, 255], 'r--', alpha=0.8, label='y=x')
            axes[idx].set_xlabel('Średnia wartość piksela - przed')
            axes[idx].set_ylabel('Średnia wartość piksela - po')
            axes[idx].set_title(f'Porównanie średnich - {class_name}', fontsize=12)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Porównanie statystyk przed i po normalizacji', fontsize=16)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

def check_pixel_ranges(example_names, class_name="glioma"):
    """
    Sprawdza rzeczywiste zakresy wartości pikseli przed i po normalizacji
    """
    print(f"\n=== ANALIZA ZAKRESÓW WARTOŚCI PIKSELI - {class_name.upper()} ===")
    
    for example_name in example_names:
        before_path = BEFORE_ROOT / "Testing" / class_name / example_name
        after_path = AFTER_ROOT / "Testing" / class_name / example_name
        
        before_img = cv2.imread(str(before_path), cv2.IMREAD_GRAYSCALE)
        after_img = cv2.imread(str(after_path), cv2.IMREAD_GRAYSCALE)
        
        if before_img is None or after_img is None:
            print(f"Nie można wczytać obrazów dla {example_name}")
            continue
            
        print(f"\n{example_name}:")
        print(f"  Przed: min={before_img.min()}, max={before_img.max()}, mean={before_img.mean():.2f}")
        print(f"  Po:    min={after_img.min()}, max={after_img.max()}, mean={after_img.mean():.2f}")
        
        # Sprawdź czy normalizacja rzeczywiście zmieniła zakres
        if before_img.min() == after_img.min() and before_img.max() == after_img.max():
            print(f"  ⚠️  UWAGA: Zakres wartości nie zmienił się!")
        else:
            print(f"  ✅ Zakres został zmieniony")

def main():
    # Sprawdź czy foldery istnieją
    if not BEFORE_ROOT.exists():
        print(f"Błąd: Folder {BEFORE_ROOT} nie istnieje!")
        print("Uruchom najpierw preprocessing i normalizację.")
        return
        
    if not AFTER_ROOT.exists():
        print(f"Błąd: Folder {AFTER_ROOT} nie istnieje!")
        print("Uruchom najpierw normalizację.")
        return
    
    # Przykładowe obrazy do porównania - używam rzeczywistych nazw plików
    example_names = ["Te-gl_0200.jpg", "Te-gl_0201.jpg", "Te-gl_0202.jpg"]
    
    print("Sprawdzanie zakresów wartości pikseli...")
    check_pixel_ranges(example_names, "glioma")
    
    print("\nGenerowanie histogramów porównawczych...")
    plot_comparison_histograms(example_names, "glioma")
    
    print("\nGenerowanie statystyk porównawczych...")
    plot_statistics_comparison()

if __name__ == "__main__":
    main() 