import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from skimage.measure import shannon_entropy
from skimage import filters
import warnings
warnings.filterwarnings('ignore')

class PreprocessingQualityAnalyzer:
    def __init__(self):
        self.base_path = Path(__file__).parent.resolve()
                
                # Ścieżki do danych w kolejności preprocessingu
        self.preprocessing_stages = {
            "Oryginał": self.base_path / "Training",
            "Filtr medianowy": self.base_path / "Preprocessing" / "debug_step1" / "Training",
            "TV": self.base_path / "Preprocessing" / "debug_step2" / "Training", 
            "CLAHE": self.base_path / "Preprocessing" / "debug_step3" / "Training",
            "Normalizacja": self.base_path / "Normalization" / "Training",
            "Zmiana rozmiaru": self.base_path / "RGB_224x224" / "Training"
        }
        
        self.tumor_types = ["glioma", "meningioma", "pituitary", "notumor"]
        self.tumor_names = {
            "glioma": "Glejak",
            "meningioma": "Oponiak", 
            "pituitary": "Gruczolak przysadki",
            "notumor": "Brak guza"
        }
        
        # Sprawdź dostępność ścieżek
        print("Sprawdzanie dostępności ścieżek:")
        for stage, path in self.preprocessing_stages.items():
            exists = path.exists()
            print(f"  {stage}: {'✓' if exists else '✗'} {path}")
    
    def load_image(self, image_path):
        """Ładuje obraz w odpowiednim formacie"""
        if not image_path or not image_path.exists():
            return None
        
        try:
            if image_path.suffix.lower() == '.npy':
                # Załaduj plik .npy
                arr = np.load(str(image_path))
                if arr.ndim == 3:
                    # Konwertuj RGB na grayscale
                    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.shape[2] == 3 else arr[:,:,0]
                else:
                    gray = arr
                return gray.astype(np.float32)
            else:
                # Załaduj normalny obraz
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return None
                return img.astype(np.float32)
        except Exception as e:
            print(f"Błąd wczytywania {image_path}: {e}")
            return None
    
    def calculate_metrics(self, image):
        """Oblicza wszystkie metryki jakości dla obrazu"""
        if image is None:
            return None
        
        # Podstawowe statystyki
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Signal-to-Noise Ratio (SNR)
        snr = mean_intensity / (std_intensity + 1e-10)
        
        # Contrast-to-Noise Ratio (CNR) - różnica między najjaśniejszym a najciemniejszym regionem
        # Podziel obraz na regiony i oblicz CNR
        h, w = image.shape
        center_region = image[h//4:3*h//4, w//4:3*w//4]
        background_region = np.concatenate([
            image[:h//4, :].flatten(),
            image[3*h//4:, :].flatten(),
            image[:, :w//4].flatten(),
            image[:, 3*w//4:].flatten()
        ])
        
        center_mean = np.mean(center_region)
        bg_mean = np.mean(background_region)
        bg_std = np.std(background_region)
        
        cnr = abs(center_mean - bg_mean) / (bg_std + 1e-10)
        
        # Laplacian variance (ostrość)
        laplacian = filters.laplace(image)
        laplacian_var = np.var(laplacian)
        
        # Entropia (złożoność)
        entropy = shannon_entropy(image)
        
        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'snr': snr,
            'cnr': cnr,
            'laplacian_variance': laplacian_var,
            'entropy': entropy
        }
    
    def get_sample_images_from_stage(self, tumor_type, stage, num_samples=20):
        """Pobiera próbkę obrazów z danego etapu preprocessingu"""
        stage_path = self.preprocessing_stages[stage] / tumor_type
        
        if not stage_path.exists():
            print(f"Ścieżka nie istnieje: {stage_path}")
            return []
        
        # Znajdź wszystkie obrazy
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.npy']:
            image_files.extend(list(stage_path.glob(f"*{ext}")))
        
        if not image_files:
            print(f"Nie znaleziono obrazów w {stage_path}")
            return []
        
        # Ograń próbkę
        if len(image_files) > num_samples:
            image_files = random.sample(image_files, num_samples)
        
        # Załaduj obrazy
        images = []
        for img_path in image_files:
            img = self.load_image(img_path)
            if img is not None:
                images.append(img)
        
        return images
    
    def analyze_stage_metrics(self, tumor_type, stage, num_samples=20):
        """Analizuje metryki dla danego typu guza i etapu"""
        print(f"Analizuję {self.tumor_names[tumor_type]} - {stage}...")
        
        # Pobierz obrazy
        images = self.get_sample_images_from_stage(tumor_type, stage, num_samples)
        
        if not images:
            return None
        
        # Oblicz metryki dla wszystkich obrazów
        all_metrics = []
        for img in images:
            metrics = self.calculate_metrics(img)
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return None
        
        # Oblicz średnie
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return avg_metrics
    
    def analyze_all_stages(self, num_samples=20):
        """Analizuje wszystkie etapy dla wszystkich typów guza"""
        print("Rozpoczynam analizę wszystkich etapów preprocessingu...")
        print(f"Próbka: {num_samples} obrazów na etap")
        print("="*70)
        
        results = {}
        
        for tumor_type in self.tumor_types:
            print(f"\nAnalizuję typ guza: {self.tumor_names[tumor_type]}")
            print("-" * 50)
            
            tumor_results = {}
            
            for stage in self.preprocessing_stages.keys():
                stage_metrics = self.analyze_stage_metrics(tumor_type, stage, num_samples)
                tumor_results[stage] = stage_metrics
            
            results[tumor_type] = tumor_results
        
        return results
    
    def create_summary_dataframe(self, results):
        """Tworzy DataFrame z podsumowaniem wyników"""
        summary_data = []
        
        for tumor_type in self.tumor_types:
            for stage in self.preprocessing_stages.keys():
                stage_data = results[tumor_type].get(stage)
                if stage_data:
                    summary_data.append({
                        'Tumor_Type': self.tumor_names[tumor_type],
                        'Stage': stage,
                        'Mean_Intensity': stage_data['mean_intensity']['mean'],
                        'Mean_Intensity_Std': stage_data['mean_intensity']['std'],
                        'Std_Intensity': stage_data['std_intensity']['mean'],
                        'Std_Intensity_Std': stage_data['std_intensity']['std'],
                        'SNR': stage_data['snr']['mean'],
                        'SNR_Std': stage_data['snr']['std'],
                        'CNR': stage_data['cnr']['mean'],
                        'CNR_Std': stage_data['cnr']['std'],
                        'Laplacian_Variance': stage_data['laplacian_variance']['mean'],
                        'Laplacian_Variance_Std': stage_data['laplacian_variance']['std'],
                        'Entropy': stage_data['entropy']['mean'],
                        'Entropy_Std': stage_data['entropy']['std']
                    })
        
        return pd.DataFrame(summary_data)
    
    def plot_bar_charts(self, summary_df):
        """Rysuje wykresy słupkowe dla wszystkich metryk z błędami standardowymi"""
        stages = list(self.preprocessing_stages.keys())
        metrics = ['Mean_Intensity', 'Std_Intensity', 'SNR', 'CNR', 'Laplacian_Variance', 'Entropy']
        metric_labels = ['Średnia intensywność', 'Odchylenie standardowe', 'SNR', 'CNR', 'Wariancja Laplace\'a', 'Entropia']
        y_labels = ['Średnia intensywność [0–255]', 'Odchylenie standardowe [0–255]', 'SNR [–]', 'CNR [–], skala logarytmiczna', 'Wariancja Laplace\'a [–], \nskala logarytmiczna', 'Entropia [bity]']
        log_scale_metrics = ['CNR', 'Laplacian_Variance']  # Metryki z dużym rozrzutem
        
        # Kolory dla lepszej czytelności
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Niebieski, pomarańczowy, zielony, czerwony
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 20))
        axes = axes.flatten()
        
        for i, (metric, label, y_label) in enumerate(zip(metrics, metric_labels, y_labels)):
            ax = axes[i]
            
            # Przygotuj dane dla wykresu
            tumor_types = summary_df['Tumor_Type'].unique()
            x = np.arange(len(stages))
            width = 0.18
            
            for j, tumor in enumerate(tumor_types):
                tumor_data = summary_df[summary_df['Tumor_Type'] == tumor]
                values = []
                errors = []
                for stage in stages:
                    stage_data = tumor_data[tumor_data['Stage'] == stage]
                    if not stage_data.empty:
                        values.append(stage_data[metric].iloc[0])
                        errors.append(stage_data[f'{metric}_Std'].iloc[0])
                    else:
                        values.append(0)
                        errors.append(0)
                
                ax.bar(x + j*width, values, width, label=tumor, alpha=0.8, 
                      yerr=errors, capsize=4, error_kw={'alpha': 0.8, 'linewidth': 1.5},
                      color=colors[j], edgecolor='black', linewidth=0.8)
            
            ax.set_title(f'{label}', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Etap preprocessingu', fontsize=11, fontweight='bold')
            ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(stages, rotation=20, ha='right', fontsize=10, fontweight='bold')
            
            # Skala logarytmiczna dla metryk z dużym rozrzutem
            if metric in log_scale_metrics:
                ax.set_yscale('log')
                ax.set_ylabel(f'{y_label}', fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.4, linewidth=0.8)
            ax.tick_params(labelsize=12)
            
            # Usuń legendy z pojedynczych wykresów
            
        # Dodaj jedną wspólną legendę pod wykresami
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=14, 
                           bbox_to_anchor=(0.5, -0.02))
        for text in legend.get_texts():
            text.set_fontweight('bold')
        
        plt.tight_layout(pad=6.0)
        plt.subplots_adjust(bottom=0.15, hspace=0.5, wspace=0.3)  # Więcej miejsca na legendę i między wykresami
        plt.savefig('preprocessing_metrics_bar_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_trend_lines(self, summary_df):
        """Rysuje wykresy liniowe pokazujące trendy z błędami standardowymi"""
        stages = list(self.preprocessing_stages.keys())
        metrics = ['Mean_Intensity', 'Std_Intensity', 'SNR', 'CNR', 'Laplacian_Variance', 'Entropy']
        metric_labels = ['Średnia intensywność', 'Odchylenie standardowe', 'SNR', 'CNR', 'Wariancja Laplace\'a', 'Entropia']
        y_labels = ['Średnia intensywność [0–255]', 'Odchylenie standardowe [0–255]', 'SNR [–]', 'CNR [–], skala logarytmiczna', 'Wariancja Laplace\'a [–], \nskala logarytmiczna', 'Entropia [bity]']
        log_scale_metrics = ['CNR', 'Laplacian_Variance']  # Metryki z dużym rozrzutem
        
        # Kolory i markery dla lepszej czytelności
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']  # Różne markery dla każdej grupy
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 20))
        axes = axes.flatten()
        
        for i, (metric, label, y_label) in enumerate(zip(metrics, metric_labels, y_labels)):
            ax = axes[i]
            
            tumor_types = summary_df['Tumor_Type'].unique()
            
            for j, tumor in enumerate(tumor_types):
                tumor_data = summary_df[summary_df['Tumor_Type'] == tumor]
                values = []
                errors = []
                for stage in stages:
                    stage_data = tumor_data[tumor_data['Stage'] == stage]
                    if not stage_data.empty:
                        values.append(stage_data[metric].iloc[0])
                        errors.append(stage_data[f'{metric}_Std'].iloc[0])
                    else:
                        values.append(np.nan)
                        errors.append(np.nan)
                
                x_range = range(len(stages))
                ax.errorbar(x_range, values, yerr=errors, fmt=f'{markers[j]}-', 
                          label=tumor, linewidth=2, markersize=6, capsize=4, 
                          capthick=1.5, alpha=0.9, color=colors[j])
            
            ax.set_title(f'{label}', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Etap preprocessingu', fontsize=11, fontweight='bold')
            ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(stages)))
            ax.set_xticklabels(stages, rotation=20, ha='right', fontsize=10, fontweight='bold')
            
            # Skala logarytmiczna dla metryk z dużym rozrzutem
            if metric in log_scale_metrics:
                ax.set_yscale('log')
                ax.set_ylabel(f'{y_label}', fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.4, linewidth=0.8)
            ax.tick_params(labelsize=12)
        
        # Dodaj jedną wspólną legendę pod wykresami
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=14, 
                           bbox_to_anchor=(0.5, -0.02))
        for text in legend.get_texts():
            text.set_fontweight('bold')
        
        plt.tight_layout(pad=8.0)
        plt.subplots_adjust(bottom=0.19, hspace=0.7, wspace=0.4)  # Więcej miejsca na legendę i między wykresami
        plt.savefig('preprocessing_metrics_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_histograms_comparison(self, num_samples=50):
        """Rysuje histogramy porównawcze przed i po preprocessingu"""
        print("Tworzę histogramy porównawcze...")
        
        fig, axes = plt.subplots(len(self.tumor_types), 2, figsize=(18, 6*len(self.tumor_types)))
        
        for i, tumor_type in enumerate(self.tumor_types):
            # Przed preprocessingiem (Original)
            original_images = self.get_sample_images_from_stage(tumor_type, "Oryginał", num_samples)
            
            # Po preprocessingu (ostatni etap)
            final_images = self.get_sample_images_from_stage(tumor_type, "Zmiana rozmiaru", num_samples)
            
            if original_images and final_images:
                # Połącz wszystkie piksele
                original_pixels = np.concatenate([img.flatten() for img in original_images])
                final_pixels = np.concatenate([img.flatten() for img in final_images])
                
                # Histogram przed
                axes[i, 0].hist(original_pixels, bins=100, alpha=0.8, color='#4472C4', 
                              edgecolor='black', linewidth=0.5, density=True)
                axes[i, 0].set_title(f'{self.tumor_names[tumor_type]} - Przed preprocessingiem', 
                                   fontsize=14, fontweight='bold', pad=30)
                axes[i, 0].set_xlabel('Intensywność pikseli [0-255]', fontsize=14, fontweight='bold')
                axes[i, 0].set_ylabel('Gęstość prawdopodobieństwa [-]', fontsize=14, fontweight='bold')
                axes[i, 0].grid(True, alpha=0.4, linewidth=0.8)
                axes[i, 0].tick_params(labelsize=12)
                
                # Histogram po
                axes[i, 1].hist(final_pixels, bins=100, alpha=0.8, color='#E74C3C', 
                              edgecolor='black', linewidth=0.5, density=True)
                axes[i, 1].set_title(f'{self.tumor_names[tumor_type]} - Po preprocessingu', 
                                   fontsize=14, fontweight='bold', pad=30)
                axes[i, 1].set_xlabel('Intensywność pikseli (z-score) [-]', fontsize=14, fontweight='bold')
                axes[i, 1].set_ylabel('Gęstość prawdopodobieństwa [-]', fontsize=14, fontweight='bold')
                axes[i, 1].grid(True, alpha=0.4, linewidth=0.8)
                axes[i, 1].tick_params(labelsize=12)
                
                # Dodaj statystyki do histogramów
                axes[i, 0].axvline(np.mean(original_pixels), color='red', linestyle='--', 
                                 linewidth=2, label=f'Średnia:\n{np.mean(original_pixels):.1f}')
                axes[i, 0].legend(fontsize=12)
                
                axes[i, 1].axvline(np.mean(final_pixels), color='blue', linestyle='--', 
                                 linewidth=2, label=f'Średnia:\n{np.mean(final_pixels):.2f}')
                axes[i, 1].legend(fontsize=12)
        
        plt.tight_layout(pad=6.0)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.savefig('preprocessing_histograms_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results_to_csv(self, summary_df):
        """Zapisuje wyniki do pliku CSV"""
        summary_df.to_csv('preprocessing_quality_metrics.csv', index=False)
        print("Wyniki zapisane do: preprocessing_quality_metrics.csv")
    
    def print_summary_table(self, summary_df):
        """Wyświetla tabelę podsumowującą w formacie średnia ± SD"""
        print("\n" + "="*120)
        print("PODSUMOWANIE METRYK JAKOŚCI PREPROCESSINGU (Format: Średnia ± SD)")
        print("="*120)
        
        for tumor in self.tumor_names.values():
            print(f"\n{tumor.upper()}")
            print("-" * 100)
            tumor_data = summary_df[summary_df['Tumor_Type'] == tumor]
            if not tumor_data.empty:
                # Formatuj wyniki jako średnia ± SD
                formatted_data = []
                for _, row in tumor_data.iterrows():
                    formatted_row = {
                        'Etap': row['Stage'],
                        'Śred. intensywność': f"{row['Mean_Intensity']:.2f} ± {row['Mean_Intensity_Std']:.2f}",
                        'SNR': f"{row['SNR']:.2f} ± {row['SNR_Std']:.2f}",
                        'CNR': f"{row['CNR']:.2f} ± {row['CNR_Std']:.2f}",
                        'War. Laplace': f"{row['Laplacian_Variance']:.1f} ± {row['Laplacian_Variance_Std']:.1f}",
                        'Entropia': f"{row['Entropy']:.3f} ± {row['Entropy_Std']:.3f}"
                    }
                    formatted_data.append(formatted_row)
                
                # Konwertuj na DataFrame i wyświetl
                formatted_df = pd.DataFrame(formatted_data)
                print(formatted_df.to_string(index=False, max_colwidth=25))
    
    def run_analysis(self, num_samples=20):
        """Uruchamia pełną analizę"""
        print("ANALIZA JAKOŚCI PREPROCESSINGU")
        print("="*70)
        
        # Analizuj wszystkie etapy
        results = self.analyze_all_stages(num_samples)
        
        # Utwórz DataFrame z podsumowaniem
        summary_df = self.create_summary_dataframe(results)
        
        # Wyświetl tabelę podsumowującą
        self.print_summary_table(summary_df)
        
        # Zapisz do CSV
        self.save_results_to_csv(summary_df)
        
        # Rysuj wykresy
        print("\nTworzę wykresy...")
        self.plot_bar_charts(summary_df)
        self.plot_trend_lines(summary_df)
        self.plot_histograms_comparison(num_samples)
        
        print("\nAnaliza zakończona!")
        print("Utworzone pliki:")
        print("   • preprocessing_quality_metrics.csv")
        print("   • preprocessing_metrics_bar_charts.png")
        print("   • preprocessing_metrics_trends.png")
        print("   • preprocessing_histograms_comparison.png")
        
        return summary_df

def main():
    # Ustawienia
    random.seed(42)  # Dla powtarzalności
    
    analyzer = PreprocessingQualityAnalyzer()
    
    # Uruchom analizę
    summary_df = analyzer.run_analysis(num_samples=25)
    
    print(f"\nGotowe! Sprawdź utworzone pliki z wynikami.")

if __name__ == "__main__":
    main()
