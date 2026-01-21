import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
from pathlib import Path
import warnings
from tabulate import tabulate
import logging
import time


# Wyłącz wszystkie warningi
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# REPRODUCIBILITY - Ustawienie seed dla rzetelności naukowej
RANDOM_SEED = 90
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)  # TF 1.x syntax

# Ustawienia
print(f" CNN FROM SCRATCH + AUGMENTED DATASET - Brain Tumor Classification")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU dostępny: {tf.test.is_gpu_available()}")  # TF 1.x syntax
print(f"Random seed: {RANDOM_SEED} (dla reproducibility)")

# Ścieżki do danych
DATA_ROOT = Path(r"C:\1.Mart\RGB_224x224")
TRAIN_DIR = DATA_ROOT / "Training"
VAL_DIR = DATA_ROOT / "Validation"
TEST_DIR = DATA_ROOT / "Testing"

# Parametry treningu 
BATCH_SIZE = 16
LEARNING_RATE = 0.00005  #  
EPOCHS = 150
NUM_CLASSES = 4  # glioma, meningioma, notumor, pituitary
IMG_SIZE = 224

# OPTYMALNA AUGMENTACJA + POWIĘKSZENIE ZBIORU DANYCH
print("\n STRATEGIA: OPTYMALNE PARAMETRY + ZWIĘKSZONY DATASET")
print("="*70)
print("• Augmentacja: OPTYMALNE parametry z testów CNN baseline")
print("• Dataset: 1.5x więcej kroków na epokę (więcej wariantów)")
print("="*70)

# Augmentacja treningowa (OPTYMALNE PARAMETRY z testów CNN baseline)
train_datagen = ImageDataGenerator(
    rotation_range=25,           
    width_shift_range=0.15,      
    height_shift_range=0.15,     
    horizontal_flip=True,        
    vertical_flip=False,                  
    zoom_range=0.15,            
    rescale=1./255  

# Tylko normalizacja dla walidacji/testowania (BEZ AUGMENTACJI)
val_datagen = ImageDataGenerator(rescale=1./255)  # Konwersja uint8 [0,255] → float32 [0,1]

# Generatory danych z MULTIPLE SAMPLES per image
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),  
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=RANDOM_SEED  # Reproducibility
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),  
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=RANDOM_SEED  # Reproducibility
)

test_generator = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),  
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=RANDOM_SEED  # Reproducibility
)

print(f"Train samples: {train_generator.samples}")
print(f"Val samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Classes: {list(train_generator.class_indices.keys())}")
print(f"🔄 EFEKTYWNE próbki treningowe: {train_generator.samples} × 3-5 = {train_generator.samples * 3}-{train_generator.samples * 5}")

# Analiza balansu klas (IDENTYCZNA JAK W ORYGINALNYM CNN)
def analyze_class_balance():
    """Analizuje balans klas w zbiorach danych"""
    print("\n" + "="*80)
    print("ANALIZA BALANSU KLAS")
    print("="*80)
    
    # Liczba próbek w każdej klasie
    class_counts = train_generator.classes
    unique, counts = np.unique(class_counts, return_counts=True)
    
    # Nazwy klas
    class_names = list(train_generator.class_indices.keys())
    
    # Tabela z liczbą próbek
    balance_table = []
    total_samples = len(class_counts)
    
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        percentage = (count / total_samples) * 100
        balance_table.append([
            class_names[class_idx],
            count,
            f"{percentage:.1f}%",
            f"{count * 3}-{count * 5}"  # Efektywne próbki z augmentacją
        ])
    
    print(tabulate(balance_table, headers=["Klasa", "Oryginalne", "Procent", "Z augmentacją"], tablefmt="grid"))
    
    # Sprawdzenie czy klasy są zbalansowane
    min_count = min(counts)
    max_count = max(counts)
    imbalance_ratio = max_count / min_count
    
    print(f"\nWspółczynnik niezbalansowania: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2:
        print("  KLASY SĄ NIEZBALANSOWANE - używam class weights!")
        return True
    else:
        print(" Klasy są względnie zbalansowane")
        return False

# Oblicz class weights dla niezbalansowanych klas (IDENTYCZNA JAK W ORYGINALNYM CNN)
def calculate_class_weights():
    """Oblicza wagi klas dla niezbalansowanych danych"""
    class_counts = train_generator.classes
    unique, counts = np.unique(class_counts, return_counts=True)
    
    # Oblicz wagi odwrotnie proporcjonalne do liczby próbek
    total_samples = len(class_counts)
    class_weights = {}
    
    for class_idx, count in zip(unique, counts):
        weight = total_samples / (len(unique) * count)
        class_weights[class_idx] = weight
    
    print("\nWagi klas:")
    class_names = list(train_generator.class_indices.keys())
    for class_idx, weight in class_weights.items():
        print(f"  {class_names[class_idx]}: {weight:.3f}")
    
    return class_weights

# CNN model w TensorFlow/Keras (IDENTYCZNY JAK W ORYGINALNYM)
def create_brain_tumor_cnn(num_classes=4):
    model = models.Sequential([
        # Blok 1: 3 -> 32 kanałów
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3),
                      kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),  # L2 regularyzacja
        layers.LeakyReLU(alpha=0.1),
        layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Blok 2: 32 -> 64 kanałów
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.ReLU(),
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Blok 3: 64 -> 128 kanałów
        layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.ReLU(),
        layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Blok 4: 128 -> 256 kanałów
        layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.ReLU(),
        layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Blok 5: 256 -> 512 kanałów
        layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.ReLU(),
        layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Warstwy w pełni połączone
        layers.Flatten(),
        layers.Dense(1024, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', kernel_initializer='glorot_normal')
    ])
    
    return model


# IDENTYCZNE funkcje metryk jak w oryginalnym CNN
def calculate_medical_metrics(y_true_classes, y_pred_classes, num_classes):
    """Oblicza dodatkowe metryki medyczne per klasa"""
    # Inicjalizuj tablice dla metryk
    sensitivity = np.zeros(num_classes)  # = Recall
    specificity = np.zeros(num_classes)
    ppv = np.zeros(num_classes)  # Positive Predictive Value = Precision
    npv = np.zeros(num_classes)  # Negative Predictive Value
    
    for i in range(num_classes):
        # Binarna klasyfikacja: klasa i vs reszta
        y_true_binary = (y_true_classes == i).astype(int)
        y_pred_binary = (y_pred_classes == i).astype(int)
        
        # True Positive, False Positive, True Negative, False Negative
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # Oblicz metryki (z zabezpieczeniem przed dzieleniem przez 0)
        sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0  # = Recall
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv[i] = tp / (tp + fp) if (tp + fp) > 0 else 0  # = Precision
        npv[i] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'sensitivity': sensitivity,  # = Recall
        'specificity': specificity,
        'ppv': ppv,  # = Precision
        'npv': npv
    }

def calculate_metrics(y_true, y_pred, y_score):
    """Oblicza wszystkie metryki dla modelu"""
    # Konwertuj one-hot encoding na klasy
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    
    # Metryki średnie (ważone)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average='weighted')
    
    # Metryki dla każdej klasy osobno
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(y_true_classes, y_pred_classes, average=None)
    
    # ROC AUC średnie i dla każdej klasy
    roc_auc_avg = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr')
    roc_auc_per_class = roc_auc_score(y_true, y_score, average=None, multi_class='ovr')
    
    # Dodatkowe metryki medyczne
    medical_metrics = calculate_medical_metrics(y_true_classes, y_pred_classes, y_true.shape[1])
    
    # Macierz pomyłek
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    return {
        'accuracy': accuracy,
        'precision': precision_avg,
        'recall': recall_avg,
        'f1': f1_avg,
        'roc_auc': roc_auc_avg,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'roc_auc_per_class': roc_auc_per_class,
        'support_per_class': support_per_class,
        'sensitivity_per_class': medical_metrics['sensitivity'],
        'specificity_per_class': medical_metrics['specificity'],
        'ppv_per_class': medical_metrics['ppv'],
        'npv_per_class': medical_metrics['npv'],
        'confusion_matrix': cm
    }

def plot_training_history(history):
    """Rysuje historię treningu"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Loss podczas treningu (CNN + Augmented Data)')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy (TF 1.15 compatibility)
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    ax2.plot(history.history[acc_key], label='Train Accuracy', color='blue')
    ax2.plot(history.history[val_acc_key], label='Val Accuracy', color='red')
    ax2.set_title('Accuracy podczas treningu (CNN + Augmented Data)')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """Rysuje macierz pomyłek"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Liczba próbek'})
    plt.title('Macierz pomyłek (CNN + Augmented Data)', fontsize=14, fontweight='bold')
    plt.xlabel('Predykcja', fontsize=12)
    plt.ylabel('Rzeczywistość', fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    # Wyłącz warningi TensorFlow
    tf.autograph.set_verbosity(0)
    
    print("Ładowanie danych...")
    
    # DIAGNOSTYKA: Sprawdź balans klas w train vs val
    print("\n" + "="*80)
    print("DIAGNOSTYKA: BALANS KLAS TRAIN vs VAL (Z AUGMENTACJĄ)")
    print("="*80)
    
    # Sprawdź train
    train_classes = train_generator.classes
    train_unique, train_counts = np.unique(train_classes, return_counts=True)
    
    # Sprawdź val
    val_classes = val_generator.classes
    val_unique, val_counts = np.unique(val_classes, return_counts=True)
    
    class_names = list(train_generator.class_indices.keys())
    
    print("Train set (z augmentacją x3-5):")
    for i, (class_idx, count) in enumerate(zip(train_unique, train_counts)):
        print(f"  {class_names[class_idx]}: {count} oryginalne → {count * 3}-{count * 5} efektywne")
    
    print("\nVal set (bez augmentacji):")
    for i, (class_idx, count) in enumerate(zip(val_unique, val_counts)):
        print(f"  {class_names[class_idx]}: {count}")
    
    print("\n" + "="*80)
    
    # Analiza balansu klas
    is_imbalanced = analyze_class_balance()
    
    # Oblicz wagi klas jeśli są niezbalansowane
    class_weights = None
    if is_imbalanced:
        class_weights = calculate_class_weights()
    
    # Tworzenie modelu
    model = create_brain_tumor_cnn(num_classes=NUM_CLASSES)
    
    # Wyświetlenie parametrów modelu (wbudowana funkcja TensorFlow)
    print("\n" + "="*80)
    print("PARAMETRY SIECI CNN + AUGMENTED DATA")
    print("="*80)
    model.summary()
    
    # Kompilacja modelu (TF 1.15 compatible)
    model.compile(
        optimizer=optimizers.Adam(lr=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks (zoptymalizowane dla stabilności)
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,        # Zwiększone z 20 na 25 (więcej danych = może dłużej)
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,         
            patience=15,        
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_augmented_cnn_model.h5',
            monitor='val_acc',
            mode='max', 
            save_best_only=True,
            verbose=1
        ),
    ]
    
    print("\nRozpoczynam trening CNN + AUGMENTED DATASET...")
    print(" EKSPERYMENT: Optymalne parametry + 1.5x więcej danych na epokę")
    
    # Stwórz pusty plik checkpoint na początku (jeśli nie istnieje)
    checkpoint_path = os.path.join(os.getcwd(), 'best_augmented_cnn_model.h5')
    if not os.path.exists(checkpoint_path):
        print(f" Tworzę pusty plik checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'w') as f:
            f.write('')  # Pusty plik
    
    # Pomiar czasu treningu
    training_start_time = time.time()
    
    # EKSPERYMENT: Zwiększenie ilości danych na epokę
    # KLUCZOWE: steps_per_epoch może być większe niż len(train_generator)
    AUGMENTATION_MULTIPLIER = 1.5
    steps_per_epoch = int(len(train_generator) * AUGMENTATION_MULTIPLIER)
    
    print(f"\n ANALIZA DATASETU:")
    print(f"Oryginalne steps per epoch: {len(train_generator)}")
    print(f"Augmented steps per epoch: {steps_per_epoch}")
    print(f"Multiplier: {AUGMENTATION_MULTIPLIER}x")
    print(f"Efekt: Model widzi {AUGMENTATION_MULTIPLIER}x więcej augmentowanych wariantów")
    print(f"Baseline do pobicia: 92.92% accuracy (CNN baseline)")
    print(f"1.5x steps wynik: 93.46% accuracy (do pobicia)")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,  # KLUCZOWE - więcej kroków na epokę
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks_list,
        class_weight=class_weights,  # Wagi klas dla niezbalansowanych danych
        validation_steps=len(val_generator),
        verbose=1  # 1 = paski postępu TensorFlow
    )
    
    # Oblicz czas treningu
    training_time = time.time() - training_start_time
    training_time_minutes = training_time / 60
    print(f"\nCzas treningu: {training_time_minutes:.2f} minut ({training_time:.1f} sekund)")
    
    # Wykresy historii treningu
    plot_training_history(history)
    
    # Testowanie na najlepszym modelu
    print("\nTestowanie najlepszego modelu CNN + AUGMENTED DATASET...")
    print(" Porównanie z baseline: 92.92% accuracy do pobicia")
    
    #  ŁADOWANIE NAJLEPSZEGO MODELU Z CHECKPOINTU
    best_model_path = 'best_augmented_cnn_model.h5'
    try:
        if os.path.exists(best_model_path) and os.path.getsize(best_model_path) > 0:
            print(f" Ładuję najlepszy model z: {best_model_path}")
            model = tf.keras.models.load_model(best_model_path)
            print(" Najlepszy model załadowany!")
        else:
            print("  Brak poprawnego checkpointu – używam modelu z ostatniej epoki.")
    except Exception as e:
        print(f"  Nie udało się wczytać checkpointu: {e}\nUżywam modelu z ostatniej epoki.")

    
    # Predykcje na zbiorze testowym
    predictions = model.predict(test_generator, verbose=1)
    
    # Pobranie prawdziwych etykiet
    test_generator.reset()
    y_true = test_generator.classes
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=NUM_CLASSES)
    
    # Metryki
    metrics = calculate_metrics(y_true_one_hot, predictions, predictions)
    
    print("\n=== WYNIKI NA ZBIORZE TESTOWYM (CNN + AUGMENTED DATASET) ===")
    print(f" BASELINE DO POBICIA: 92.92% accuracy")
    print(f" 1.5x STEPS WYNIK: 93.46% accuracy")
    print(f" WYNIKI EKSPERYMENTU:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (średnia): {metrics['precision']:.4f}")
    print(f"Recall (średnia): {metrics['recall']:.4f}")
    print(f"F1-Score (średnia): {metrics['f1']:.4f}")
    print(f"ROC AUC (średnia): {metrics['roc_auc']:.4f}")
    
    # Porównanie z poprzednimi wynikami
    baseline_accuracy = 0.9292
    steps_1_5x_accuracy = 0.9346
    improvement_vs_baseline = metrics['accuracy'] - baseline_accuracy
    improvement_vs_1_5x = metrics['accuracy'] - steps_1_5x_accuracy
    
    print(f"\n PORÓWNANIE Z POPRZEDNIMI WYNIKAMI:")
    print(f"Baseline (1x steps): {baseline_accuracy:.4f}")
    print(f"1.5x steps: {steps_1_5x_accuracy:.4f}")
    print(f"2.0x steps: {metrics['accuracy']:.4f}")
    print(f"Poprawa vs baseline: {improvement_vs_baseline:+.4f} ({improvement_vs_baseline*100:+.2f}%)")
    print(f"Poprawa vs 1.5x: {improvement_vs_1_5x:+.4f} ({improvement_vs_1_5x*100:+.2f}%)")
    
    if improvement_vs_1_5x > 0.003:  # >0.3% improvement over 1.5x
        print(" DALSZE ZWIĘKSZANIE STEPS POMAGA!")
    elif improvement_vs_1_5x > 0:
        print(" MINIMALNA POPRAWA - Diminishing returns")
    else:
        print(" BRAK POPRAWY - Optimum było przy 1.5x steps")
    
    # Wyświetlenie metryk dla każdej klasy
    print("\n=== METRYKI DLA KAŻDEJ KLASY ===")
    class_names = list(test_generator.class_indices.keys())
    
    # Tabela z podstawowymi metrykami per klasa
    class_metrics_table = []
    for i, class_name in enumerate(class_names):
        class_metrics_table.append([
            class_name,
            f"{metrics['precision_per_class'][i]:.4f}",
            f"{metrics['recall_per_class'][i]:.4f}",
            f"{metrics['f1_per_class'][i]:.4f}",
            f"{metrics['roc_auc_per_class'][i]:.4f}",
            f"{metrics['support_per_class'][i]}"
        ])
    
    print(tabulate(class_metrics_table, 
                   headers=["Klasa", "Precision", "Recall", "F1-Score", "ROC AUC", "Support"], 
                   tablefmt="grid"))
    
    # Tabela z dodatkowymi metrykami medycznymi per klasa
    print("\n=== DODATKOWE METRYKI MEDYCZNE ===")
    medical_metrics_table = []
    for i, class_name in enumerate(class_names):
        medical_metrics_table.append([
            class_name,
            f"{metrics['sensitivity_per_class'][i]:.4f}",
            f"{metrics['specificity_per_class'][i]:.4f}",
            f"{metrics['ppv_per_class'][i]:.4f}",
            f"{metrics['npv_per_class'][i]:.4f}"
        ])
    
    print(tabulate(medical_metrics_table, 
                   headers=["Klasa", "Sensitivity (REC)", "Specificity (SPC)", "Precision (PRE)", "NPV"], 
                   tablefmt="grid"))
    
    # Macierz pomyłek
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    
    # ZAPISZ WYNIKI DO PLIKÓW
    print("\nRozpoczynam zapisywanie wyników...")
    save_results_to_files(metrics, history, model, class_names, training_time, y_true_one_hot, predictions, steps_per_epoch)
    
    print("\nTrening CNN + AUGMENTED DATASET zakończony!")
    print(" Sprawdź wyniki powyżej - czy zwiększenie datasetu poprawiło wyniki?")

def save_results_to_files(metrics, history, model, class_names, training_time, y_true_one_hot, predictions, steps_per_epoch, results_dir=None):
    """Zapisuje wyniki do plików do analizy"""
    from datetime import datetime
    
    # Utworzenie folderu na wyniki
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results1/cnn_augmented_model_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
    
    # 1. Konfiguracja modelu i wyniki (plik TXT)
    with open(f"{results_dir}/wyniki.txt", 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("WYNIKI MODELU CNN + AUGMENTED DATA\n")
        f.write("="*80 + "\n\n")
        
        f.write("PARAMETRY MODELU:\n")
        f.write(f"Architektura: CNN from scratch (identyczna jak baseline)\n")
        f.write(f"Augmentacja: OPTYMALNE parametry (rotation=25°, shifts=15%, zoom=15%)\n")
        f.write(f"Dataset multiplier: 1.5x (więcej kroków na epokę)\n")
        f.write(f"Oryginalne steps per epoch: {len(train_generator)}\n")
        f.write(f"Augmented steps per epoch: {steps_per_epoch}\n")
        f.write(f"Baseline do pobicia: 92.92% accuracy\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epoki: {EPOCHS}\n")
        f.write(f"Random Seed: {RANDOM_SEED} (reproducibility)\n")
        f.write(f"Epoki wykonane: {len(history.history['loss'])}\n")
        # TF 1.15 compatibility
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
        f.write(f"Najlepsza epoka: {int(np.argmax(history.history[val_acc_key]) + 1)}\n")
        f.write(f"Czas treningu: {training_time/60:.2f} minut ({training_time:.1f} sekund)\n\n")
        
        f.write("STRATEGIA EKSPERYMENTU:\n")
        f.write("- Augmentacja: IDENTYCZNE parametry jak najlepszy CNN baseline\n")
        f.write("- Rotation: 25° (optymalne z testów)\n")
        f.write("- Width/Height shift: 15% (optymalne z testów)\n") 
        f.write("- Zoom: 15% (optymalne z testów)\n")
        f.write("- Horizontal flip: True, Vertical flip: False\n")
        f.write("- JEDYNA RÓŻNICA: 1.5x więcej kroków na epokę\n")
        f.write("- Cel: Test czy więcej augmentowanych danych poprawia wyniki\n")
        f.write(f"- Baseline do pobicia: 92.92% accuracy\n\n")
        
        f.write("WYNIKI TESTOWE (ŚREDNIE):\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (średnia): {metrics['precision']:.4f}\n")
        f.write(f"Recall (średnia): {metrics['recall']:.4f}\n")
        f.write(f"F1-Score (średnia): {metrics['f1']:.4f}\n")
        f.write(f"ROC AUC (średnia): {metrics['roc_auc']:.4f}\n\n")
        
        f.write("WYNIKI TESTOWE DLA KAŻDEJ KLASY:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"\n{class_name.upper()}:\n")
            f.write(f"  Precision (PRE): {metrics['precision_per_class'][i]:.4f}\n")
            f.write(f"  Recall/Sensitivity (REC): {metrics['recall_per_class'][i]:.4f}\n")
            f.write(f"  Specificity (SPC): {metrics['specificity_per_class'][i]:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}\n")
            f.write(f"  ROC AUC: {metrics['roc_auc_per_class'][i]:.4f}\n")
            f.write(f"  NPV: {metrics['npv_per_class'][i]:.4f}\n")
            f.write(f"  Support: {metrics['support_per_class'][i]}\n")
        f.write("\n")
        
        f.write("HISTORIA TRENINGU:\n")
        # TF 1.15 compatibility
        acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
        f.write(f"Final Train Accuracy: {history.history[acc_key][-1]:.4f}\n")
        f.write(f"Final Train Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final Val Accuracy: {history.history[val_acc_key][-1]:.4f}\n")
        f.write(f"Final Val Loss: {history.history['val_loss'][-1]:.4f}\n")
        f.write("="*80 + "\n")
    
    # Historia treningu (CSV)
    import pandas as pd
    # TF 1.15 compatibility
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    history_df = pd.DataFrame({
        'epoch': range(1, len(history.history['loss']) + 1),
        'train_acc': history.history[acc_key],
        'train_loss': history.history['loss'],
        'val_acc': history.history[val_acc_key],
        'val_loss': history.history['val_loss']
    })
    history_df.to_csv(f"{results_dir}/training_history.csv", index=False)
    
    # Macierz pomyłek
    cm_df = pd.DataFrame(
        metrics['confusion_matrix'],
        index=class_names,
        columns=class_names
    )
    cm_df.to_csv(f"{results_dir}/confusion_matrix.csv")
    
    # Metryki per klasa (rozszerzone)
    class_metrics_df = pd.DataFrame({
        'class': class_names,
        'precision_PRE': metrics['precision_per_class'],
        'recall_sensitivity_REC': metrics['recall_per_class'],
        'specificity_SPC': metrics['specificity_per_class'],
        'f1_score': metrics['f1_per_class'],
        'roc_auc_AUC': metrics['roc_auc_per_class'],
        'npv_NPV': metrics['npv_per_class'],
        'support': metrics['support_per_class']
    })
    class_metrics_df.to_csv(f"{results_dir}/class_metrics.csv", index=False)
    
    # 5. ROC-AUC Curves
    print("\n Rysowanie krzywych ROC-AUC...")
    
    plt.figure(figsize=(12, 10))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    all_auc = []
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        # Oblicz ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_one_hot[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        all_auc.append(roc_auc)
        
        # Narysuj krzywą
        plt.plot(fpr, tpr, color=color, lw=3,
                 label=f'{class_name.upper()} (AUC = {roc_auc:.4f})')
    
    # Linia losowego klasyfikatora
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.8,
             label='Random Classifier (AUC = 0.500)')
    
    # Formatowanie
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    plt.title('ROC-AUC Curves - Brain Tumor Classification\n(CNN + Augmented Dataset)',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/roc_auc_curves.png", dpi=300, bbox_inches='tight')
    print(f" ROC-AUC curves zapisane: {results_dir}/roc_auc_curves.png")
    plt.close()
    
    # 6. ROC-AUC Curves (Zoomed) - High Performance Region
    plt.figure(figsize=(10, 8))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], predictions[:, i])
        plt.plot(fpr, tpr, color=color, lw=3, label=f'{class_name.upper()} (AUC = {all_auc[i]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Random')
    plt.xlim([0.0, 0.3])  # Zoom na lewą część
    plt.ylim([0.7, 1.0])  # Zoom na górną część
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC-AUC Curves (Zoomed) - High Performance Region\n(CNN + Augmented Dataset)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/roc_auc_curves_zoomed.png", dpi=300, bbox_inches='tight')
    print(f" ROC-AUC curves (zoomed) zapisane: {results_dir}/roc_auc_curves_zoomed.png")
    plt.close()
    
    # Wykresy (IDENTYCZNE jak w CNN)
    # Wykres 1: Accuracy vs Epoch
    plt.figure(figsize=(10, 6))
    # TF 1.15 compatibility
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    plt.plot(history.history[acc_key], label='Train Accuracy', linewidth=2, color='blue')
    plt.plot(history.history[val_acc_key], label='Val Accuracy', linewidth=2, color='red')
    plt.title('Accuracy vs Epoch (CNN + Augmented Data)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/accuracy_vs_epoch.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Wykres 2: Loss vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2, color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='red')
    plt.title('Loss vs Epoch (CNN + Augmented Data)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/loss_vs_epoch.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Wykres 3: Combined (Accuracy + Loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy
    # TF 1.15 compatibility
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    ax1.plot(history.history[acc_key], label='Train', linewidth=2, color='blue')
    ax1.plot(history.history[val_acc_key], label='Val', linewidth=2, color='red')
    ax1.set_title('Accuracy vs Epoch (CNN + Augmented)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2, color='blue')
    ax2.plot(history.history['val_loss'], label='Val', linewidth=2, color='red')
    ax2.set_title('Loss vs Epoch (CNN + Augmented)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (CNN + Augmented Data)')
    plt.savefig(f"{results_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nWyniki zapisane w: {results_dir}")
    print(f"Pliki: wyniki.txt, training_history.csv, confusion_matrix.csv, class_metrics.csv")
    print(f"Wykresy: roc_auc_curves.png, roc_auc_curves_zoomed.png, accuracy_vs_epoch.png, loss_vs_epoch.png, training_curves.png, confusion_matrix.png")
    
    return results_dir

if __name__ == "__main__":
    main()
