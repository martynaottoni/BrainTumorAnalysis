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

# Wyłącz podwójne wyświetlanie pasków postępu w TF 1.x
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Dodatkowe ustawienia dla TF 1.x
try:
    tf.logging.set_verbosity(tf.logging.ERROR)
except:
    pass

# REPRODUCIBILITY - Ustawienie seed dla rzetelności naukowej
RANDOM_SEED = 150
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)  # TF 1.x syntax

# Dodatkowa kontrola nad losowością
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Ustawienia
print(f"CNN FROM SCRATCH - Brain Tumor Classification")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU dostępny: {tf.test.is_gpu_available()}")  # TF 1.x syntax
print(f"Random seed: {RANDOM_SEED} (dla reproducibility)")

# Ścieżki do danych
DATA_ROOT = Path(r"C:\1.Mart\RGB_224x224")
TRAIN_DIR = DATA_ROOT / "Training"
VAL_DIR = DATA_ROOT / "Validation"
TEST_DIR = DATA_ROOT / "Testing"

# Parametry treninguZ
BATCH_SIZE = 16
LEARNING_RATE = 0.00005
EPOCHS = 150  
NUM_CLASSES = 4  # glioma, meningioma, notumor, pituitary
IMG_SIZE = 224

# Augmentacja danych dla treningu (obrazy już są 224x224 RGB)
#  AUGMENTACJA TYLKO NA ZBIORZE TRENINGOWYM
#  NORMALIZACJA: uint8 [0,255] → float32 [0,1] (wymagane przez TensorFlow)h
train_datagen = ImageDataGenerator( 
    rotation_range=25,             
    width_shift_range=0.15,
    height_shift_range=0.15, 
    horizontal_flip=True, 
    vertical_flip=False,
    zoom_range=0.15,       
    rescale=1./255  
)

# Tylko normalizacja dla walidacji/testowania (BEZ AUGMENTACJI)
val_datagen = ImageDataGenerator(rescale=1./255)  # Konwersja uint8 [0,255] → float32 [0,1]
test_datagen = ImageDataGenerator(rescale=1./255)  # Osobny datagen dla testów

# Generatory danych (obrazy już są 224x224 RGB)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=RANDOM_SEED
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=RANDOM_SEED
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=RANDOM_SEED
)

print(f"Train samples: {train_generator.samples}")
print(f"Val samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Classes: {list(train_generator.class_indices.keys())}")

# Analiza balansu klas
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
            f"{percentage:.1f}%"
        ])
    
    print(tabulate(balance_table, headers=["Klasa", "Liczba próbek", "Procent"], tablefmt="grid"))
    
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

# Oblicz class weights dla niezbalansowanych klas
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

# CNN model w TensorFlow/Keras
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
    ax1.set_title('Loss podczas treningu')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy (TF 1.15 compatibility)
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    ax2.plot(history.history[acc_key], label='Train Accuracy', color='blue')
    ax2.plot(history.history[val_acc_key], label='Val Accuracy', color='red')
    ax2.set_title('Accuracy podczas treningu')
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
    plt.title('Macierz pomyłek', fontsize=14, fontweight='bold')
    plt.xlabel('Predykcja', fontsize=12)
    plt.ylabel('Rzeczywistość', fontsize=12)
    plt.tight_layout()
    plt.show()



def main():
    # Dodatkowe ustawienia dla TF 1.x - wyłącz podwójne paski postępu
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Wyłącz warningi TensorFlow i podwójne paski postępu
    try:
        tf.autograph.set_verbosity(0)
    except:
        pass  # Ignoruj błąd jeśli autograph nie jest dostępny w TF 1.x
    
    # Wyłącz podwójne wyświetlanie pasków postępu
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    print("Ładowanie danych...")
    
    # DIAGNOSTYKA: Sprawdź balans klas w train vs val
    print("\n" + "="*80)
    print("DIAGNOSTYKA: BALANS KLAS TRAIN vs VAL")
    print("="*80)
    
    # Sprawdzenie train
    train_classes = train_generator.classes
    train_unique, train_counts = np.unique(train_classes, return_counts=True)
    
    # Sprawdzenie val
    val_classes = val_generator.classes
    val_unique, val_counts = np.unique(val_classes, return_counts=True)
    
    class_names = list(train_generator.class_indices.keys())
    
    print("Train set:")
    for i, (class_idx, count) in enumerate(zip(train_unique, train_counts)):
        print(f"  {class_names[class_idx]}: {count}")
    
    print("\nVal set:")
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
    print("PARAMETRY SIECI CNN (TensorFlow/Keras)")
    print("="*80)
    model.summary()
    
    # Kompilacja modelu (TF 1.15 compatible)
    model.compile(
        optimizer=optimizers.Adam(lr=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks (zoptymalizowane dla stabilności)S
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,        
            min_delta=0.001,    # TOLERANCJA: zatrzyma jeśli poprawa < 0.001
            restore_best_weights=True,  # PRZYWRACA NAJLEPSZE WAGI
            verbose=1  # Włączone verbose żeby widzieć kiedy się zatrzymuje
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,        
            patience=15,        
            verbose=0  
        ),
        callbacks.ModelCheckpoint(
            'best_tf_model.h5',
            monitor='val_acc',
            mode='max', 
            save_best_only=True,
            verbose=1  # Włącz verbose żeby pokazać kiedy model jest zapisywany
        ),
    ]
    
    # Model będzie zapisywany w dwóch miejscach:
    print("Model będzie zapisywany:")
    print("   - Podczas treningu: best_tf_model.h5 (folder główny)")
    print("   - Po zakończeniu: results1/ (folder z wynikami)")
    

    
    print("\nRozpoczynam trening...")
    
    # Historia treningu (jak w PyTorch)
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0
    
    # Pomiar czasu treningu
    training_start_time = time.time()
    
    # Trening modelu (paski postępu z wszystkimi metrykami)
    print(f"Rozpoczynam trening: {EPOCHS} epok, batch_size={BATCH_SIZE}")
    
    # Dodaj custom callback do wyświetlania postępu
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                print(f"Epoch {epoch+1}/{EPOCHS} - "
                      f"loss: {logs.get('loss', 0):.4f} - "
                      f"acc: {logs.get('acc', logs.get('accuracy', 0)):.4f} - "
                      f"val_loss: {logs.get('val_loss', 0):.4f} - "
                      f"val_acc: {logs.get('val_acc', logs.get('val_accuracy', 0)):.4f}")
    
    # Dodaj custom callback do listy
   # callbacks_list.append(ProgressCallback())
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    # Obliczanie czas treningu
    training_time = time.time() - training_start_time
    training_time_minutes = training_time / 60
    print(f"\nCzas treningu: {training_time_minutes:.2f} minut ({training_time:.1f} sekund)")
    
    # Sprawdzenie czy model został zapisany w folderze głównym
    if os.path.exists('best_tf_model.h5'):
        model_size = os.path.getsize('best_tf_model.h5') / (1024*1024)  # MB
        print(f"Najlepszy model zapisany w folderze głównym: best_tf_model.h5 ({model_size:.1f} MB)")
    else:
        print("Model nie został zapisany w folderze głównym")
    
    # Zapisanie historii do list - TF 1.15 compatibility
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    train_losses = history.history['loss']
    train_accs = history.history[acc_key]
    val_losses = history.history['val_loss']
    val_accs = history.history[val_acc_key]
    
    # Wykresy historii treningu
    plot_training_history(history)
    
    # Testowanie na najlepszym modelu
    print("\nTestowanie najlepszego modelu...")
    
    # ŁADOWANIE NAJLEPSZEGO MODELU Z CHECKPOINTU
    best_model_path = 'best_tf_model.h5'
    if os.path.exists(best_model_path):
        print(f"Ładuję najlepszy model z: {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
        print("Najlepszy model załadowany!")
    else:
        print("Plik najlepszego modelu nie istnieje, używam modelu z ostatniej epoki")
    
    # Reset generatorów przed testowaniem
    print("Reset generatorów...")
    train_generator.reset()
    val_generator.reset()
    test_generator.reset()
    
    # Predykcje na zbiorze testowym
    print("Wykonuję predykcje...")
    predictions = model.predict(test_generator, verbose=0)  
    
    # Pobranie prawdziwych etykiet
    y_true = test_generator.classes
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=NUM_CLASSES)
    
    # Metryki
    metrics = calculate_metrics(y_true_one_hot, predictions, predictions)
    
    print("\n=== WYNIKI NA ZBIORZE TESTOWYM ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (średnia): {metrics['precision']:.4f}")
    print(f"Recall (średnia): {metrics['recall']:.4f}")
    print(f"F1-Score (średnia): {metrics['f1']:.4f}")
    print(f"ROC AUC (średnia): {metrics['roc_auc']:.4f}")
    
    # Wyświetlenie metryk dla każdej klasy
    print("\n=== METRYKI DLA KAŻDEJ KLASY ===")
    class_names = list(test_generator.class_indices.keys())
    
    # Tabela z podstawowymi metrykami per klasa
    from tabulate import tabulate
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
    class_names = list(test_generator.class_indices.keys())
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    
    # ZAPISanie WYNIKI DO PLIKÓW
    print("\nRozpoczynam zapisywanie wyników...")
    results_dir = save_results_to_files(metrics, history, model, class_names, training_time, y_true_one_hot, predictions)
    
    # Model jest już zapisany w folderze results1 przez save_results_to_files
    print("Model został zapisany w folderze results1")
    
    print("\nTrening zakończony!")



def save_results_to_files(metrics, history, model, class_names, training_time, y_true_one_hot, predictions, results_dir=None):
    """Zapisuje wyniki do plików do analizy"""
    from datetime import datetime
    
    # Utworzenie folderu na wyniki
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results1/model_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
    
    # 1. Konfiguracja modelu i wyniki (plik TXT)
    with open(f"{results_dir}/wyniki.txt", 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("WYNIKI MODELU\n")
        f.write("="*80 + "\n\n")
        
        f.write("PARAMETRY MODELU:\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epoki: {EPOCHS}\n")
        f.write(f"Random Seed: {RANDOM_SEED} (reproducibility)\n")
        f.write(f"Augmentacja: rotation={15}, zoom={0.02}\n")
        # TF 1.15 compatibility
        acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
        f.write(f"Epoki wykonane: {len(history.history['loss'])}\n")
        f.write(f"Najlepsza epoka: {int(np.argmax(history.history[val_acc_key]) + 1)}\n")
        f.write(f"Czas treningu: {training_time/60:.2f} minut ({training_time:.1f} sekund)\n\n")
        
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
        f.write(f"Final Train Accuracy: {history.history[acc_key][-1]:.4f}\n")
        f.write(f"Final Train Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final Val Accuracy: {history.history[val_acc_key][-1]:.4f}\n")
        f.write(f"Final Val Loss: {history.history['val_loss'][-1]:.4f}\n")
        f.write(f"\nZAPISANY MODEL:\n")
        f.write(f"Ścieżka: {results_dir}/best_tf_model.h5\n")
        f.write(f"Typ: Najlepszy model (z najlepszą walidacją)\n")
        f.write("="*80 + "\n")
    
    # 3. Historia treningu (CSV)
    import pandas as pd
    history_df = pd.DataFrame({
        'epoch': range(1, len(history.history['loss']) + 1),
        'train_acc': history.history[acc_key],
        'train_loss': history.history['loss'],
        'val_acc': history.history[val_acc_key],
        'val_loss': history.history['val_loss']
    })
    history_df.to_csv(f"{results_dir}/training_history.csv", index=False)
    
    # 4. Macierz pomyłek
    cm_df = pd.DataFrame(
        metrics['confusion_matrix'],
        index=class_names,
        columns=class_names
    )
    cm_df.to_csv(f"{results_dir}/confusion_matrix.csv")
    
    # 5. Metryki per klasa (rozszerzone)
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
    print("\nRysowanie krzywych ROC-AUC...")
    
    plt.figure(figsize=(12, 10))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    all_auc = []
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        # Oblicz ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_one_hot[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        all_auc.append(roc_auc)
        
        # Rysowanie krzywej
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
    plt.title('ROC-AUC Curves - Brain Tumor Classification\n(CNN from Scratch)',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/roc_auc_curves.png", dpi=300, bbox_inches='tight')
    print(f"ROC-AUC curves zapisane: {results_dir}/roc_auc_curves.png")
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
    plt.title('ROC-AUC Curves (Zoomed) - High Performance Region\n(CNN from Scratch)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/roc_auc_curves_zoomed.png", dpi=300, bbox_inches='tight')
    print(f"ROC-AUC curves (zoomed) zapisane: {results_dir}/roc_auc_curves_zoomed.png")
    plt.close()
    
    # 7. Wykresy treningu
    # Wykres 1: Accuracy vs Epoch
    plt.figure(figsize=(10, 6))
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    plt.plot(history.history[acc_key], label='Train Accuracy', linewidth=2, color='blue')
    plt.plot(history.history[val_acc_key], label='Val Accuracy', linewidth=2, color='red')
    plt.title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
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
    plt.title('Loss vs Epoch', fontsize=14, fontweight='bold')
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
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    ax1.plot(history.history[acc_key], label='Train', linewidth=2, color='blue')
    ax1.plot(history.history[val_acc_key], label='Val', linewidth=2, color='red')
    ax1.set_title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2, color='blue')
    ax2.plot(history.history['val_loss'], label='Val', linewidth=2, color='red')
    ax2.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
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
    plt.title('Confusion Matrix')
    plt.savefig(f"{results_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nWyniki zapisane w: {results_dir}")
    print(f"Pliki: wyniki.txt, training_history.csv, confusion_matrix.csv, class_metrics.csv")
    print(f"Wykresy: roc_auc_curves.png, roc_auc_curves_zoomed.png, accuracy_vs_epoch.png, loss_vs_epoch.png, training_curves.png, confusion_matrix.png")
    
    # ZAPISZ NAJLEPSZY MODEL
    print("\n Zapisuję najlepszy model...")
    model_path = f"{results_dir}/best_tf_model.h5"
    
    try:
        model.save(model_path)
        model_size = os.path.getsize(model_path) / (1024*1024)  # MB
        print(f"Model zapisany: {model_path} ({model_size:.1f} MB)")
        print(f"Ścieżka: {os.path.abspath(model_path)}")
    except Exception as e:
        print(f"Błąd podczas zapisu modelu: {e}")
        print("Próbuję alternatywną metodę zapisu...")
        
        try:
            # Alternatywna metoda zapisu
            model.save_weights(f"{results_dir}/model_weights.h5")
            print(f"Wagi modelu zapisane: {results_dir}/model_weights.h5")
        except Exception as e2:
            print(f"Błąd podczas zapisu wag: {e2}")
    
    return results_dir

if __name__ == "__main__":
    main() 