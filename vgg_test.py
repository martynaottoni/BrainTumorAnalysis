"""
VGG16 + Transfer Learning dla klasyfikacji guzów mózgu
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

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

# REPRODUCIBILITY
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

print(f"VGG16 + TRANSFER LEARNING - Brain Tumor Classification")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU dostępny: {tf.test.is_gpu_available()}")
print(f"Random seed: {RANDOM_SEED} (dla reproducibility)")

# ============================================================================
# KONFIGURACJA
# ============================================================================

class VGGConfig:
    """Konfiguracja VGG16"""
    
    # Podstawowe parametry
    BATCH_SIZE = 32
    NUM_CLASSES = 4
    IMG_SIZE = 224
    RANDOM_SEED = 42
    
    # Ścieżki do danych
    DATA_ROOT = Path(r"C:\1.Mart\RGB_224x224")
    TRAIN_DIR = DATA_ROOT / "Training"
    VAL_DIR = DATA_ROOT / "Validation"
    TEST_DIR = DATA_ROOT / "Testing"
    
    # Parametry treningu
    EPOCHS = 50
    PHASE1_EPOCHS = 15  # Tylko head
    PHASE2_EPOCHS = EPOCHS - PHASE1_EPOCHS  # Fine-tuning
    
    # Learning rates
    PHASE1_LR = 0.001  # Head training
    PHASE2_LR = 5e-5  # Fine-tuning
    
    # Zamrażanie warstw
    PHASE1_FREEZE_PERCENTAGE = 100  # Wszystkie warstwy zamrożone
    PHASE2_FREEZE_PERCENTAGE = 50   # 25% warstw zamrożone
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    
    # Augmentacja
    ROTATION_RANGE = 25
    WIDTH_SHIFT_RANGE = 0.15
    HEIGHT_SHIFT_RANGE = 0.15
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = False
    ZOOM_RANGE = 0.15
    
    # Wzmocnienie wag
    MENINGIOMA_BOOST = 2.0
    PITUITARY_BOOST = 3.0
    
    # Walidacja krzyżowa (nie używana - mamy wydzielony zbiór walidacyjny)
    VALIDATION_SPLIT = 0.2
    
    @classmethod
    def print_config(cls):
        """Wyświetla konfigurację"""
        print("\n" + "="*80)
        print("KONFIGURACJA VGG16")
        print("="*80)
        print(f"Epoki: {cls.EPOCHS} (Faza 1: {cls.PHASE1_EPOCHS}, Faza 2: {cls.PHASE2_EPOCHS})")
        print(f"Learning Rates: Faza 1 = {cls.PHASE1_LR}, Faza 2 = {cls.PHASE2_LR}")
        print(f"Zamrażanie: Faza 1 = {cls.PHASE1_FREEZE_PERCENTAGE}%, Faza 2 = {cls.PHASE2_FREEZE_PERCENTAGE}%")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Walidacja: Wydzielony zbiór walidacyjny")
        print(f"Wzmocnienie wag: Meningioma ×{cls.MENINGIOMA_BOOST}, Pituitary ×{cls.PITUITARY_BOOST}")
        print("="*80)

# Użyj konfiguracji
DATA_ROOT = VGGConfig.DATA_ROOT
TRAIN_DIR = VGGConfig.TRAIN_DIR
VAL_DIR = VGGConfig.VAL_DIR
TEST_DIR = VGGConfig.TEST_DIR
BATCH_SIZE = VGGConfig.BATCH_SIZE
EPOCHS = VGGConfig.EPOCHS
NUM_CLASSES = VGGConfig.NUM_CLASSES
IMG_SIZE = VGGConfig.IMG_SIZE

# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def preprocess_vgg16(img):
    """Aplikuje tylko preprocessing VGG16 (obrazy już są 224x224)"""
    return preprocess_input(img)

def set_trainable_layers(model, freeze_percentage):
    """Ustawia ile % warstw VGG16 ma być zamrożonych"""
    # VGG16 ma 19 warstw (bez input i top)
    vgg_layers = model.layers[1:20]  # warstwy VGG16 (19 warstw)
    total = len(vgg_layers)
    freeze_layers = int(total * freeze_percentage / 100)

    for i, layer in enumerate(vgg_layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False  # BN zawsze zamrożone
        elif i < freeze_layers:
            layer.trainable = False
        else:
            layer.trainable = True

    print(f"Zamrożone warstwy: {freeze_layers}/{total} ({freeze_percentage}%)")
    print(f"Trenowane warstwy: {total - freeze_layers}/{total} ({100 - freeze_percentage}%)")
    
    return freeze_layers, total

def calculate_class_weights():
    """Oblicza wagi klas z wzmocnieniem problematycznych klas"""
    class_counts = train_generator.classes
    unique, counts = np.unique(class_counts, return_counts=True)
    
    total_samples = len(class_counts)
    class_weights = {}
    class_names = list(train_generator.class_indices.keys())
    
    for class_idx, count in zip(unique, counts):
        weight = total_samples / (len(unique) * count)
        
        class_name = class_names[class_idx]
        if class_name == 'meningioma':
            weight *= VGGConfig.MENINGIOMA_BOOST
        elif class_name == 'pituitary':
            weight *= VGGConfig.PITUITARY_BOOST
        
        class_weights[class_idx] = weight
    
    print("\nWagi klas (z wzmocnieniem):")
    for class_idx, weight in class_weights.items():
        class_name = class_names[class_idx]
        boost_info = ""
        if class_name == 'meningioma':
            boost_info = f" (×{VGGConfig.MENINGIOMA_BOOST})"
        elif class_name == 'pituitary':
            boost_info = f" (×{VGGConfig.PITUITARY_BOOST})"
        print(f"  {class_name}: {weight:.3f}{boost_info}")
    
    return class_weights

# ============================================================================
# PRZYGOTOWANIE DANYCH
# ============================================================================

# Augmentacja + preprocessing VGG16
train_datagen = ImageDataGenerator(
    rotation_range=VGGConfig.ROTATION_RANGE,
    width_shift_range=VGGConfig.WIDTH_SHIFT_RANGE,
    height_shift_range=VGGConfig.HEIGHT_SHIFT_RANGE,
    horizontal_flip=VGGConfig.HORIZONTAL_FLIP,
    vertical_flip=VGGConfig.VERTICAL_FLIP,
    zoom_range=VGGConfig.ZOOM_RANGE,
    preprocessing_function=preprocess_vgg16
)

# Tylko preprocessing VGG16
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_vgg16)

# Generatory danych (używamy bezpośrednio folderów)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Train samples: {train_generator.samples}")
print(f"Val samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Classes: {list(train_generator.class_indices.keys())}")

# ============================================================================
# MODEL VGG16
# ============================================================================

def create_vgg16_model(freeze_percentage=100, num_classes=4):
    """Tworzy model VGG16 + Transfer Learning"""
    print("\n" + "="*80)
    print(f"BUDOWANIE MODELU VGG16 + TRANSFER LEARNING")
    print(f"ZAMROŻONE WARSTWY: {freeze_percentage}%")
    print("="*80)
    
    # 1. Załaduj pre-trained VGG16
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(VGGConfig.IMG_SIZE, VGGConfig.IMG_SIZE, 3),
        pooling=None
    )
    
    print(f"VGG16 base model załadowany (parametry: {base_model.count_params():,})")
    
    # 2. Dodaj custom classifier
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(1024, activation='relu', kernel_initializer='glorot_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_initializer='glorot_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer='glorot_normal')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    print("Custom classifier dodany")
    
    # 3. Ustaw zamrożone warstwy
    freeze_layers, total_layers = set_trainable_layers(model, freeze_percentage)
    
    print(f"Total parameters: {model.count_params():,}")
    trainable_count = sum([np.prod(var.get_shape().as_list()) for var in model.trainable_variables])
    print(f"Trainable parameters: {trainable_count:,}")
    
    return model

# ============================================================================
# FUNKCJE METRYK
# ============================================================================

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
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='weighted'
    )
    
    # Metryki dla każdej klasy osobno
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average=None
    )
    
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

# ============================================================================
# GŁÓWNA FUNKCJA
# ============================================================================

def main():
    # Wyłącz warningi TensorFlow
    tf.autograph.set_verbosity(0)
    
    print("Ładowanie danych...")
    
    # Oblicz wagi klas
    class_weights = calculate_class_weights()
    
    # Wyświetl konfigurację
    VGGConfig.print_config()
    
    # Tworzenie modelu - Faza 1
    model = create_vgg16_model(freeze_percentage=VGGConfig.PHASE1_FREEZE_PERCENTAGE, num_classes=VGGConfig.NUM_CLASSES)
    
    # Wyświetlenie parametrów modelu
    print("\n" + "="*80)
    print(f"PARAMETRY MODELU VGG16 + TRANSFER LEARNING")
    print(f"ZAMROŻONE WARSTWY: {VGGConfig.PHASE1_FREEZE_PERCENTAGE}%")
    print("="*80)
    model.summary()
    
    # Faza 1 – tylko head
    model.compile(optimizer=optimizers.Adam(learning_rate=VGGConfig.PHASE1_LR),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05), 
                  metrics=['accuracy'])
    print(f"Faza 1: Learning rate {VGGConfig.PHASE1_LR} (tylko head)")
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=VGGConfig.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=VGGConfig.REDUCE_LR_FACTOR,
            patience=VGGConfig.REDUCE_LR_PATIENCE,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_vgg16_model.h5',
            monitor='val_acc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
    ]
    
    print(f"\n=== TRENING 1: GŁOWICA (epoki 1-{VGGConfig.PHASE1_EPOCHS}) ===")
    print("VGG16 zamrożony, trenuje się tylko głowica...")
    
    # Pomiar czasu treningu
    training_start_time = time.time()
    
    # EKSPERYMENT: Zwiększenie ilości danych na epokę (1.5x)
    AUGMENTATION_MULTIPLIER = 1.5
    steps_per_epoch = int(len(train_generator) * AUGMENTATION_MULTIPLIER)
    
    print(f"\nANALIZA DATASETU:")
    print(f"Oryginalne steps per epoch: {len(train_generator)}")
    print(f"Augmented steps per epoch: {steps_per_epoch}")
    print(f"Multiplier: {AUGMENTATION_MULTIPLIER}x")
    print(f"Efekt: Model widzi {AUGMENTATION_MULTIPLIER}x więcej augmentowanych wariantów")
    
    # Trening 1: Głowica
    history1 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,  #  więcej kroków na epokę- test augmentacji z powiekszeniem zbioru
        epochs=VGGConfig.PHASE1_EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    print("\n=== ODMRZAŻANIE VGG16 ===")
    print("Odmrażanie warstw VGG16")
    
    # Ustaw warstwy dla fazy 2
    freeze_layers, total_layers = set_trainable_layers(model, VGGConfig.PHASE2_FREEZE_PERCENTAGE)

    # Faza 2 – fine-tuning
    model.compile(optimizer=optimizers.Adam(learning_rate=VGGConfig.PHASE2_LR),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=['accuracy'])
    print("Model rekompilowany po odmrożeniu VGG16")
    print(f"Learning rate dla VGG16: {VGGConfig.PHASE2_LR} (Adam)")

    print(f"\n=== TRENING 2: FINE-TUNING (epoki {VGGConfig.PHASE1_EPOCHS+1}-{VGGConfig.EPOCHS}) ===")
    print("VGG16 odmrożony, fine-tuning całego modelu...")

    # Trening 2: Fine-tuning
    if VGGConfig.EPOCHS > VGGConfig.PHASE1_EPOCHS:
        history2 = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,  # KLUCZOWE - więcej kroków na epokę
            epochs=VGGConfig.EPOCHS,
            initial_epoch=VGGConfig.PHASE1_EPOCHS,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        # Połączenie historii treningu
        history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'accuracy': (history1.history.get('accuracy', history1.history.get('acc', [])) + 
                        history2.history.get('accuracy', history2.history.get('acc', []))),
            'val_accuracy': (history1.history.get('val_accuracy', history1.history.get('val_acc', [])) + 
                           history2.history.get('val_accuracy', history2.history.get('val_acc', [])))
        }
    else:
        # Tylko pierwsza faza treningu
        history = {
            'loss': history1.history['loss'],
            'val_loss': history1.history['val_loss'],
            'accuracy': history1.history.get('accuracy', history1.history.get('acc', [])),
            'val_accuracy': history1.history.get('val_accuracy', history1.history.get('val_acc', []))
        }
    
    # Oblicz czas treningu
    training_time = time.time() - training_start_time
    training_time_minutes = training_time / 60
    print(f"\nCzas treningu: {training_time_minutes:.2f} minut ({training_time:.1f} sekund)")
    
    # Testowanie na najlepszym modelu
    print("\nTestowanie najlepszego modelu...")
    
    # Ładowanie najlepszego modelu
    best_model_path = 'best_vgg16_model.h5'
    if os.path.exists(best_model_path):
        print(f"Ładowanie najlepszego modelu z: {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
        print("Najlepszy model załadowany!")
    else:
        print("Plik najlepszego modelu nie istnieje, używam modelu z ostatniej epoki")
    
    # Reset generatorów
    train_generator.reset()
    val_generator.reset()
    test_generator.reset()
    
    # Predykcje na zbiorze testowym
    print("Wykonuję predykcje...")
    predictions = model.predict(test_generator, verbose=0)
    
    # Pobranie prawdziwych etykiet
    test_generator.reset()
    y_true = test_generator.classes
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=NUM_CLASSES)
    
    # Metryki
    metrics = calculate_metrics(y_true_one_hot, predictions, predictions)
    
    print("\n=== WYNIKI NA ZBIORZE TESTOWYM (VGG16 + Transfer Learning) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (średnia): {metrics['precision']:.4f}")
    print(f"Recall (średnia): {metrics['recall']:.4f}")
    print(f"F1-Score (średnia): {metrics['f1']:.4f}")
    print(f"ROC AUC (średnia): {metrics['roc_auc']:.4f}")
    
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Liczba próbek'})
    plt.title('Macierz pomyłek (VGG16+TL)', fontsize=14, fontweight='bold')
    plt.xlabel('Predykcja', fontsize=12)
    plt.ylabel('Rzeczywistość', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # ZAPISZ WYNIKI
    print("\nRozpoczynam zapisywanie wyników...")
    save_results_to_files(metrics, history, model, class_names, training_time, y_true_one_hot, predictions)
    
    print("\nTrening VGG16 + Transfer Learning zakończony!")

def save_results_to_files(metrics, history, model, class_names, training_time, y_true_one_hot, predictions, results_dir=None):
    """Zapisuje wyniki do plików do analizy (IDENTYCZNE jak w CNN)"""
    from datetime import datetime
    
    # Utworzenie folderu na wyniki
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results1/vgg16_model_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Konfiguracja modelu i wyniki (plik TXT)
    with open(f"{results_dir}/wyniki.txt", 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("WYNIKI MODELU VGG16 + TRANSFER LEARNING\n")
        f.write("="*80 + "\n\n")
        
        f.write("PARAMETRY MODELU:\n")
        f.write(f"Architektura: VGG16 (pre-trained) + Custom Classifier\n")
        f.write(f"Transfer Learning: ImageNet → Brain Tumor\n")
        f.write(f"Źródło danych: {VGGConfig.DATA_ROOT}\n")
        f.write(f"Preprocessing: Obrazy już przetworzone (224x224 RGB)\n")
        f.write(f"Walidacja: Wydzielony zbiór walidacyjny\n")
        f.write(f"Zamrożone warstwy: Faza 1 = {VGGConfig.PHASE1_FREEZE_PERCENTAGE}%, Faza 2 = {VGGConfig.PHASE2_FREEZE_PERCENTAGE}%\n")
        f.write(f"Learning Rate: Faza 1 = {VGGConfig.PHASE1_LR}, Faza 2 = {VGGConfig.PHASE2_LR}\n")
        f.write(f"Batch Size: {VGGConfig.BATCH_SIZE}\n")
        f.write(f"Epoki: {VGGConfig.EPOCHS} (Faza 1: {VGGConfig.PHASE1_EPOCHS}, Faza 2: {VGGConfig.PHASE2_EPOCHS})\n")
        f.write(f"Random Seed: {VGGConfig.RANDOM_SEED} (reproducibility)\n")
        f.write(f"Augmentacja: IDENTYCZNA jak CNN from scratch\n")
        f.write(f"Epoki wykonane: {len(history['loss'])}\n")
        f.write(f"Najlepsza epoka: {int(np.argmax(history['val_accuracy']) + 1)}\n")
        f.write(f"Czas treningu: {training_time/60:.2f} minut ({training_time:.1f} sekund)\n")
        f.write(f"Klasyfikator: IDENTYCZNY jak CNN from scratch (fair comparison)\n\n")
        
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
        f.write(f"Final Train Accuracy: {history['accuracy'][-1]:.4f}\n")
        f.write(f"Final Train Loss: {history['loss'][-1]:.4f}\n")
        f.write(f"Final Val Accuracy: {history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Final Val Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"\nZAPISANY MODEL:\n")
        f.write(f"Ścieżka: {results_dir}/best_vgg16_model.h5\n")
        f.write(f"Typ: Najlepszy model (z najlepszą walidacją)\n")
        f.write("="*80 + "\n")
    
    # 3. Historia treningu (CSV)
    import pandas as pd
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['loss']) + 1),
        'train_acc': history['accuracy'],
        'train_loss': history['loss'],
        'val_acc': history['val_accuracy'],
        'val_loss': history['val_loss']
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
    plt.title('ROC-AUC Curves - Brain Tumor Classification\n(VGG16 + Transfer Learning)',
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
    plt.title('ROC-AUC Curves (Zoomed) - High Performance Region\n(VGG16 + Transfer Learning)', 
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
    plt.plot(history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
    plt.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2, color='red')
    plt.title('Accuracy vs Epoch (VGG16+TL)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/accuracy_vs_epoch.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Wykres 2: Loss vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss', linewidth=2, color='blue')
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2, color='red')
    plt.title('Loss vs Epoch (VGG16+TL)', fontsize=14, fontweight='bold')
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
    ax1.plot(history['accuracy'], label='Train', linewidth=2, color='blue')
    ax1.plot(history['val_accuracy'], label='Val', linewidth=2, color='red')
    ax1.set_title('Accuracy vs Epoch (VGG16+TL)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history['loss'], label='Train', linewidth=2, color='blue')
    ax2.plot(history['val_loss'], label='Val', linewidth=2, color='red')
    ax2.set_title('Loss vs Epoch (VGG16+TL)', fontsize=14, fontweight='bold')
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
    plt.title('Confusion Matrix (VGG16+TL)')
    plt.savefig(f"{results_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nWyniki zapisane w: {results_dir}")
    print(f"Pliki: wyniki.txt, training_history.csv, confusion_matrix.csv, class_metrics.csv")
    print(f"Wykresy: roc_auc_curves.png, roc_auc_curves_zoomed.png, accuracy_vs_epoch.png, loss_vs_epoch.png, training_curves.png, confusion_matrix.png")
    
    # ZAPISZ NAJLEPSZY MODEL
    print("\n Zapisuję najlepszy model...")
    model_path = f"{results_dir}/best_vgg16_model.h5"
    
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
