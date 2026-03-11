# Automatic Classification of Brain Tumor Types Using Deep Learning Methods

## Project Description

Master's thesis project focused on automatic brain tumor classification from MRI images using deep neural networks. The system classifies images into four categories:

- **Glioma** - malignant tumor originating from glial cells
- **Meningioma** - tumor of the meninges (brain and spinal cord membranes)  
- **Pituitary** - pituitary adenoma
- **No Tumor** - no tumor present

## Key Achievements

- **CNN from scratch**: 94.53% accuracy
- **VGG16 Transfer Learning**: 95.52% accuracy
- **Comprehensive MRI preprocessing pipeline**
- **Detailed medical metrics analysis** (sensitivity, specificity, NPV)
- **Reproducible research** with fixed seeds

## System Architecture

### 1. CNN from Scratch
```
Input (224×224×3) → Conv Blocks (5) → Dense Layers → Softmax (4 classes)
- 30.9M parametrów
- Dropout regularization
- Data augmentation
- Class weighting
```

### 2. VGG16 Transfer Learning
```
VGG16 (ImageNet) → Global Average Pooling → Dense Layers → Softmax (4 classes)
- Pre-trained feature extraction
- Two-phase training (frozen → fine-tuning)
- Custom classifier head
```

## Experimental Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| **VGG16 Transfer Learning** | **95.52%** | **95.53%** | **95.52%** | **95.49%** | **99.29%** | **20.31 min** |
| CNN from Scratch | 94.53% | 94.51% | 94.53% | 94.50% | 99.17% | 110.77 min |

### Per-class Results (VGG16 Transfer Learning - Best Model)

| Class | Precision | Recall | Specificity | F1-Score | ROC AUC | NPV |
|-------|-----------|--------|-------------|----------|---------|-----|
| **Glioma** | 94.00% | 99.65% | 97.84% | 96.74% | 99.88% | 99.88% |
| **Meningioma** | 96.73% | 89.43% | 99.06% | 92.94% | 99.17% | 96.79% |
| **Pituitary** | 94.14% | 99.31% | 97.82% | 96.66% | 99.63% | 99.75% |
| **No Tumor** | 96.59% | 92.06% | 98.93% | 94.27% | 99.13% | 97.42% |

### Per-class Results (CNN from Scratch)

| Class | Precision | Recall | Specificity | F1-Score | ROC AUC | NPV |
|-------|-----------|--------|-------------|----------|---------|-----|
| **Glioma** | 96.49% | 97.17% | 98.80% | 96.83% | 99.77% | 99.04% |
| **Meningioma** | 92.94% | 89.43% | 97.88% | 91.15% | 98.21% | 96.75% |
| **Pituitary** | 95.03% | 98.63% | 98.18% | 96.80% | 99.85% | 99.51% |
| **No Tumor** | 93.43% | 92.42% | 97.85% | 92.92% | 98.76% | 97.51% |

## Installation and Requirements

### System Requirements
```bash
Python 3.7+
TensorFlow 1.15
CUDA 10.0+ (dla GPU)
```

### Dependencies Installation
```bash
pip install tensorflow==1.15
pip install opencv-python
pip install scikit-image
pip install matplotlib seaborn
pip install pandas numpy
pip install Pillow tabulate tqdm
pip install scikit-learn
```

## Usage

### 1. Data Preprocessing
```bash
# Remove duplicates
python find_image_duplicates.py

# Preprocessing pipeline
python preprocess_images_debug.py --step 1   # Median filter
python preprocess_images_debug.py --step 2   # TV denoising  
python preprocess_images_debug.py --step 3   # CLAHE

# Normalization and conversion
python normalize_images.py
python convert_to_rgb.py
```

### 2. Model Training
```bash
# CNN from scratch
python cnn_brain_tumor.py

# CNN z augmentacją
python cnn_brain_tumor_augmented.py

# VGG16 Transfer Learning
python vgg_test.py
```

### 3. Results Analysis
```bash
# Preprocessing quality analysis
python preprocessing_quality_metrics.py

# Histogram analysis
python histogram_check.py
```

## Project Structure

```
├── README.md
├── Main Scripts
│   ├── cnn_brain_tumor.py              # CNN from scratch
│   ├── cnn_brain_tumor_augmented.py    # CNN with augmentation
│   ├── vgg_test.py                     # VGG16 Transfer Learning
│   └── ensemble_brain_tumor.py         # Ensemble methods
├── Preprocessing
│   ├── find_image_duplicates.py        # Remove duplicates
│   ├── preprocess_images_debug.py      # Preprocessing pipeline
│   ├── normalize_images.py             # Image normalization
│   ├── convert_to_rgb.py               # RGB 224x224 conversion
│   └── histogram_check.py              # Histogram analysis
├── Analysis and Metrics
│   ├── preprocessing_quality_metrics.py # Quality metrics
│   ├── analyze_augmentation_impact.py   # Augmentation analysis
│   └── cnn_architecture_table.md       # CNN architecture
├── Results
│   ├── results/                        # Experiment results
│   ├── Ottoni/wyniki_CNN/             # CNN results
│   ├── Ottoni/wyniki_VGG/             # VGG16 results
│   └── najlepsze_modele/              # Best models
└── Documentation
    ├── Ottoni/praca_dyplomowa_Ottoni.docx
    ├── Ottoni/Karta_dyplomowa.pdf
    └── bazy_zdjec.txt                 # Data sources
```

## Preprocessing Pipeline

### Stage 1: Noise Filtering
- **Median filter** - impulse noise removal
- **Total Variation denoising** - edge preservation

### Stage 2: Contrast Enhancement
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- Local histogram normalization

### Stage 3: Standardization
- **Z-score normalization** per image
- **RGB conversion** 224×224
- **Rescaling** to [0,1] for TensorFlow

## Data Augmentation

```python
# Augmentation parameters
rotation_range=15°
width_shift_range=0.02
height_shift_range=0.02  
horizontal_flip=True
zoom_range=0.02
```

## Hyperparameters

### CNN from Scratch
- **Learning Rate**: 5e-5
- **Batch Size**: 8 (najlepszy wynik)
- **Optimizer**: Adam + L2 regularization
- **Epochs**: 150 (early stopping)
- **Dropout**: 0.2-0.5 (progressive)

### VGG16 Transfer Learning
- **Phase 1 LR**: 1e-3 (head training)
- **Phase 2 LR**: 5e-5 (fine-tuning)
- **Batch Size**: 16
- **Frozen Layers**: 100% → 50%

## Dataset

- **Source**: Kaggle + Figshare (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Size**: ~3000 MRI images
- **Split**: Train/Validation/Test
- **Format**: JPEG, 224×224 RGB
- **Classes**: 4

## Evaluation Metrics

### Basic Metrics
- **Accuracy, Precision, Recall, F1-Score**
- **ROC AUC** (macro and per class)

### Medical Metrics
- **Sensitivity** (True Positive Rate)
- **Specificity** (True Negative Rate)  
- **NPV** (Negative Predictive Value)
- **Confusion Matrix**

## Best Models

Trained models available on [Google Drive](https://drive.google.com/drive/folders/16omLYLM_DkO1M9XeQx9ktNDbfnet-JAN?hl=pl):

- `best_vgg16_model.h5` - VGG16 Transfer Learning (95.52% acc)
- `best_cnn_model.h5` - CNN from scratch (94.53% acc)

## Reproducibility

The project ensures full reproducibility:
- **Fixed random seeds** (42)
- **Deterministic TensorFlow operations**
- **Saved hyperparameter configurations**
- **Detailed training logs**

## Contribution to the Field

1. **Comprehensive MRI preprocessing pipeline**
2. **CNN vs Transfer Learning comparison** on the same dataset
3. **Detailed medical metrics analysis**
4. **Open source implementation** with full documentation
5. **Reproducible research** with fixed parameters

## Contact

**Martyna Ottoni**
- Portfolio: [martyna-ottoni.vercel.app](https://martyna-ottoni.vercel.app)
- GitHub: [github.com/martynaottoni](https://github.com/martynaottoni)
- Project: [github.com/martynaottoni/BrainTumorAnalysis](https://github.com/martynaottoni/BrainTumorAnalysis)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

*Master's thesis completed as part of Biomedical Engineering studies, specialization: Artificial Intelligence*
