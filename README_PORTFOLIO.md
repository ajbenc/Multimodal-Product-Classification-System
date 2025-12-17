# 🛍️ Multimodal Product Classification System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art multimodal deep learning system for classifying Best Buy products using both image and text data. This project combines computer vision and natural language processing to achieve **87.56% accuracy** on product category classification.

**Portfolio Project by:** [Julian Bencina](https://github.com/ajbenc)

---

## 📊 Key Achievements

| Model Type | Accuracy | F1-Score | Key Feature |
|------------|----------|----------|-------------|
| 🔗 **Multimodal** | **87.56%** | **87%** | Early fusion of image + text embeddings |
| 📝 **Text-Only** | **94.16%** | **94%** | Zero overfitting with regularization |
| 🖼️ **Image-Only** | **82.00%** | **81%** | Synthetic data augmentation |

---

## 🎯 Project Overview

This project tackles the challenging problem of multimodal product classification by:

1. **Extracting rich embeddings** from product images using state-of-the-art vision models (ConvNeXtV2, ResNet50)
2. **Processing product descriptions** with transformer-based language models (MiniLM)
3. **Fusing multimodal features** through an early fusion architecture
4. **Training optimized MLPs** with regularization and advanced techniques

The system demonstrates how combining multiple data modalities can improve classification performance compared to single-modality approaches.

---

## 🏗️ Architecture

### Overall Pipeline

```
Product Data (Images + Text)
         ↓
    ┌────────────────┐
    │  Preprocessing │
    └────────────────┘
         ↓
    ┌─────────────────────────────────┐
    │   Feature Extraction Layer      │
    │                                 │
    │  ┌──────────────┐  ┌──────────┐│
    │  │ Vision Model │  │ NLP Model││
    │  │ (ConvNeXtV2) │  │ (MiniLM) ││
    │  └──────────────┘  └──────────┘│
    └─────────────────────────────────┘
         ↓           ↓
    Image Emb.   Text Emb.
      (768-d)     (384-d)
         ↓           ↓
    ┌─────────────────────┐
    │   Early Fusion      │
    │   Concatenation     │
    └─────────────────────┘
         ↓
    ┌─────────────────────┐
    │  MLP Classifier     │
    │  • BatchNorm        │
    │  • Dropout          │
    │  • Dense Layers     │
    └─────────────────────┘
         ↓
    Category Prediction
```

### Model Components

**Vision Models:**
- **ConvNeXtV2** (Hugging Face): Modern CNN architecture with 768-d embeddings
- **ResNet50** (Keras): Classic deep residual network with 2048-d embeddings

**NLP Model:**
- **sentence-transformers/all-MiniLM-L6-v2**: Efficient transformer producing 384-d semantic embeddings

**Fusion Strategy:**
- **Early Fusion**: Concatenate normalized embeddings before classification
- **MLP Classifier**: Deep neural network with dropout and batch normalization

---

## 📈 Results & Analysis

### Classification Performance

**Multimodal Model (Image + Text):**
- **Test Accuracy:** 87.56%
- **F1-Score:** 87%
- **Key Insight:** Combining modalities provides robust predictions, especially when one modality has ambiguous features

**Text-Only Model:**
- **Test Accuracy:** 94.16%
- **F1-Score:** 94%
- **Key Insight:** Product descriptions are highly informative; achieved zero overfitting through L2 regularization

**Image-Only Model:**
- **Test Accuracy:** 82.00%
- **F1-Score:** 81%
- **Key Insight:** Synthetic data augmentation significantly improved generalization

### Training Insights

1. **Text embeddings are highly discriminative** - The language model captures semantic product information effectively
2. **Early fusion works well** - Simple concatenation of normalized embeddings provides strong performance
3. **Regularization is critical** - L2 regularization and dropout prevent overfitting on limited data
4. **Synthetic data helps** - Generated training samples improved image model generalization

---

## 🛠️ Technologies & Tools

### Deep Learning Frameworks
- **TensorFlow 2.x** - Neural network training and deployment
- **Keras** - High-level API for model building
- **Transformers** (Hugging Face) - Pre-trained vision and language models

### Machine Learning Libraries
- **scikit-learn** - Classical ML models, preprocessing, metrics
- **NumPy & Pandas** - Data manipulation and analysis
- **Pillow** - Image processing

### Pre-trained Models
- **facebook/convnextv2-tiny-1k-224** - Vision transformer for image embeddings
- **ResNet50** - Deep residual network
- **sentence-transformers/all-MiniLM-L6-v2** - Sentence embeddings

### Development Tools
- **Jupyter Notebook** - Interactive development and analysis
- **pytest** - Unit testing
- **Docker** - Containerization for reproducibility

---

## 📂 Project Structure

```
├── src/
│   ├── vision_embeddings_tf.py    # Computer vision models & embedding extraction
│   ├── nlp_models.py              # NLP models & text embedding extraction
│   ├── classifiers_mlp.py         # MLP models & multimodal fusion
│   ├── classifiers_classic_ml.py  # Classical ML baselines
│   └── utils.py                   # Data preprocessing utilities
│
├── tests/
│   ├── test_vision_embeddings_tf.py
│   ├── test_nlp_models.py
│   ├── test_classifiers_mlp.py
│   ├── test_classifiers_classic_ml.py
│   └── test_utils.py
│
├── results/
│   ├── multimodal_results.csv     # Multimodal model predictions
│   ├── text_results.csv           # Text-only model predictions
│   └── image_results.csv          # Image-only model predictions
│
├── data/
│   └── processed_products_with_images.csv
│
├── AnyoneAI - Sprint Project 04.ipynb  # Main analysis notebook
├── requirements.txt
└── Dockerfile
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- TensorFlow 2.x
- 8GB+ RAM recommended
- GPU optional but recommended for training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ajbenc/multimodal-product-classification.git
   cd multimodal-product-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   
   Download the preprocessed dataset and images from [Google Drive](https://drive.google.com/file/d/14s2aDNTEWse86cWyLhvVIKmob6EbQrm_/view?usp=sharing) and extract to `data/`.

### Usage

#### 1. **Generate Embeddings**

```python
from src.vision_embeddings_tf import FoundationalCVModel, get_embeddings_df
from src.nlp_models import HuggingFaceEmbeddings

# Extract image embeddings
vision_model = FoundationalCVModel(backbone='convnextv2_tiny', mode='eval')
image_embeddings = get_embeddings_df(df, vision_model, directory='Embeddings/')

# Extract text embeddings
nlp_model = HuggingFaceEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
text_embeddings = nlp_model.get_embeddings_df(df, 'description')
```

#### 2. **Train Multimodal Classifier**

```python
from src.classifiers_mlp import train_mlp, create_early_fusion_model

# Train multimodal model
history, model = train_mlp(
    X_train, y_train,
    X_test, y_test,
    text_input_size=384,
    image_input_size=768,
    epochs=100,
    batch_size=32
)
```

#### 3. **Run Complete Pipeline**

```bash
# Launch Jupyter notebook
jupyter notebook

# Open and run: AnyoneAI - Sprint Project 04.ipynb
```

### Docker Support

```bash
# Build container
docker build -t product-classifier .

# Run container
docker run -p 8888:8888 -v $(pwd):/app product-classifier
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_classifiers_mlp.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- ✅ Image preprocessing and embedding extraction
- ✅ Text embedding generation
- ✅ Multimodal dataset creation
- ✅ MLP model architecture
- ✅ Training pipeline
- ✅ Classical ML baselines

---

## 💡 Key Implementation Details

### 1. **Handling Hugging Face Models in Keras**

ConvNeXtV2 from Transformers requires special handling:
- **Lambda layer wrapping** for Keras Functional API compatibility
- **Channels-first format** (NCHW) via `tf.transpose`
- **Safetensors compatibility** workaround with `use_safetensors=False`

```python
# ConvNeXtV2 integration
input_layer_transposed = Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2]))(input_layer)
outputs = Lambda(lambda x: self.base_model(x).pooler_output, output_shape=(768,))(input_layer_transposed)
```

### 2. **Regularization Strategy**

Achieved **zero overfitting** on text model through:
- L2 regularization (λ = 0.01)
- Dropout (50%)
- Batch normalization
- Early stopping with patience

### 3. **Data Augmentation**

Improved image model with synthetic data generation:
- Rotation, flipping, zooming
- Color jittering
- Gaussian noise injection

---

## 📊 Dataset

**Best Buy Product Dataset:**
- **Products:** ~7,000 items
- **Categories:** Multiple product categories
- **Features:**
  - High-resolution product images (224×224)
  - Product descriptions (text)
  - Metadata (price, manufacturer, etc.)

**Data Split:**
- Training: 80%
- Testing: 20%

---

## 🔮 Future Improvements

- [ ] **Late Fusion Architecture** - Compare with attention-based fusion
- [ ] **Larger Vision Models** - Test ViT-Large, CLIP for better image understanding
- [ ] **Fine-tuning** - Fine-tune pre-trained models on product data
- [ ] **Cross-Attention** - Implement cross-modal attention mechanisms
- [ ] **Active Learning** - Identify and label ambiguous samples
- [ ] **Deployment** - Create REST API for real-time predictions
- [ ] **Model Compression** - Quantization and pruning for production deployment

---

## 📝 Technical Challenges & Solutions

| Challenge | Solution | Result |
|-----------|----------|--------|
| ConvNeXtV2 incompatibility with Keras | Used Lambda layers + transpose | ✅ Successful integration |
| Dictionary key mismatch in tests | Standardized to 'text'/'image' | ✅ All tests passing |
| Text model overfitting | L2 regularization + dropout | ✅ Zero overfitting |
| Limited training data | Synthetic data augmentation | ✅ +5% image accuracy |
| Safetensors compatibility | `use_safetensors=False` flag | ✅ Model loading works |

---

## 📚 References

- [ConvNeXt V2 Paper](https://arxiv.org/abs/2301.00808) - Meta AI Research
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) - Efficient sentence embeddings
- [Early vs Late Fusion](https://arxiv.org/abs/2005.08271) - Multimodal fusion strategies

---

## 👨‍💻 About

**Author:** Julian Bencina  
**GitHub:** [@ajbenc](https://github.com/ajbenc)  
**LinkedIn:** [Connect with me](https://linkedin.com/in/julian-bencina)

This project demonstrates expertise in:
- 🧠 Deep Learning & Neural Networks
- 👁️ Computer Vision (CNNs, Vision Transformers)
- 📝 Natural Language Processing (Transformers, Embeddings)
- 🔗 Multimodal Machine Learning
- 🐍 Python & TensorFlow/Keras
- 🧪 Software Testing & Best Practices
- 📊 Data Science & ML Engineering

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Best Buy** for the product dataset
- **Hugging Face** for transformer models and infrastructure
- **TensorFlow/Keras** team for excellent documentation
- **Anyone AI** for project inspiration

---

## 📧 Contact

Questions or collaboration opportunities? Reach out!

- 📧 Email: your.email@example.com
- 💼 LinkedIn: [linkedin.com/in/julian-bencina](https://linkedin.com/in/julian-bencina)
- 🐱 GitHub: [@ajbenc](https://github.com/ajbenc)

---

<div align="center">
  <p>⭐ If you found this project interesting, please give it a star! ⭐</p>
  <p>Made with ❤️ and Python</p>
</div>
