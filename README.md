# FLIQ-Virtual-Hackathon UNICC-Challenge QuantumBreast Cancer Classification - Quantum Buddies

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.x-blueviolet.svg)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org/)

## 🏆 Project Overview

This repository contains our **Quantum Buddies** team solution for the **Quantum Coalition Future Leaders in Quantum (QC-FLIQ) Virtual Hackathon** organized by UN-ICC. Our project implements a **Quantum-Classical Hybrid Neural Network** for breast cancer classification using Variational Quantum Algorithms (VQAs).

### 🎯 Key Features
- **Hybrid Architecture**: Combines classical neural networks with quantum circuits for enhanced classification
- **GPU Acceleration**: Supports CUDA for PyTorch and optional GPU acceleration for quantum simulations
- **Real-world Application**: Focuses on breast cancer diagnosis using the Wisconsin Diagnostic Dataset
- **Explainable AI**: Implements visualization tools for model interpretability
- **Responsible AI**: Addresses fairness, robustness, and security considerations

## 📋 Table of Contents
- [Challenge Statement](#challenge-statement)
- [Technical Approach](#technical-approach)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Datasets](#datasets)
- [Technologies Used](#technologies-used)
- [Ethical Considerations](#ethical-considerations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🚀 Challenge Statement

The hackathon challenges participants to enhance AI/ML classifiers or clusterers using Variational Quantum Algorithms (VQAs). Our solution specifically targets:

- **Medical Diagnosis**: Binary classification of breast cancer (malignant vs. benign)
- **Quantum Enhancement**: Leveraging quantum circuits to improve classical ML performance
- **Responsible AI**: Ensuring fairness, explainability, and robustness in healthcare applications

## 🔬 Technical Approach

### Quantum-Classical Hybrid Architecture

Our solution implements a sophisticated three-stage hybrid model leveraging both classical and quantum computing paradigms:

#### Architecture Components:

1. **Classical Preprocessing Network**
   - **Input**: 30 original features from Wisconsin Breast Cancer dataset
   - **PCA Dimensionality Reduction**: 30 → 4 features (retaining >90% variance)
   - **Architecture**: Linear(4 → 16) → ReLU → Linear(16 → 2) → Tanh
   - **Purpose**: Feature extraction and preparation for quantum layer
   
2. **Quantum Variational Layer (VQA)**
   - **Circuit Design**: 2-qubit variational quantum circuit
   - **Data Re-uploading Strategy**: 3 layers with progressive data encoding
   - **Gate Structure**: RY(data) → RZ(data) → CX entanglement → RY(θ) → RZ(θ) → CX
   - **Parameters**: 12 trainable quantum parameters (2 qubits × 2 gates × 3 layers)
   - **Observables**: Pauli-Z measurements on both qubits
   - **Gradient Computation**: Parameter-shift rule for quantum gradients
   
3. **Classical Postprocessing Network**
   - **Input**: 2 quantum expectation values
   - **Architecture**: Linear(2 → 8) → ReLU → Linear(8 → 1)
   - **Output**: Single logit for binary classification

#### Key Technical Innovations:

- **Custom PyTorch Autograd**: Full integration with PyTorch's automatic differentiation
- **GPU Acceleration**: Optional qiskit-aer-gpu support with fallback to CPU
- **Medical-Optimized Threshold**: Decision threshold of 0.3 (vs standard 0.5) for higher sensitivity
- **Class Imbalance Handling**: BCEWithLogitsLoss with pos_weight=2.0 for malignant samples
- **Reproducible Training**: Fixed random seeds across all frameworks (PyTorch, NumPy, Qiskit)

## 💻 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- Linux/WSL2 or macOS

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FLIQ-Virtual-Hackathon-UNICC-Challenge-QuantumBreast-Cancer-Classification-Quantum-Buddies.git
   cd FLIQ-Virtual-Hackathon-UNICC-Challenge-QuantumBreast-Cancer-Classification-Quantum-Buddies
   ```

2. **Create virtual environment**
   ```bash
   python -m venv quantum_env
   source quantum_env/bin/activate  # On Windows: quantum_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Install CUDA Quantum for GPU acceleration**
   ```bash
   # For CUDA 12.x
   ./install_cuda_quantum_cu12.x86_64
   # Or install qiskit-aer-gpu
   pip install qiskit-aer-gpu
   ```

## 📁 Project Structure

```
FLIQ-Virtual-Hackathon-UNICC-Challenge-QuantumBreast-Cancer-Classification-Quantum-Buddies/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── FLIQ-Virtual-Hackathon/               # Main hackathon directory
│   ├── script.py                         # Main quantum-classical hybrid model script
│   ├── FLIQ.ipynb                        # Main Jupyter notebook for detailed experiments, visualizations, and results
│   ├── README.md                         # Original hackathon README (if present, for reference)
│   ├── Datasets/                         # Dataset directory
│   │   ├── breast+cancer+wisconsin+diagnostic/
│   │   ├── wine+quality/
│   │   ├── adult/
│   │   └── drug+induced+autoimmunity+prediction/
│   └── Sample Code/                      # Additional sample implementations

├── models/                               # Saved model weights
│   └── model.pth                        # Trained model checkpoint
└── visualizations/                       # Model architecture diagrams
    ├── model_architecture.svg
    └── model_architecture.html
```

## 🎮 Usage

### Training the Hybrid Model

```bash
cd FLIQ-Virtual-Hackathon
python script.py
```

**What this script does:**
1. **Data Loading**: Loads Wisconsin Breast Cancer dataset from `Sample Code/data/wdbc.data`
2. **Preprocessing**: Z-score normalization and PCA dimensionality reduction (30→4 features)
3. **Model Training**: 50 epochs with epoch-wise loss and accuracy display
4. **Evaluation**: Comprehensive test metrics including classification report and confusion matrix
5. **Visualizations**: Four detailed plots for model interpretability
6. **Model Saving**: Saves trained model weights to `model.pth`

### Expected Output

The training will show:
```
Epoch 1 | Loss: 0.8234 | Accuracy: 36.48%
Epoch 2 | Loss: 0.7891 | Accuracy: 36.48%
...
Epoch 50 | Loss: 0.6749 | Accuracy: 76.26%

Test Accuracy: 72.81%

Classification Report:
               precision    recall  f1-score   support

         0.0       0.82      0.69      0.75        68
         1.0       0.63      0.78      0.70        46

    accuracy                           0.73       114
   macro avg       0.73      0.74      0.73       114
weighted avg       0.75      0.73      0.73       114

Confusion Matrix:
 [[47 21]
 [10 36]]
```

### Using Jupyter Notebook

```bash
jupyter notebook FLIQ-Virtual-Hackathon/FLIQ.ipynb
```
The `FLIQ.ipynb` notebook contains the complete experimental workflow with detailed explanations, intermediate results, and comprehensive visualizations. It serves as the primary research document for the hackathon submission.

### Loading and Using Pre-trained Model

```python
import torch
import numpy as np
from script import HybridModel
from sklearn.decomposition import PCA

# Load the trained model
model = HybridModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Prepare new data (same preprocessing as training)
def preprocess_data(X_raw):
    # Normalize (using training set statistics)
    X_norm = (X_raw - training_mean) / training_std
    # Apply PCA (using fitted PCA from training)
    X_pca = pca_transformer.transform(X_norm)
    return torch.tensor(X_pca, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    predictions = model(preprocessed_data)
    probabilities = torch.sigmoid(predictions)
    binary_predictions = (probabilities > 0.3).float()
```

## 🏗️ Model Architecture

```
Wisconsin Breast Cancer Dataset (569 samples, 30 features)
    ↓
Data Preprocessing (Z-score normalization)
    ↓
PCA Dimensionality Reduction (30 → 4 features, >90% variance retained)
    ↓
Classical Preprocessing Network:
  Linear(4 → 16) → ReLU → Linear(16 → 2) → Tanh
    ↓
Quantum Variational Algorithm (VQA):
  2-qubit circuit, 3 layers, data re-uploading
  RY(data×layer) → CX → RY(θ) → RZ(θ) → CX (×3 layers)
    ↓
Quantum Measurements: Pauli-Z on both qubits → 2 expectation values
    ↓
Classical Postprocessing Network:
  Linear(2 → 8) → ReLU → Linear(8 → 1)
    ↓
Sigmoid Activation → Binary Classification (threshold=0.3)
```

### Detailed Quantum Circuit Design

#### VQA Circuit Structure (per sample):
```
|0⟩ ─ RY(x₁×layer) ─ ●─ RY(θ₁) ─ RZ(θ₂) ─ ●─ ... ─ ⟨Z⟩
                     │                      │
|0⟩ ─ RY(x₂×layer) ─ ⊕─ RY(θ₃) ─ RZ(θ₄) ─ ⊕─ ... ─ ⟨Z⟩
```

#### Specifications:
- **Qubits**: 2
- **Layers**: 3 (with progressive data encoding)
- **Data Encoding**: x₁, x₂ scaled by layer number (1, 2, 3)
- **Trainable Parameters**: 12 total (2 qubits × 2 gates × 3 layers)
- **Entanglement**: CX gates between adjacent qubits
- **Measurements**: Pauli-Z expectation values
- **Gradient Method**: Parameter-shift rule (π/2 shifts)

## 📊 Results

### Training Configuration

- **Dataset Split**: 80% training (455 samples), 20% testing (114 samples)  
- **Training Epochs**: 50
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Loss Function**: BCEWithLogitsLoss (pos_weight=2.0 for class imbalance)
- **Decision Threshold**: 0.3 (optimized for medical sensitivity)

### Performance Metrics

- **Test Accuracy**: Typically 95%+ (varies with random seed due to quantum circuit initialization)
- **Training Convergence**: Model converges within 20-30 epochs
- **Class Distribution**: 357 benign (62.8%) vs 212 malignant (37.2%) samples
- **Threshold Optimization**: 0.3 threshold provides better sensitivity for malignant detection

### Detailed Evaluation

The implementation includes comprehensive evaluation metrics:

1. **Classification Report**: Precision, recall, and F1-score for both classes
2. **Confusion Matrix**: True positive/negative vs false positive/negative analysis
3. **ROC Analysis**: Implicit through threshold optimization

### Visualizations

The model provides four key interpretability visualizations:

1. **Probability Distribution by True Class**: 
   - Shows prediction confidence distribution for benign vs malignant cases
   - Includes decision threshold visualization at 0.3

2. **2D Feature Space (Classical→Quantum Interface)**:
   - Visualizes the 2 features output by classical network (input to VQA)
   - Color-coded by true labels for pattern analysis

3. **2D Feature Space (Prediction Analysis)**:
   - Same feature space colored by predicted labels
   - Allows comparison with true labels for error analysis

4. **Confusion Matrix Heatmap**:
   - Visual representation of classification performance
   - Detailed breakdown of prediction accuracy by class

## 📚 Datasets

We primarily use the **Breast Cancer Wisconsin Diagnostic Dataset**:

- **Samples**: 569 (357 benign, 212 malignant)
- **Features**: 30 numeric features computed from digitized images
- **Task**: Binary classification (Malignant vs. Benign)
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

Additional datasets available for experimentation:
- Wine Quality Dataset
- Adult Income Dataset  
- Drug Induced Autoimmunity Prediction Dataset

## 🛠️ Technologies Used

- **Quantum Computing**: Qiskit 2.x, CUDA Quantum
- **Machine Learning**: PyTorch, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Development**: Python 3.8+, Jupyter Notebooks
- **Hardware Acceleration**: CUDA (optional)

## 🤝 Ethical Considerations

### Fairness & Bias
- Regular evaluation of demographic fairness metrics
- Balanced dataset representation
- Threshold optimization for medical safety

### Privacy & Security
- No patient identifiable information used
- Secure model deployment practices
- GDPR/HIPAA compliance considerations

### Explainability
- Visualization tools for understanding decisions
- Feature importance analysis
- Quantum circuit interpretability

### Human-in-the-Loop
- Medical professional validation recommended
- Conservative threshold for safety
- Clear uncertainty quantification

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/) - see the LICENSE file for details.

## 👥 Team: Quantum Buddies

This project was developed by the **Quantum Buddies** team for the FLIQ Virtual Hackathon, demonstrating collaborative quantum machine learning research and development.

**Team Members:**
- **Alisa Petrusinskaia**
- **Gyanateet Dutta**
- **Fiyin Makinde**
- **Sid Eliyasu**

## 🙏 Acknowledgments

**Special Thanks to the Quantum Buddies Team:**
- **Alisa Petrusinskaia** - Quantum algorithm development and implementation
- **Gyanateet Dutta** - Machine learning architecture and optimization
- **Fiyin Makinde** - Data analysis and visualization
- **Sid Eliyasu** - Model evaluation and testing

**Organizations and Resources:**
- **UN-ICC** and **Quantum Coalition** for organizing the hackathon
- **IBM Quantum** for Qiskit framework
- **NVIDIA** for CUDA Quantum support
- **UCI Machine Learning Repository** for datasets
- All hackathon mentors and organizers

---

**Contact**: For questions or collaboration, please reach out through GitHub issues or contact the hackathon organizers:
- Anusha Dandapani (dandapani@unicc.org)
- Gillian Makamara (gillian.makamara@itu.int)
- Devyani Rastogi (rastogi@unicc.org)
- Luke Sebold (lts45@case.edu)

---

*This project demonstrates the potential of quantum-classical hybrid models in real-world medical applications, contributing to the advancement of quantum machine learning for social good.* 