# FLIQ-Virtual-Hackathon UNICC-Challenge QuantumBreast Cancer Classification

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.x-blueviolet.svg)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org/)

## 🏆 Project Overview

This repository contains our solution for the **Quantum Coalition Future Leaders in Quantum (QC-FLIQ) Virtual Hackathon** organized by UN-ICC. Our project implements a **Quantum-Classical Hybrid Neural Network** for breast cancer classification using Variational Quantum Algorithms (VQAs).

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

Our solution implements a three-stage hybrid model:

1. **Classical Preprocessing Layer**
   - Feature extraction and dimensionality reduction using PCA
   - Classical neural network for initial feature transformation
   
2. **Quantum Variational Layer**
   - 2-qubit variational quantum circuit
   - Data re-uploading strategy with 3 layers
   - Parameter sharing for efficient training
   
3. **Classical Postprocessing Layer**
   - Final classification head
   - Sigmoid activation for probability estimation

### Key Innovations

- **Adaptive Threshold**: Dynamic decision threshold (0.3) optimized for medical diagnosis
- **GPU Acceleration**: Optional CUDA Quantum support for faster training
- **Gradient Computation**: Custom PyTorch autograd function for quantum gradients
- **Feature Engineering**: PCA-based dimensionality reduction (30 → 4 features)

## 💻 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- Linux/WSL2 or macOS

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FLIQ-Virtual-Hackathon-UNICC-Challenge-QuantumBreast-Cancer-Classification.git
   cd FLIQ-Virtual-Hackathon-UNICC-Challenge-QuantumBreast-Cancer-Classification
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
FLIQ-Virtual-Hackathon-UNICC-Challenge-QuantumBreast-Cancer-Classification/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── FLIQ-Virtual-Hackathon/               # Main hackathon directory
│   ├── script.py                         # Main quantum-classical hybrid model
│   ├── FLIQ.ipynb                        # Jupyter notebook with experiments
│   ├── README.md                         # Original hackathon README
│   ├── Datasets/                         # Dataset directory
│   │   ├── breast+cancer+wisconsin+diagnostic/
│   │   ├── wine+quality/
│   │   ├── adult/
│   │   └── drug+induced+autoimmunity+prediction/
│   └── Sample Code/                      # Additional sample implementations
├── notebooks/                            # Additional experiments
│   ├── quantum_neural_networks.ipynb
│   ├── quantum_convolutional_neural_network.ipynb
│   ├── quantum_autoencoder.ipynb
│   ├── quantum_kernel_machine_learning.ipynb
│   ├── pytorch_qGAN_implemenation.ipynb
│   ├── torch_connector_hybrid_qnn.ipynb
│   └── training_qnn_real_dataset.ipynb
├── models/                               # Saved model weights
│   └── model.pth                        # Trained model checkpoint
└── visualizations/                       # Model architecture diagrams
    ├── model_architecture.svg
    └── model_architecture.html
```

## 🎮 Usage

### Training the Model

```bash
cd FLIQ-Virtual-Hackathon
python script.py
```

### Using Jupyter Notebooks

```bash
jupyter notebook FLIQ.ipynb
```

### Loading Pre-trained Model

```python
import torch
from script import HybridModel

# Load model
model = HybridModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(test_data)
```

## 🏗️ Model Architecture

```
Input (30 features)
    ↓
PCA (30 → 4 features)
    ↓
Classical NN (4 → 16 → 2)
    ↓
Quantum VQA Layer (2 qubits, 3 layers, 12 parameters)
    ↓
Classical NN (2 → 8 → 1)
    ↓
Output (Binary Classification)
```

### Quantum Circuit Design

- **Qubits**: 2
- **Layers**: 3 (with data re-uploading)
- **Gates**: RY, RZ rotations + CX entanglement
- **Parameters**: 12 trainable quantum parameters
- **Observables**: Pauli-Z measurements on each qubit

## 📊 Results

### Performance Metrics

- **Test Accuracy**: ~95%+ (varies with random seed)
- **Precision/Recall**: Balanced for both classes
- **F1-Score**: High performance on medical diagnosis task

### Visualizations

The model provides several interpretability visualizations:

1. **Probability Distribution**: Shows prediction confidence by class
2. **Feature Space**: 2D visualization of quantum layer inputs
3. **Confusion Matrix**: Detailed classification results
4. **Loss Curves**: Training convergence analysis

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

## 🙏 Acknowledgments

- **UN-ICC** and **Quantum Coalition** for organizing the hackathon
- **IBM Quantum** for Qiskit framework
- **NVIDIA** for CUDA Quantum support
- **UCI Machine Learning Repository** for datasets
- All contributors and mentors

---

**Contact**: For questions or collaboration, please reach out through GitHub issues or contact the hackathon organizers:
- Anusha Dandapani (dandapani@unicc.org)
- Gillian Makamara (gillian.makamara@itu.int)
- Devyani Rastogi (rastogi@unicc.org)
- Luke Sebold (lts45@case.edu)

---

*This project demonstrates the potential of quantum-classical hybrid models in real-world medical applications, contributing to the advancement of quantum machine learning for social good.* 