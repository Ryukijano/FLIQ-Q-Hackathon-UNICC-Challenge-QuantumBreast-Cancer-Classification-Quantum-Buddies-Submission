import os, multiprocessing
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt
import seaborn as sns
# Attempt to import Aer Estimator for GPU, fallback to CPU StatevectorEstimator
try:
    from qiskit_aer.primitives import Estimator as AerEstimator
    # Check if GPU is available for Aer
    import qiskit_aer.library.set_instructions as aer_instr
    _HAS_AER_GPU = True
    print("qiskit-aer-gpu found. Will attempt to use GPU for Qiskit simulations.")
except ImportError:
    _HAS_AER_GPU = False
    print("qiskit-aer-gpu not found. Qiskit simulations will run on CPU.")

# -------------------------------------------------------------
#   Enable multi-core CPU utilisation (PyTorch + Qiskit)
# -------------------------------------------------------------
_n_cores = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(_n_cores)     # NumPy / OpenMP
os.environ["MKL_NUM_THREADS"] = str(_n_cores)     # Intel MKL
# Optional: limit Torch to the same number to avoid oversubscription
torch.set_num_threads(_n_cores)

# -------------------------------------------------------------
#    Set Random Seeds for Reproducibility
# -------------------------------------------------------------
SEED = 42 # Or any integer you like
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED) # for multi-GPU.
# torch.backends.cudnn.deterministic = True # Optional: can impact performance
# torch.backends.cudnn.benchmark = False    # Optional: can impact performance
# -------------------------------------------------------------

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Prep Training Data (Breast Cancer Wisconsin)

#Load the data file
df = pd.read_csv("/home/ryukijano/cuquantum/FLIQ-Virtual-Hackathon/Sample Code/data/wdbc.data", header=None)

#Assign column names
columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df.columns = columns

#Drop the ID column
df = df.drop(columns=['id'])

#Encode diagnosis: M = 1, B = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

#Convert to numpy arrays
X = df.drop(columns=['diagnosis']).values.astype(np.float32)
Y = df['diagnosis'].values.astype(np.float32).reshape(-1, 1)

#Normalize features manually (z-score)
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# --- Feature reduction: keep first 4 principal components (retain >90% var) ---
X_pca = PCA(n_components=4).fit_transform(X)

#Convert to torch tensors (use PCA-reduced features)
X_tensor = torch.tensor(X_pca, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

#Manual train/test split (80/20)
num_samples = X_tensor.shape[0]
indices = torch.randperm(num_samples)

split_idx = int(num_samples * 0.8)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = X_tensor[train_indices].to(device)
Y_train = Y_tensor[train_indices].to(device)
X_test = X_tensor[test_indices].to(device)
Y_test = Y_tensor[test_indices].to(device)

# ----- 2-qubit VQA (data-reuploading, 3 layers, 12 params) -----
n_qubits = 2
param_count = 12  # 2 qubits × 2 params (Ry,Rz) × 3 layers
params = ParameterVector('theta', length=param_count)

# Helper to add an entangling ring
_def_entangle = lambda qc: qc.cx(0, 1)

def create_vqa_circuit(input_data, weights):
    """2-qubit, 3-layer data-reuploading circuit."""
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(3):
        # encode data
        qc.ry(float(input_data[0]) * (layer+1), 0)
        qc.ry(float(input_data[1]) * (layer+1), 1)
        _def_entangle(qc)
        # trainable block
        for q in range(n_qubits):
            qc.ry(float(weights[idx]), q); idx += 1
            qc.rz(float(weights[idx]), q); idx += 1
        _def_entangle(qc)
    return qc

# Use AerEstimator if available and GPU is supported, otherwise fallback
if _HAS_AER_GPU:
    try:
        # Attempt to initialize with GPU, this might still fail if CUDA not configured properly for Aer
        estimator = AerEstimator(device="GPU", precision="single")
        # Verify GPU execution is possible by trying to instantiate an AerSimulator for GPU
        from qiskit_aer import AerSimulator
        AerSimulator(method='statevector', device='GPU') # This will raise an error if GPU is not usable by Aer
        print("Successfully initialized AerEstimator on GPU.")
        _EST_OPTS = {} # AerEstimator handles its own parallelism on GPU
    except Exception as e:
        print(f"Failed to initialize AerEstimator on GPU: {e}. Falling back to CPU StatevectorEstimator.")
        estimator = StatevectorEstimator() # Fallback to CPU
else:
    estimator = StatevectorEstimator() # Fallback to CPU

# Observables: Z expectation on each qubit
observables = [SparsePauliOp("ZI"), SparsePauliOp("IZ")]

#PyTorch Custom Autograd Function For VQA Layer
class VQALayerFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, weights):
        # input_tensor shape (4,), weights shape (12,)
        input_vals  = input_tensor.cpu().detach().numpy()
        weight_vals = weights.cpu().detach().numpy()
        ctx.save_for_backward(input_tensor, weights)

        qc = create_vqa_circuit(input_vals, weight_vals)
        circuits = [(qc, obs) for obs in observables]
        res = estimator.run(circuits, options=_EST_OPTS).result()
        expvals = np.array([r.data.evs for r in res])  # shape (4,)
        return torch.tensor(expvals, dtype=torch.float32).to(input_tensor.device)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weights = ctx.saved_tensors
        x      = input_tensor.cpu().detach().numpy()
        theta  = weights.cpu().detach().numpy()
        shift  = np.pi/2
        # gradients wrt weights
        grad_w = []
        for k in range(len(theta)):
            theta_p, theta_m = theta.copy(), theta.copy()
            theta_p[k] += shift; theta_m[k] -= shift
            qc_p = create_vqa_circuit(x, theta_p)
            qc_m = create_vqa_circuit(x, theta_m)
            res_p = estimator.run([(qc_p, obs) for obs in observables], options=_EST_OPTS).result()
            res_m = estimator.run([(qc_m, obs) for obs in observables], options=_EST_OPTS).result()
            grad_w.append(0.5 * (np.array([r.data.evs for r in res_p]) -
                                  np.array([r.data.evs for r in res_m])))
        grad_w = torch.tensor(grad_w, dtype=torch.float32).to(grad_output.device)  # (P,2)
        weight_grad = torch.mv(grad_w, grad_output.view(-1))  # (P,)

        # gradients wrt inputs (allow classical NN to train)
        grad_x = []
        for j in range(len(x)):
            x_p, x_m = x.copy(), x.copy()
            x_p[j] += shift; x_m[j] -= shift
            qc_p = create_vqa_circuit(x_p, theta)
            qc_m = create_vqa_circuit(x_m, theta)
            res_p = estimator.run([(qc_p, obs) for obs in observables], options=_EST_OPTS).result()
            res_m = estimator.run([(qc_m, obs) for obs in observables], options=_EST_OPTS).result()
            grad_x.append(0.5 * (np.array([r.data.evs for r in res_p]) -
                                  np.array([r.data.evs for r in res_m])))
        grad_x = torch.tensor(grad_x, dtype=torch.float32).to(grad_output.device)  # (2,2)
        input_grad = torch.mv(grad_x, grad_output.view(-1))  # (features,)

        return input_grad, weight_grad

#Quantum Layer as PyTorch Module
class VQALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(param_count, device=device))

    def forward(self, x):
        # x shape (batch,2)
        outputs = [VQALayerFunction.apply(x[i], self.weights) for i in range(x.size(0))]
        return torch.stack(outputs)  # (batch,2)

#Full Hybrid Model
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical = nn.Sequential(
            nn.Linear(X_tensor.shape[1], 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # output matches qubit count
            nn.Tanh()
        )
        self.quantum = VQALayer()
        self.output = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.classical(x)
        x = self.quantum(x)
        x = self.output(x)
        return x

model = HybridModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))

#Training Loop
for epoch in range(50):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, Y_train)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        acc = ((preds > 0.5).float() == Y_train).float().mean()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Accuracy: {acc.item()*100:.2f}%")

#Implement Testing Loop:
with torch.no_grad():
    test_preds_logits = model(X_test)
    test_probs = torch.sigmoid(test_preds_logits)
    test_labels_on_device = (test_probs > 0.3).float()

    test_acc = (test_labels_on_device == Y_test).float().mean().item()
    print("\nTest Accuracy: {:.2f}%".format(test_acc*100))

    Y_test_np = Y_test.cpu().numpy().flatten()
    test_labels_np = test_labels_on_device.cpu().numpy().flatten()
    test_probs_np = test_probs.cpu().numpy().flatten()

    print("\nClassification Report:\n", classification_report(Y_test_np, test_labels_np))
    cm = confusion_matrix(Y_test_np, test_labels_np)
    print("Confusion Matrix:\n", cm)

    # ----- Start Visualizations -----

    # 1. Distribution of Predicted Probabilities
    plt.figure(figsize=(10, 6))
    sns.histplot(test_probs_np[Y_test_np == 0], color="skyblue", label="Actual Benign (0)", kde=True, stat="density", common_norm=False)
    sns.histplot(test_probs_np[Y_test_np == 1], color="salmon", label="Actual Malignant (1)", kde=True, stat="density", common_norm=False)
    plt.axvline(0.3, color='black', linestyle='--', label="Decision Threshold (0.3)")
    plt.title('Distribution of Predicted Probabilities by True Class')
    plt.xlabel('Predicted Probability (Malignant)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # 2. Features input to VQA, colored by True Labels
    # Get the 2 features that are input to the VQA layer
    features_for_vqa = model.classical(X_test).cpu().numpy()

    plt.figure(figsize=(10, 6))
    scatter_true = plt.scatter(features_for_vqa[:, 0], features_for_vqa[:, 1], c=Y_test_np, cmap='coolwarm', alpha=0.7)
    handles_true, _ = scatter_true.legend_elements(prop="colors", alpha=0.6)
    legend_labels_true = ['Benign (0)', 'Malignant (1)'] # Assuming Y_test_np contains 0 and 1
    plt.legend(handles_true, legend_labels_true, title="True Labels")
    plt.title('2D Features (Output of Classical Head, Input to VQA) - Colored by True Labels')
    plt.xlabel('Classical Feature 1 (for VQA)')
    plt.ylabel('Classical Feature 2 (for VQA)')
    plt.grid(True)
    plt.show()

    # 3. Features input to VQA, colored by Predicted Labels
    plt.figure(figsize=(10, 6))
    scatter_pred = plt.scatter(features_for_vqa[:, 0], features_for_vqa[:, 1], c=test_labels_np, cmap='coolwarm', alpha=0.7)
    handles_pred, _ = scatter_pred.legend_elements(prop="colors", alpha=0.6)
    legend_labels_pred = ['Benign (0)', 'Malignant (1)'] # Assuming test_labels_np contains 0 and 1
    plt.legend(handles_pred, legend_labels_pred, title="Predicted Labels")
    plt.title('2D Features (Output of Classical Head, Input to VQA) - Colored by Predicted Labels')
    plt.xlabel('Classical Feature 1 (for VQA)')
    plt.ylabel('Classical Feature 2 (for VQA)')
    plt.grid(True)
    plt.show()

    # 4. Graphical Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign (0)', 'Malignant (1)'], yticklabels=['Benign (0)', 'Malignant (1)'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # ----- End Visualizations -----

# Save the trained model
model_save_path = "model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\nTrained model saved to: {model_save_path}")

# Example of loading the model (optional, for verification)
# loaded_model = HybridModel() # Create a new instance of the model
# loaded_model.load_state_dict(torch.load(model_save_path))
# loaded_model.to(device) # Don't forget to move it to the correct device
# loaded_model.eval() # Set it to evaluation mode if you are doing inference
# print("\nModel loaded successfully and moved to evaluation mode.")
