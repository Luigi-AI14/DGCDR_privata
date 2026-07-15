import argparse
import os
import glob
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from recbole_cdr.quick_start.quick_start import load_data_and_model

# ==========================================
# CHANGE TO THE DESIRED CHECKPOINT PATH
# ==========================================
DEFAULT_MODEL_PATH = "saved/DGCDR-Jul-14-2026_12-22-57.pth"

def get_latest_checkpoint(checkpoint_dir='saved'):
    pth_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not pth_files:
        return None
    # Sort by modification time (most recent first)
    pth_files.sort(key=os.path.getmtime, reverse=True)
    return pth_files[0]

def compute_distance_matrix(X):
    # X: [N, D]
    dot_product = torch.mm(X, X.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0)
    return torch.sqrt(distances)

def double_center(D):
    # D: [N, N]
    N = D.size(0)
    row_means = torch.mean(D, dim=1, keepdim=True)
    col_means = torch.mean(D, dim=0, keepdim=True)
    grand_mean = torch.mean(D)
    return D - row_means - col_means + grand_mean

def distance_correlation(X, Y):
    """
    Computes the distance correlation (dCor) between X and Y.
    dCor = 0 if and only if X and Y are independent.
    """
    N = X.size(0)
    if N <= 1:
        return 0.0
    
    # 1. Distance matrices
    D_X = compute_distance_matrix(X)
    D_Y = compute_distance_matrix(Y)
    
    # 2. Double centering
    A = double_center(D_X)
    B = double_center(D_Y)
    
    # 3. Covariance and variances
    dcov2 = torch.sum(A * B) / (N * N)
    dvarX2 = torch.sum(A * A) / (N * N)
    dvarY2 = torch.sum(B * B) / (N * N)
    
    if dvarX2 <= 0.0 or dvarY2 <= 0.0:
        return 0.0
    
    dcor = torch.sqrt(dcov2 / torch.sqrt(dvarX2 * dvarY2))
    return dcor.item()

def evaluate_embeddings(model_path, output_dir='disentanglement_eval'):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data and model checkpoint from {model_path}...")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_path)
    model.eval()
    
    print("\nExtracting embeddings via model.forward()...")
    with torch.no_grad():
        user_disentangled_list, item_disentangled_list, _, _, _, _ = model.forward()
    
    if not user_disentangled_list:
        raise ValueError("Error: The loaded model does not have preference disentanglement enabled.")
    
    # Extract user common and specific features for overlapping users
    source_c = user_disentangled_list[0].cpu()
    target_c = user_disentangled_list[1].cpu()
    source_s = user_disentangled_list[2].cpu()
    target_s = user_disentangled_list[3].cpu()
    
    print(f"Extracted features for {source_c.size(0)} overlapping users.")
    print(f"Embedding size: {source_c.size(1)}")
    
    # --- Metric 1: Average Absolute Cosine Similarity ---
    source_cos = torch.nn.functional.cosine_similarity(source_c, source_s, dim=1).abs().mean().item()
    target_cos = torch.nn.functional.cosine_similarity(target_c, target_s, dim=1).abs().mean().item()
    
    # --- Metric 2: Distance Correlation (dCor) ---
    source_dcor = distance_correlation(source_c, source_s)
    target_dcor = distance_correlation(target_c, target_s)
    
    # --- Metric 3: Probing Datasets Preparation ---
    X_common = torch.cat([source_c, target_c], dim=0).numpy()
    X_specific = torch.cat([source_s, target_s], dim=0).numpy()
    y = np.array([0] * len(source_c) + [1] * len(target_c))
    
    # Train-test split with stratification
    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(
        X_common, y, test_size=0.3, random_state=42, stratify=y
    )
    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
        X_specific, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # --- Metric 4: Linear Probe (Logistic Regression) ---
    # Common features linear probe
    clf_c_linear = LogisticRegression(max_iter=1000)
    clf_c_linear.fit(X_c_train, y_c_train)
    acc_c_linear = accuracy_score(y_c_test, clf_c_linear.predict(X_c_test))
    
    # Specific features linear probe
    clf_s_linear = LogisticRegression(max_iter=1000)
    clf_s_linear.fit(X_s_train, y_s_train)
    acc_s_linear = accuracy_score(y_s_test, clf_s_linear.predict(X_s_test))
    
    # --- Metric 5: Non-Linear Probe (Multi-Layer Perceptron) ---
    # A 2-layer MLP (100 -> 50 -> ReLU) to capture non-linear domain leakage
    clf_c_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)
    clf_c_mlp.fit(X_c_train, y_c_train)
    acc_c_mlp = accuracy_score(y_c_test, clf_c_mlp.predict(X_c_test))
    
    clf_s_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)
    clf_s_mlp.fit(X_s_train, y_s_train)
    acc_s_mlp = accuracy_score(y_s_test, clf_s_mlp.predict(X_s_test))

    # Calculate Disentanglement Scores
    dis_score_linear = acc_s_linear - acc_c_linear
    dis_score_mlp = acc_s_mlp - acc_c_mlp

    # Save metrics to TXT and CSV with unique name
    import csv
    from datetime import datetime
    
    model_name = os.path.basename(model_path).replace('.pth', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    readable_timestamp = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
    
    # Generate Text Report with unique name
    report = f"""
======================================================================
                     DISENTANGLEMENT METRICS REPORT
======================================================================
Model Checkpoint    : {model_name}
Timestamp           : {readable_timestamp}

1. FEATURE INDEPENDENCE METRICS (e^c vs e^s)
----------------------------------------------------------------------
Cosine Similarity   | Source: {source_cos:.4f} | Target: {target_cos:.4f}  (Ideal: ~0.00)
Distance Corr (dCor)| Source: {source_dcor:.4f} | Target: {target_dcor:.4f}  (Ideal: ~0.00)

2. ADVERSARIAL DOMAIN PROBING ACCURACIES
----------------------------------------------------------------------
LINEAR PROBE (Logistic Regression):
  - Common Features (e^c) Accuracy   : {acc_c_linear * 100:.2f}%          (Ideal: ~50.00%)
  - Specific Features (e^s) Accuracy : {acc_s_linear * 100:.2f}%          (Ideal: >85.00%)
  - Disentanglement Score            : {dis_score_linear * 100:.2f}%          (Ideal: >35.00%)

NON-LINEAR PROBE (2-Layer MLP):
  - Common Features (e^c) Accuracy   : {acc_c_mlp * 100:.2f}%          (Ideal: ~50.00%)
  - Specific Features (e^s) Accuracy : {acc_s_mlp * 100:.2f}%          (Ideal: >85.00%)
  - Disentanglement Score            : {dis_score_mlp * 100:.2f}%          (Ideal: >35.00%)

======================================================================
                           GUIDE & LEGEND
======================================================================
* Cosine Similarity & Distance Correlation (dCor) measure vector 
  sharing between common and specific spaces. Lower is better (~0.0).
* Domain Probing Accuracy on Shared Features (e^c) measures domain
  invariance. Lower is better (ideal: ~50.00% / random chance).
* Domain Probing Accuracy on Specific Features (e^s) measures domain
  retention. Higher is better (ideal: >85.00%).
* Disentanglement Score = Accuracy(e^s) - Accuracy(e^c).
  Higher is better (ideal: >35.00%).
======================================================================
"""
    
    txt_path = os.path.join(output_dir, f'report_{model_name}_{timestamp}.txt')
    with open(txt_path, 'w') as f:
        f.write(report)
        
    # Append to central history CSV (with compatibility checks for headers)
    csv_path = os.path.join(output_dir, 'disentanglement_history.csv')
    
    write_header = True
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                first_line = f.readline().strip()
                if 'DisentanglementScore_Linear' in first_line:
                    write_header = False
        except:
            write_header = True
            
    open_mode = 'w' if write_header else 'a'
    
    with open(csv_path, open_mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'Timestamp', 'Model_Checkpoint', 'CosSim_Src', 'CosSim_Tgt',
                'dCor_Src', 'dCor_Tgt', 
                'LinearProbe_Common', 'LinearProbe_Specific', 'DisentanglementScore_Linear',
                'MLPProbe_Common', 'MLPProbe_Specific', 'DisentanglementScore_MLP'
            ])
        writer.writerow([
            timestamp, model_name,
            f"{source_cos:.4f}", f"{target_cos:.4f}",
            f"{source_dcor:.4f}", f"{target_dcor:.4f}",
            f"{acc_c_linear:.4f}", f"{acc_s_linear:.4f}", f"{dis_score_linear:.4f}",
            f"{acc_c_mlp:.4f}", f"{acc_s_mlp:.4f}", f"{dis_score_mlp:.4f}"
        ])

    # --- Print Results ---
    print(report)
    print(f"Report saved to TXT:     {txt_path}")
    print(f"Appended to history CSV: {csv_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Disentanglement Quality")
    parser.add_argument('--model_path', '-m', type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to checkpoint .pth file (defaults to DEFAULT_MODEL_PATH configured in the script)")
    parser.add_argument('--output_dir', '-o', type=str, default='disentanglement_eval', help="Output directory")
    args = parser.parse_args()
    
    model_path = args.model_path
    if not model_path or not os.path.exists(model_path):
        latest = get_latest_checkpoint('saved')
        if latest:
            print(f"Configured path '{model_path}' not found or empty. Falling back to latest checkpoint: {latest}")
            model_path = latest
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{model_path}' and no checkpoints found in 'saved/'.")
            
    evaluate_embeddings(model_path, args.output_dir)
