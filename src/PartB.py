import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import time
import argparse
import sys
from typing import Any, Dict, Iterable, List
from scipy import stats
from scipy.stats import mode, skew, kurtosis
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from joblib import Parallel, delayed, cpu_count
import psutil

# ==================================================================================
#                       IMPORTS DO MAIN ACTIVITY
# ==================================================================================
from mainActivity import (
    load_all_data, 
    extract_basic_features, 
    reliefF, 
    get_plot_mode, 
    check_plot_permission
)

# ==================================================================================
#                       ESTRUTURAS DE DADOS E CLASSES
# ==================================================================================

class DatasetSplit:
    """ Encapsula os conjuntos de treino, validação e teste. """
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.mu = None
        self.sigma = None

    def get_train(self): 
        return self.X_train, self.y_train
    
    def get_val(self): 
        return self.X_val, self.y_val
    
    def get_test(self): 
        return self.X_test, self.y_test
    
    def get_train_val_combined(self):
        """ Retorna Train + Val concatenados para o treino final """
        if len(self.X_train) == 0: 
            return self.X_train, self.y_train
        
        X_full = np.concatenate((self.X_train, self.X_val), axis=0)
        y_full = np.concatenate((self.y_train, self.y_val), axis=0)
        return X_full, y_full

class CustomKNN:
    """ Implementação manual otimizada do k-NN. """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Modelo não treinado.")

        n_test = X_test.shape[0]
        batch_size = (1024*2)
        predictions = []
        
        # Limitar K ao tamanho do treino
        k = min(self.n_neighbors, len(self.X_train))
        printed_debug = False
        # Processar em blocos para não estourar a RAM
        for i in range(0, n_test, batch_size):
            X_batch = X_test[i : i + batch_size]

            if not printed_debug:
                debug_memory(f"KNN Predict (Batch {X_batch.shape})")
                printed_debug = True
            
            dists = cdist(X_batch, self.X_train, metric='sqeuclidean')
            if k < len(self.X_train):
                k_indices = np.argpartition(dists, kth=k-1, axis=1)[:, :k]
            else:
                k_indices = np.argsort(dists, axis=1)[:, :k]
            
            k_labels = self.y_train[k_indices]
            mode_res = mode(k_labels, axis=1, keepdims=False)
    
            batch_preds = mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0]
            predictions.append(np.ravel(batch_preds))
            
        return np.concatenate(predictions)
        
    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        return accuracy_score(y_test, preds)

# ==================================================================================
#                                  SECÇÃO 0: UTILITÁRIOS
# ==================================================================================

def filter_activities(X, y, participants, max_activity=7):
    print(f"  > A filtrar atividades (Manter 1 a {max_activity})...")
    mask = (y >= 1) & (y <= max_activity)
    return X[mask], y[mask], participants[mask]

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:.2f}s"
    else:
        h = int(seconds // 3600)
        rem = seconds % 3600
        m = int(rem // 60)
        s = rem % 60
        return f"{h}h {m}m {s:.0f}s"

def print_timing_table(timings):
    print("\n" + "="*65)
    print(f"{'ETAPA DA PIPELINE':<40} | {'TEMPO':>20}")
    print("-" * 65)
    
    total_time = 0.0
    for task, t in timings.items():
        current_time = 0.0
        time_str = ""
        if isinstance(t, tuple): 
            total, avg = t
            time_str = f"{format_time(total)} (Avg: {format_time(avg)})"
            current_time = float(total)
        elif isinstance(t, (int, float)):
            time_str = format_time(t)
            current_time = float(t)
            
        total_time += current_time
        print(f"{task:<40} | {time_str:>20}")
        
    print("-" * 65)
    print(f"{'TEMPO TOTAL':<40} | {format_time(total_time):>20}")
    print("="*65 + "\n")

def build_linear_resampling_matrix(input_len=256, output_len=150):
    """
    Cria uma matriz de transformação (Input_Len x Output_Len) para interpolação linear.
    Permite substituir loops de np.interp por uma única multiplicação de matrizes.
    """
    matrix = np.zeros((input_len, output_len), dtype=np.float32)
    scale = (input_len - 1) / (output_len - 1)
    
    for i in range(output_len):
        v = i * scale
        idx = int(v)
        alpha = v - idx
        
        if idx >= input_len - 1:
            matrix[input_len - 1, i] = 1.0
        else:
            matrix[idx, i] = 1.0 - alpha
            matrix[idx + 1, i] = alpha
            
    return matrix

def debug_memory(tag=""):
    """ Imprime a memória RAM usada pelo processo atual """
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss # Resident Set Size
    mem_mb = mem_bytes / (1024 * 1024)
    print(f"  [DEBUG-MEM] PID {os.getpid()} | {tag}: {mem_mb:.2f} MB")

# ==================================================================================
#                          SECÇÃO 1: DATA AUGMENTATION (EX 1)
# ==================================================================================

def analyze_class_balance(y, plot_mode, title="Distribuição de Classes"):
    unique, counts = np.unique(y, return_counts=True)
    if plot_mode == 'custom' or plot_mode == 'all':
        print(f"\n--- {title} ---")
        for u, c in zip(unique, counts):
            print(f"  Atividade {int(u)}: {c} amostras")
    
    is_balanced = (np.std(counts) / np.mean(counts)) < 0.2
    print(f"  > O dataset é balanceado? {'Sim' if is_balanced else 'Não'}.")

    if check_plot_permission(plot_mode, "[1.1] Mostrar gráfico de barras do balanceamento?"):
        plt.figure(figsize=(10, 5))
        plt.bar(unique, counts, color='skyblue', edgecolor='black')
        plt.xlabel('Atividade'); plt.ylabel('Amostras'); plt.title(title)
        plt.show()

def smote_manual(X, k_new_samples, k_neighbors=5):
    n_samples = X.shape[0]
    if n_samples < 2: return None
    
    n_neighbors = min(k_neighbors, n_samples - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    synthetic_samples = []
    
    for _ in range(k_new_samples):
        idx_base = np.random.randint(0, n_samples)
        sample_base = X[idx_base]
        _, indices = nbrs.kneighbors(sample_base.reshape(1, -1))
        neighbor_indices = indices[0][1:]
        
        if len(neighbor_indices) > 0:
            idx_neighbor = np.random.choice(neighbor_indices)
            sample_neighbor = X[idx_neighbor] 
            diff = sample_neighbor - sample_base
            new_sample = sample_base + (np.random.random() * diff)
            synthetic_samples.append(new_sample)
            
    return np.array(synthetic_samples)

def visualize_smote_participant(X, y, participants, plot_mode, part_id=3, act_id=4, k_gen=3, feat_cols=(0, 2)):
    print(f"\n--- 1.3 SMOTE: Participante {part_id}, Atividade {act_id} ---")
    mask_p = participants == part_id
    X_p, y_p = X[mask_p], y[mask_p]
    X_target = X_p[y_p == act_id]
    
    if len(X_target) < 2: 
        print("  > Amostras insuficientes para SMOTE.")
        return

    X_syn = smote_manual(X_target, k_new_samples=k_gen, k_neighbors=3)
    if X_syn is not None:
        print(f"  > Geradas {len(X_syn)} amostras sintéticas.")

    if check_plot_permission(plot_mode, f"[1.3] Visualizar SMOTE (Features {feat_cols})?"):
        plt.figure(figsize=(10, 6))
        idx_x, idx_y = feat_cols
        for act in np.unique(y_p):
            mask_act = (y_p == act)
            plt.scatter(X_p[mask_act, idx_x], X_p[mask_act, idx_y], label=f'Act {int(act)}', alpha=0.6)
        
        if X_syn is not None:
            plt.scatter(X_syn[:, idx_x], X_syn[:, idx_y], 
                        c='red', marker='*', s=200, label='Sintético', edgecolors='black')
        
        plt.title(f"SMOTE (Part {part_id}) - Feat {idx_x} vs Feat {idx_y}")
        plt.xlabel(f"Feature Índice {idx_x}")
        plt.ylabel(f"Feature Índice {idx_y}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

# ==================================================================================
#                          SECÇÃO 2: EMBEDDINGS (EX 2)
# ==================================================================================

def load_harnet_model():
    print("  > [Embeddings] A carregar modelo HARNet5...")
    try:
        model: Any = torch.hub.load('OxWearables/ssl-wearables', 'harnet5', class_num=5, pretrained=True, trust_repo=True) # nosec
        feature_encoder = model.feature_extractor
        feature_encoder.eval()
        return feature_encoder
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None

def resample_segment(acc_xyz, original_fs=51.2, target_fs=30.0):
    t_in = np.arange(acc_xyz.shape[0]) / original_fs
    t_out = np.arange(0, 5, 1.0/target_fs)
    if len(t_out) > int(5 * target_fs): t_out = t_out[:int(5 * target_fs)]
    
    acc_res = np.zeros((len(t_out), 3), dtype=np.float32)
    for axis in range(3):
        acc_res[:, axis] = np.interp(t_out, t_in, acc_xyz[:, axis])
    return acc_res

def segment_raw_data_for_embeddings(all_data, fs=51.2):
    acc_raw = all_data[:, 1:4] 
    labels = all_data[:, 11]
    participants = all_data[:, -1]
    
    win_size = int(5 * fs) # 256
    step = int(win_size * 0.5) # 128
    X_wins = sliding_window_view(acc_raw, window_shape=win_size, axis=0)[::step]
    y_wins_view = sliding_window_view(labels, window_shape=win_size, axis=0)[::step]
    p_wins_view = sliding_window_view(participants, window_shape=win_size, axis=0)[::step]
    y_mins = np.min(y_wins_view, axis=1)
    y_maxs = np.max(y_wins_view, axis=1)
    consistent_mask = (y_mins == y_maxs)
    valid_act_mask = (y_mins >= 1) & (y_mins <= 7)
    
    final_mask = consistent_mask & valid_act_mask
    return (X_wins[final_mask].copy(), 
            y_mins[final_mask].copy(), 
            p_wins_view[final_mask, 0].copy())

def generate_embeddings_dataset(all_data, batch_size=512):
    debug_memory("Start Embeddings Gen")
    t_start = time.time()
    X_raw, y, p = segment_raw_data_for_embeddings(all_data)
    debug_memory("After Segmentation")
    X_transposed = X_raw.transpose(0, 2, 1)
    resample_mat = build_linear_resampling_matrix(input_len=256, output_len=150)
    X_res = np.dot(X_transposed, resample_mat)
    debug_memory("After Resampling")
    encoder = load_harnet_model()
    if encoder is None: return None, None, None
    debug_memory("Model Loaded")
    dataset_tensor = torch.from_numpy(X_res).float()
    embs = []
    
    with torch.no_grad():
        total_samples = dataset_tensor.shape[0]
        for i in range(0, total_samples, batch_size):
            batch = dataset_tensor[i : i + batch_size]
            out = encoder(batch)
            embs.append(out.numpy())
            
    X_emb = np.concatenate(embs, axis=0)
    
    elapsed = time.time() - t_start
    print(f"  > Dataset Embeddings Criado: {X_emb.shape} em {format_time(elapsed)}")
    debug_memory("End Embeddings Gen")
    return X_emb, y, p

# ==================================================================================
#                          SECÇÃO 3: SPLITTING E TRANSFORMAÇÃO (EX 3)
# ==================================================================================

def standardize_split(split: DatasetSplit):
    mu = np.mean(split.X_train, axis=0)
    sigma = np.std(split.X_train, axis=0) + 1e-12
    split.X_train = (split.X_train - mu) / sigma
    split.X_val = (split.X_val - mu) / sigma
    split.X_test = (split.X_test - mu) / sigma
    split.mu = mu; split.sigma = sigma
    return split

def split_within_subjects(X, y, participants, seed=42):
    unique_parts = np.unique(participants)
    X_tr_l, y_tr_l, X_v_l, y_v_l, X_ts_l, y_ts_l = [], [], [], [], [], []

    for part in unique_parts:
        mask = (participants == part)
        X_p, y_p = X[mask], y[mask]
        n_samples = len(X_p)
        if n_samples < 5: continue 
        
        # Split Cronológico: 60% Train, 20% Val, 20% Test
        idx_tr = int(n_samples * 0.6)
        idx_val = int(n_samples * 0.8)
        
        X_tr_l.append(X_p[:idx_tr])
        y_tr_l.append(y_p[:idx_tr])
        X_v_l.append(X_p[idx_tr:idx_val])
        y_v_l.append(y_p[idx_tr:idx_val])
        X_ts_l.append(X_p[idx_val:])
        y_ts_l.append(y_p[idx_val:])

    if not X_tr_l: return DatasetSplit(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    
    return DatasetSplit(
        np.concatenate(X_tr_l), np.concatenate(y_tr_l),
        np.concatenate(X_v_l), np.concatenate(y_v_l),
        np.concatenate(X_ts_l), np.concatenate(y_ts_l)
    )

def split_between_subjects(X, y, participants, seed=42):
    unique_parts = np.unique(participants)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_parts) 
    
    n_total = len(unique_parts)
    n_val = int(n_total * 0.2); n_test = int(n_total * 0.2); n_train = n_total - n_val - n_test
    if n_val == 0 and n_total >= 3: n_val = 1; n_test = 1; n_train = n_total - 2

    train_parts = unique_parts[:n_train]
    val_parts = unique_parts[n_train : n_train + n_val]
    test_parts = unique_parts[n_train + n_val :]

    mask_tr = np.isin(participants, train_parts)
    mask_v = np.isin(participants, val_parts)
    mask_ts = np.isin(participants, test_parts)

    return DatasetSplit(X[mask_tr], y[mask_tr], X[mask_v], y[mask_v], X[mask_ts], y[mask_ts])

def apply_pca_transformation(split_data: DatasetSplit, variance_threshold=0.90, verbose=True):
    if len(split_data.X_train) == 0: return split_data
    
    mu = np.mean(split_data.X_train, axis=0)
    sigma = np.std(split_data.X_train, axis=0) + 1e-12
    X_tr_norm = (split_data.X_train - mu) / sigma
    
    cov = np.cov(X_tr_norm, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = np.argsort(eig_vals)[::-1]
    
    cumulative = np.cumsum(eig_vals[idx] / np.sum(eig_vals))
    n_comp = np.argmax(cumulative >= variance_threshold) + 1
    proj_mat = eig_vecs[:, idx][:, :n_comp]
    
    def project(X): return np.dot((X - mu) / sigma, proj_mat)
    
    ds = DatasetSplit(project(split_data.X_train), split_data.y_train, 
                      project(split_data.X_val), split_data.y_val, 
                      project(split_data.X_test), split_data.y_test)
    ds.mu = mu; ds.sigma = sigma
    return ds

def apply_relieff_transformation(split_data: DatasetSplit, n_features=15, verbose=True):
    if len(split_data.X_train) == 0: return split_data
    
    X_calc = split_data.X_train
    y_calc = split_data.y_train
    
    limit = 2000 
    
    if len(X_calc) > limit:
        # Usar seed fixa para consistência dentro da mesma run
        rng = np.random.default_rng(42) 
        indices = rng.choice(len(X_calc), limit, replace=False)
        X_calc = X_calc[indices]
        y_calc = y_calc[indices]
    
    # Calcular scores no subset (Rápido e leve)
    scores = reliefF(X_calc, y_calc)
    
    # Aplicar a seleção ao dataset COMPLETO
    top_indices = np.argsort(scores)[::-1][:n_features]
    
    new_split = DatasetSplit(
        split_data.X_train[:, top_indices], split_data.y_train, 
        split_data.X_val[:, top_indices], split_data.y_val, 
        split_data.X_test[:, top_indices], split_data.y_test
    )
    return standardize_split(new_split)

def generate_scenarios(split_data: DatasetSplit, verbose=True):
    all_std = standardize_split(DatasetSplit(
        split_data.X_train.copy(), split_data.y_train,
        split_data.X_val.copy(), split_data.y_val,
        split_data.X_test.copy(), split_data.y_test
    ))
    scenarios = {'all': all_std}
    scenarios['pca'] = apply_pca_transformation(split_data, variance_threshold=0.90, verbose=verbose)
    scenarios['relieff'] = apply_relieff_transformation(split_data, n_features=15, verbose=verbose)
    return scenarios

# ==================================================================================
#                          SECÇÃO 4: MODEL LEARNING (EX 4)
# ==================================================================================

def train_knn_model(X_train, y_train, n_neighbors=5):
    clf = CustomKNN(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    return clf

def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0) 
    }

def plot_confusion_matrix(y_true, y_pred, title, plot_mode):
    """ Plota a matriz de confusão se o utilizador permitir. """
    if not check_plot_permission(plot_mode, f"Mostrar Matriz de Confusão: {title}?"):
        return

    cm = confusion_matrix(y_true, y_pred)
    # Ajustar labels conforme as classes existentes (1 a 7 geralmente)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Criar figura explicitamente para controlar tamanho
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=False)
    
    plt.title(f"Confusion Matrix - {title}")
    plt.grid(False) # Desligar grid para ficar limpo
    plt.show()

# ==================================================================================
#                          SECÇÃO 5: EVALUATION (EX 5 - PARALELIZADO)
# ==================================================================================

def tune_best_k(split_data: DatasetSplit, k_range: Iterable[int] = range(1, 22, 2)):
    X_train, y_train = split_data.X_train, split_data.y_train
    X_val, y_val = split_data.X_val, split_data.y_val
    
    ks = list(k_range)
    n_train = len(X_train)
    ks = [k for k in ks if k <= n_train]
    if not ks: ks = [1]
    
    max_k = max(ks)
    batch_size = 1024 * 2
    n_val = len(X_val)
    correct_counts = {k: 0 for k in ks}
    printed_debug = False
    for i in range(0, n_val, batch_size):
        X_batch = X_val[i : i + batch_size]
        y_batch = y_val[i : i + batch_size]
        if not printed_debug:
            debug_memory(f"Tune Best K (Batch {X_batch.shape})")
            printed_debug = True
        
        dists = cdist(X_batch, X_train, metric='sqeuclidean')
        
        if max_k < n_train:
            # 1. Isolar os top max_k (não ordenados)
            part_indices = np.argpartition(dists, kth=max_k-1, axis=1)[:, :max_k]
            # 2. Ordenar apenas esses top max_k
            row_indices = np.arange(dists.shape[0])[:, None]
            top_dists = dists[row_indices, part_indices]
            sorted_order = np.argsort(top_dists, axis=1)
            sorted_indices = part_indices[row_indices, sorted_order]
        else:
            sorted_indices = np.argsort(dists, axis=1)[:, :max_k]
            
        neighbor_labels = y_train[sorted_indices]
        
        for k in ks:
            k_labels = neighbor_labels[:, :k]
            mode_res = mode(k_labels, axis=1, keepdims=False)
            preds = mode_res.mode if hasattr(mode_res, 'mode') else mode_res[0]
            preds = np.ravel(preds)
            correct_counts[k] += np.sum(preds == y_batch)
            
    best_k = ks[0]
    best_val_acc = -1.0
    
    for k in ks:
        acc = correct_counts[k] / n_val
        if acc > best_val_acc:
            best_val_acc = acc
            best_k = k
            
    X_full, y_full = split_data.get_train_val_combined()
    return train_knn_model(X_full, y_full, n_neighbors=best_k), best_k, best_val_acc

def _worker_single_run(i, X, y, p, strat, transforms):
    """ Função auxiliar executada em paralelo (1 Run) """
    # Evitar oversubscription
    debug_memory(f"Worker Start Run {i}")
    seed = 1000 + i
    if strat == 'within':
        split = split_within_subjects(X, y, p, seed=seed)
    else:
        split = split_between_subjects(X, y, p, seed=seed)
        
    if len(split.X_train) == 0: return None

    debug_memory(f"After Split Run {i}")
    scenarios = generate_scenarios(split, verbose=False)
    results = {}

    for trans in transforms:
        split_data = scenarios[trans]
        model, best_k, _ = tune_best_k(split_data, k_range=range(1, 22, 2))
        y_pred = model.predict(split_data.X_test)
        acc = accuracy_score(split_data.y_test, y_pred)
        results[trans] = (acc, best_k)
    debug_memory(f"Worker End Run {i} ({strat})")
    return results

def plot_performance_comparison(results_dict):
    plt.figure(figsize=(14, 8))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in np.linspace(0, 1, len(results_dict))]
    
    all_values = np.concatenate(list(results_dict.values()))
    min_x, max_x = np.min(all_values), np.max(all_values)
    size = 0.001
    bins = np.arange(min_x - (10*size), max_x + (10*size), size)
    sorted_keys = sorted(results_dict.keys())
    
    for i, key in enumerate(sorted_keys):
        data = results_dict[key]
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        hist_smooth = gaussian_filter1d(hist, sigma=-np.log(size))
        
        clean_label = key.replace('_', ' ') 
        ls = '-' if 'within' in key.lower() else '--'
        marker_shape = 'o' if 'Features' in key else 's'
        
        plt.plot(bin_centers, hist_smooth, label=clean_label, linestyle=ls, 
                 color=colors[i], alpha=0.8, linewidth=2)

    plt.title("Distribuição de Performance Estimada (Suavizada)")
    plt.xlabel("Accuracy")
    plt.ylabel("Densidade")
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def run_full_evaluation(datasets_raw, plot_mode='none', n_runs=10):
    print(f"\n" + "="*80)
    print(f"--- 5. EVALUATION: A executar {n_runs} iterações (PARALELO NON-BLOCKING) ---")
    print("="*80)
    
    results_distribution = {} 
    global_best_config = {'accuracy': -1.0, 'dataset': '', 'strategy': '', 'transform': '', 'k': 5}
    strategies = ['within', 'between']
    transforms = ['all', 'pca', 'relieff']

    # Intervalo de reporte (minimo 1 run ou 10%)
    milestones = [0.25, 0.50, 0.75, 1.0]
    next_milestone_idx = 0
    
    for ds_name, (X, y, p) in datasets_raw.items():
        if X is None: continue
        
        for strat in strategies:
            acc_lists = {t: [] for t in transforms}
            k_lists = {t: [] for t in transforms} 
            
            print(f"\n>> [INÍCIO] {ds_name} | {strat} | CPUs: {cpu_count()-1}")
            start_run = time.time()
            last_checkpoint = start_run
            next_milestone_idx = 0
        
            tasks = (delayed(_worker_single_run)(i, X, y, p, strat, transforms) for i in range(n_runs))
            results_generator = Parallel(n_jobs=-2, return_as="generator")(tasks)
            
            # Consumir resultados em tempo real (sem bloquear o batch)
            for i, res in enumerate(results_generator):
                if res is None: continue
                
                # Guardar dados
                for trans, (acc, k) in res.items():
                    acc_lists[trans].append(acc)
                    k_lists[trans].append(k)

                
                count = i + 1
                progress_ratio = count / n_runs
                if next_milestone_idx < len(milestones) and progress_ratio >= milestones[next_milestone_idx]:
                    current_time = time.time()
                    chunk_time = current_time - last_checkpoint
                    total_elapsed = current_time - start_run
                    percent = progress_ratio * 100
                    
                    print(f"    --> [PROG {percent:>3.0f}%] {count}/{n_runs} concluídas. \n"
                          f"\t(Desde último print: {format_time(chunk_time)} | Total: {format_time(total_elapsed)})")
                    
                    last_checkpoint = current_time
                    # Avançar para o próximo marco (loop while para casos onde 1 run salta >25%)
                    while next_milestone_idx < len(milestones) and progress_ratio >= milestones[next_milestone_idx]:
                        next_milestone_idx += 1

            # Fim do loop de runs
            elapsed = time.time() - start_run
            print(f" >> [FIM] {ds_name}/{strat} FINALIZADO em {format_time(elapsed)}")

            # Processamento estatístico
            for trans in transforms:
                vals = acc_lists[trans]
                ks = k_lists[trans]
                if not vals: continue
                
                key = f"{ds_name}_{strat}_{trans}"
                results_distribution[key] = vals
                mean_acc = np.mean(vals)
                
                try:
                    m_res = mode(ks, keepdims=False)
                    k_mode = m_res.mode[0] if hasattr(m_res, 'mode') and np.ndim(m_res.mode)>0 else m_res[0]
                except: k_mode = ks[0]

                if mean_acc > global_best_config['accuracy']:
                    global_best_config['accuracy'] = mean_acc
                    global_best_config['dataset'] = ds_name.lower() 
                    global_best_config['strategy'] = strat
                    global_best_config['transform'] = trans
                    global_best_config['k'] = int(k_mode)

    if check_plot_permission(plot_mode, "[5.2] Mostrar gráficos de avaliação?"):
        plot_performance_comparison(results_distribution)
            
    return results_distribution, global_best_config

# ==================================================================================
#                  ESTATÍSTICA: 1 VS ALL (OTIMIZADO)
# ==================================================================================

def _worker_1vsAll(i, datasets, best_cfg, challengers_list, n_runs):
    """ 
    Executa 1 run onde treina o Modelo Vencedor E todos os Desafiantes na MESMA seed.
    Retorna os scores de todos para esta iteração.
    """
    # Desligar threads internas para paralelismo eficiente
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    seed = 20000 + i
    results = {}
    
    # Estratégia do vencedor define o split (para ser justo, usamos o mesmo método de split)
    strat = best_cfg['strategy']
    
    # 1. Preparar Splits (Features e Embeddings) com a MESMA SEED
    splits = {}
    
    # Split Features
    xf, yf, pf = datasets['Features']
    if strat == 'within': sf = split_within_subjects(xf, yf, pf, seed=seed)
    else: sf = split_between_subjects(xf, yf, pf, seed=seed)
    if len(sf.X_train) == 0: return None
    splits['features'] = generate_scenarios(sf, verbose=False)

    # Split Embeddings
    xe, ye, pe = datasets['Embeddings']
    if xe is not None:
        if strat == 'within': se = split_within_subjects(xe, ye, pe, seed=seed)
        else: se = split_between_subjects(xe, ye, pe, seed=seed)
        if len(se.X_train) == 0: return None
        splits['embeddings'] = generate_scenarios(se, verbose=False)
    
    # Função auxiliar para treinar e avaliar
    def evaluate_model(ds_name, trans_name):
        data_map = splits[ds_name] # 'features' ou 'embeddings'
        data_final = data_map[trans_name] # 'pca' ou 'relieff'
        # Usamos k_range reduzido para rapidez no teste estatístico
        model, _, _ = tune_best_k(data_final, k_range=[1, 3, 5])
        return model.score(data_final.X_test, data_final.y_test)

    # 2. Avaliar Vencedor
    score_best = evaluate_model(best_cfg['dataset'], best_cfg['transform'])
    results['best'] = score_best
    
    # 3. Avaliar Desafiantes
    for idx, chall in enumerate(challengers_list):
        if chall['dataset'] == 'embeddings' and 'embeddings' not in splits:
            results[f'chall_{idx}'] = 0.0
            continue
            
        score_c = evaluate_model(chall['dataset'], chall['transform'])
        results[f'chall_{idx}'] = score_c
        
    return results

def run_one_vs_all_stats(datasets_raw, best_config, n_runs=10):
    best_ds = best_config['dataset']
    best_tr = best_config['transform']
    
    print(f"\n" + "="*80)
    print(f"=== 5.3 TESTE DE SIGNIFICÂNCIA: {best_ds.upper()}+{best_tr.upper()} (Best) vs OUTROS ===")
    print("="*80)
    
    possible_ds = ['features', 'embeddings']
    possible_tr = ['all', 'pca', 'relieff']
    
    challengers = []
    for ds in possible_ds:
        if ds == 'embeddings' and datasets_raw['Embeddings'][0] is None: continue
        for tr in possible_tr:
            if ds == best_ds and tr == best_tr: continue 
            challengers.append({'dataset': ds, 'transform': tr})
            
    print(f"  > Executando {n_runs} runs emparelhadas em paralelo (Non-Blocking)...")
    t_start = time.time()
    
    tasks = (delayed(_worker_1vsAll)(i, datasets_raw, best_config, challengers, n_runs) for i in range(n_runs))
    results_gen = Parallel(n_jobs=-2, return_as="generator")(tasks)
    
    valid_results = []
    milestones = [0.25, 0.50, 0.75, 1.0]
    next_milestone_idx = 0
    
    for i, res in enumerate(results_gen):
        if res is not None:
            valid_results.append(res)
        
        # Feedback visual simples
        count = i + 1
        progress = count / n_runs
        if next_milestone_idx < len(milestones) and progress >= milestones[next_milestone_idx]:
             print(f"    --> [STATS] {count}/{n_runs} batalhas processadas ({progress:.0%})...")
             while next_milestone_idx < len(milestones) and progress >= milestones[next_milestone_idx]:
                next_milestone_idx += 1
    
    print(f"    --> [STATS] Processamento concluído.")

    if not valid_results:
        print("  [ERRO] Falha crítica nos splits.")
        return best_ds, best_tr

    scores_best = [r['best'] for r in valid_results]
    mean_best = np.mean(scores_best)
    
    print(f"\n  MODELO VENCEDOR ({best_ds}+{best_tr}): Média = {mean_best:.2%}")
    print("-" * 80)
    print(f"  {'DESAFIANTE':<30} | {'MÉDIA':<10} | {'P-VALUE':<10} | {'CONCLUSÃO'}")
    print("-" * 80)
    
    all_significant = True
    
    for idx, chall in enumerate(challengers):
        name = f"{chall['dataset']}+{chall['transform']}"
        scores_chall = [r[f'chall_{idx}'] for r in valid_results]
        mean_chall = np.mean(scores_chall)
        
        t_stat, p_val_raw = stats.ttest_rel(scores_best, scores_chall, alternative='greater') #type: ignore
        p_val = float(p_val_raw) #type: ignore
        
        sig_label = "SIG (Melhor)" if p_val < 0.05 else "EMPATE/PIOR"
        if p_val >= 0.05: all_significant = False
            
        print(f"  {name:<30} | {mean_chall:.2%}   | {p_val:.1e}   | {sig_label}")

    print("-" * 80)
    if all_significant:
        print("  >>> CONCLUSÃO: O Modelo Vencedor é ESTATISTICAMENTE SUPERIOR a todos os outros.")
    else:
        print("  >>> CONCLUSÃO: O Modelo Vencedor é o melhor na média, mas empata estatisticamente com alguns.")
    
    print(f"  Tempo decorrido: {format_time(time.time()-t_start)}")
    
    return best_ds, best_tr

# ==================================================================================
#                          SECÇÃO 6: DEPLOYMENT (OTIMIZADO)
# ==================================================================================

class ActivityPredictor:
    def __init__(self, model, strategy_type, transform_params):
        self.model = model
        self.strategy = strategy_type 
        self.params = transform_params

    def _extract_features(self, segments):
        if segments.ndim == 2: segments = segments[np.newaxis, ...]
        N = segments.shape[0]

        acc_mag = np.linalg.norm(segments[:, :, 0:3], axis=2, keepdims=True)
        gyr_mag = np.linalg.norm(segments[:, :, 3:6], axis=2, keepdims=True)
        mag_mag = np.linalg.norm(segments[:, :, 6:9], axis=2, keepdims=True)
        combined = np.concatenate([segments, acc_mag, gyr_mag, mag_mag], axis=2)
        X_win = combined.transpose(0, 2, 1)
        
        mean = np.mean(X_win, axis=2)
        median = np.median(X_win, axis=2)
        std = np.std(X_win, axis=2)
        var = np.var(X_win, axis=2)
        energy = np.mean(X_win**2, axis=2)
        rms = np.sqrt(energy)
        q75, q25 = np.percentile(X_win, [75, 25], axis=2)
        iqr = q75 - q25
        
        centered = X_win - mean[:, :, np.newaxis]
        crossings = np.diff(np.sign(centered), axis=2) != 0
        mcr = np.mean(crossings, axis=2)
        
        sk = skew(X_win, axis=2)
        ku = kurtosis(X_win, axis=2)
        
        yf = np.abs(np.fft.rfft(centered, axis=2))[:, :, 1:]
        freqs = np.fft.rfftfreq(256, d=1/51.2)[1:]
        idx_dom = np.argmax(yf, axis=2)
        freq_dom = freqs[idx_dom] 
        spec_en = np.sum(yf**2, axis=2) / (yf.shape[2] + 1e-12)

        entropy = np.zeros((N, 12))
        for i in range(N):
            for c in range(12):
                h, _ = np.histogram(X_win[i, c, :], bins=50, density=True)
                pk = h[h > 0] / h.sum()
                entropy[i, c] = -np.sum(pk * np.log2(pk))

        feats_stack = np.stack([mean, median, std, var, iqr, mcr, sk, ku, energy, rms, entropy, freq_dom, spec_en], axis=2)
        return feats_stack.reshape(N, -1)

    def _extract_embedding(self, segments):
        if segments.ndim == 2: segments = segments[np.newaxis, ...]
        N = segments.shape[0]
        enc = load_harnet_model()
        if enc is None: raise RuntimeError("Model fail")
        
        resampled_batch = []
        for i in range(N):
            r = resample_segment(segments[i, :, 0:3], 51.2, 30.0)
            resampled_batch.append(r.T)
        batch_torch = torch.tensor(np.array(resampled_batch)).float()
        with torch.no_grad(): emb = enc(batch_torch).cpu().numpy()
        return emb.reshape(N, -1)
    
    def predict_batch(self, raw_segments_batch):
        X = np.array(raw_segments_batch)
        
        should_extract = True
        if self.strategy == 'features':
            # Se já tem 156 colunas, assumimos que já foi extraído
            if X.ndim == 2 and X.shape[1] == 156:
                should_extract = False
        else:
            # Embeddings geralmente têm muitas dimensões
            if X.ndim == 2 and X.shape[1] > 20:
                should_extract = False
        
        if should_extract:
            if self.strategy == 'features': X = self._extract_features(X)
            else: X = self._extract_embedding(X)
            
        X = np.nan_to_num(X)
        
        if self.params.get('mu') is not None:
            X = (X - self.params['mu']) / self.params['sigma']
            
        t_type = self.params.get('type', 'all')
        if t_type == 'relieff': X = X[:, self.params['top_cols']]
        elif t_type == 'pca': X = np.dot(X, self.params['proj_matrix'])
            
        return self.model.predict(X)

def build_deployment_model(X_full, y_full, strategy='features', transformation='relieff', k=5):
    print(f"  > Treinando modelo final ({strategy}+{transformation}) com k={k}...")
    
    mu = np.mean(X_full, axis=0)
    sigma = np.std(X_full, axis=0) + 1e-12
    X_std = (X_full - mu) / sigma
    
    params = {'mu': mu, 'sigma': sigma, 'type': transformation}
    X_train_final = X_std.copy()
    
    if transformation == 'relieff':
        cols = np.argsort(reliefF(X_full, y_full))[::-1][:15]
        params['top_cols'] = cols
        X_train_final = X_std[:, cols]
        
    elif transformation == 'pca':
        cov = np.cov(X_std, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        cumulative = np.cumsum(vals[idx] / np.sum(vals))
        n = np.argmax(cumulative >= 0.90) + 1
        pm = vecs[:, idx][:, :n]
        params['proj_matrix'] = pm
        X_train_final = np.dot(X_std, pm)
        
    clf = CustomKNN(n_neighbors=k).fit(X_train_final, y_full)
    return ActivityPredictor(clf, strategy, params)

# ==================================================================================
#                                   MAIN PIPELINE
# ==================================================================================

def main():
    base_path = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['all', 'none', 'default', 'custom'])
    if 'ipykernel' in sys.modules: args = parser.parse_args([]) 
    else: args = parser.parse_args()

    timings = {}
    t_start_pipeline = time.time()
    print(f"=== INICIANDO PIPELINE - {time.ctime()} ===")

    # --- LOAD DATA ---
    plot_mode = args.mode if args.mode else get_plot_mode()
    t0 = time.time()
    all_data = load_all_data(base_path)
    if all_data is None: return
    print(f">>> Dados carregados em {time.time()-t0:.2f}s")

    # --- EX 1 ---
    print("\n" + "="*40 + "\n=== 1. EXTRAÇÃO DE FEATURES & SMOTE ===\n" + "="*40)
    t0 = time.time()
    _, X_feat, y_feat, _, p_feat = extract_basic_features(all_data[:, :11], all_data[:, 11], all_data[:, -1], verbose=False)
    X_feat, y_feat, p_feat = filter_activities(X_feat, y_feat, p_feat)
    analyze_class_balance(y_feat, plot_mode)
    
    smote_config = {'part_id': 3, 'act_id': 4, 'k_gen': 3, 'feat_cols': (0, 2)}
    visualize_smote_participant(X_feat, y_feat, p_feat, plot_mode, **smote_config)
    
    dur_ex1 = time.time() - t0
    timings['Ex 1: Features & SMOTE'] = dur_ex1
    print(f"\n>>> [CONCLUÍDO] Ex 1 finalizado em {format_time(dur_ex1)}")

    # --- EX 2 ---
    print("\n" + "="*40 + "\n=== 2. DATASET DE EMBEDDINGS (HARNet) ===\n" + "="*40)
    t0 = time.time()
    emb_file = "embeddings_dataset.npz"
    X_emb, y_emb, p_emb = None, None, None
    
    if os.path.exists(emb_file):
        d = np.load(emb_file)
        X_emb, y_emb, p_emb = d['X'], d['y'], d['participants']
        if X_emb.ndim > 2: X_emb = X_emb.reshape(X_emb.shape[0], -1)
        print("  > Embeddings carregados.")
    else:
        if plot_mode != 'none':
            res = generate_embeddings_dataset(all_data)
            if res and res[0] is not None:
                X_emb, y_emb, p_emb = res
                np.savez(emb_file, X=X_emb, y=y_emb, participants=p_emb)
    
    dur_ex2 = time.time() - t0
    timings['Ex 2: Embeddings'] = dur_ex2
    print(f">>> [CONCLUÍDO] Ex 2 finalizado em {format_time(dur_ex2)}")

    # --- EX 3 ---
    print("\n=== 3. PREPARAÇÃO DOS DATASETS ===")
    t0 = time.time()
    datasets_raw = { "Features": (X_feat, y_feat, p_feat), "Embeddings": (X_emb, y_emb, p_emb) }
    dur_ex3 = time.time() - t0
    timings['Ex 3: Prep'] = dur_ex3
    print(f">>> [CONCLUÍDO] Dados preparados em {dur_ex3:.4f}s")

    # --- EX 5 (Avaliação) ---
    t0 = time.time()
    n_runs_eval = 14
    
    _, best_config  = run_full_evaluation(datasets_raw, plot_mode=plot_mode, n_runs=n_runs_eval)
    
    dur_ex5 = time.time() - t0
    timings['Ex 5: Full Evaluation'] = (dur_ex5, dur_ex5/n_runs_eval)
    print(f"\n>>> [CONCLUÍDO] Ex 5 finalizado em {format_time(dur_ex5)}")

    print(f"\n[MELHOR CONFIGURAÇÃO GLOBAL]: {best_config['dataset'].upper()} | {best_config['strategy']} | {best_config['transform']} | K={best_config['k']} (Acc: {best_config['accuracy']:.2%})")

    # --- EX 5.3 (Torneio) ---
    if (X_feat is not None) and (plot_mode != 'none'):
        if datasets_raw['Features'][0] is not None:
            t_start_stats = time.time()
            
            # Executa a comparação 1 vs Todos
            win_ds, win_trans = run_one_vs_all_stats(datasets_raw, best_config, n_runs=14)
            
            dur_stats = time.time() - t_start_stats
            timings['Ex 5.3: Significance Test'] = dur_stats

            print("\n  > Gerando Matriz de Confusão para o Vencedor Global...")
            
            # Recuperar dados do vencedor
            if win_ds == 'features':
                X_win, y_win, p_win = datasets_raw['Features']
            else:
                X_win, y_win, p_win = datasets_raw['Embeddings']
                
            # Split e Transformação (Uma execução representativa para o gráfico)
            win_strat = best_config['strategy']
            if win_strat == 'within':
                split_win = split_within_subjects(X_win, y_win, p_win, seed=123)
            else:
                split_win = split_between_subjects(X_win, y_win, p_win, seed=123)
                
            scenarios_win = generate_scenarios(split_win, verbose=False)
            data_win_final = scenarios_win[win_trans]
            
            # Treino e Predição
            model_win, _, _ = tune_best_k(data_win_final, k_range=[1, 3, 5, 7])
            preds_win = model_win.predict(data_win_final.X_test)
            
            # Plot
            plot_confusion_matrix(data_win_final.y_test, preds_win, f"{win_ds.upper()} + {win_trans.upper()}", plot_mode)

    # --- EX 6 ---
    best_strat_name = best_config['dataset']
    best_trans_name = best_config['transform']
    best_k_val = best_config['k']
    
    print(f"\n" + "="*40 + f"\n=== 6. DEPLOYMENT (Modelo: {best_strat_name}+{best_trans_name}, k={best_k_val}) ===\n" + "="*40)
    t0 = time.time()
    
    if best_strat_name == 'features':
        X_curr, y_curr, p_curr = X_feat, y_feat, p_feat
    else:
        X_curr, y_curr, p_curr = X_emb, y_emb, p_emb
        
    if X_curr is not None:
        # 1. Obter Split Rigoroso (60/20/20) para avaliar performance
        if best_config['strategy'] == 'within':
            split_deploy = split_within_subjects(X_curr, y_curr, p_curr, seed=999)
        else:
            split_deploy = split_between_subjects(X_curr, y_curr, p_curr, seed=999)
            
        X_final_train, y_final_train = split_deploy.get_train_val_combined()
        
        # 2. Treinar Modelo (Train + Val = 80%)
        # Este modelo será usado tanto para a métrica de teste como para a simulação
        model_deploy = build_deployment_model(X_final_train, y_final_train, best_strat_name, best_trans_name, best_k_val)
        
        # 3. Teste A: Validação de Performance (Métrica Real)
        print("\n>>> FASE A: Validação de Performance (Split Rigoroso 80/20)")
        print(f"    Treino: {len(X_final_train)} amostras | Teste: {len(split_deploy.X_test)} amostras")
        
        preds_deploy = model_deploy.predict_batch(split_deploy.X_test)
        acc_deploy = accuracy_score(split_deploy.y_test, preds_deploy)
        print(f"    -> Acurácia no Test Set: {acc_deploy:.2%}")

        # 4. Teste B: Verificação de Funcionalidade (Input Bruto)
        print("\n>>> FASE B: Teste de Funcionalidade (Input Raw Data)")
        print("    A testar a função 'predict_batch' com segmentos brutos aleatórios...")
        
        total = 5000 
        rng = np.random.default_rng(123)
        raw_segments = []
        true_labels = []
        valid_count = 0; attempts = 0
        
        while valid_count < total:
            idx = int(rng.integers(0, all_data.shape[0]-256))
            if np.all(all_data[idx:idx+256, 11] == all_data[idx, 11]) and 1 <= all_data[idx, 11] <= 7:
                raw_segments.append(all_data[idx:idx+256, 1:10])
                true_labels.append(int(all_data[idx, 11]))
                valid_count += 1
            attempts += 1
            if attempts > total * 10: break 
            
        # Usa o MESMO modelo (treinado em 80%) para prever novos dados brutos
        predictions = model_deploy.predict_batch(raw_segments)
        hits = np.sum(predictions == true_labels)
        print(f"    -> Pipeline executada sem erros. Acurácia estimada: {hits/len(true_labels):.2%}")

    dur_ex6 = time.time() - t0
    timings['Ex 6: Deployment'] = dur_ex6
    print(f">>> [CONCLUÍDO] Deployment finalizado em {format_time(dur_ex6)}")
    
    print_timing_table(timings)
    print(f"TEMPO TOTAL DO SCRIPT: {format_time(time.time() - t_start_pipeline)}")

if __name__ == "__main__":
    main()