# Made by Rafael Bernardo and Diogo

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.spatial import distance
from numpy.lib.stride_tricks import sliding_window_view

# ==================================================================================
#                                   AUXILIARES DE IO
# ==================================================================================

def get_plot_mode():
    """
    Menu inicial para definir o comportamento dos gráficos.
    Retorna: 'all', 'none', 'default', 'custom'
    """
    print("\n" + "="*50)
    print("ESCOLHA O MODO DE EXIBIÇÃO DE GRÁFICOS")
    print("="*50)
    print("1 - Todos (Plotar tudo com 100% dos dados)")
    print("2 - Nenhum (Apenas cálculos e prints) [DEFAULT]")
    print("3 - Default (Plotar gráficos representativos - define % uma vez)")
    print("4 - Personalizado (Perguntar gráfico a gráfico)")
    print("="*50)
    
    choice = input("Opção [1-4] (Enter = Nenhum): ").strip()
    
    if choice == '1': return 'all'
    if choice == '3': return 'default'
    if choice == '4': return 'custom'
    
    print("-> Modo selecionado: Nenhum")
    return 'none'

def check_plot_permission(mode, prompt_text, is_repetitive_variant=False, is_first_variant=True):
    """
    Decide se deve plotar com base no modo escolhido.
    """
    if mode == 'all':
        return True
    if mode == 'none':
        return False
    if mode == 'default':
        # Se for repetitivo (loops), plota apenas o primeiro
        if not is_repetitive_variant:
            return True
        return is_first_variant
    if mode == 'custom':
        return input(f"\n{prompt_text} [s/n]: ").strip().lower() == 's'
    return False

# ==================================================================================
#                                   CARREGAMENTO
# ==================================================================================

def _load_single_file(file_path, participant_id=None):
    try:
        data = pd.read_csv(file_path, header=None).to_numpy()
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if participant_id is not None:
            part_col = np.full((data.shape[0], 1), participant_id)
            data = np.hstack((data, part_col))
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        return None

def load_all_data(base_path, dataset_folder="FORTH_TRACE_DATASET-master"):
    all_files_data = []
    print("--- A carregar dados de todos os participantes e dispositivos ---")
    full_path = os.path.join(base_path, dataset_folder)
    
    if not os.path.exists(full_path):
         print(f"[ERRO] Pasta não encontrada em: {full_path}")
         return None

    for part_id in range(15):
        for dev_id in range(1, 6):
            folder_name = f'part{part_id}'
            file_name = f'part{part_id}dev{dev_id}.csv'
            file_path = os.path.join(full_path, folder_name, file_name)
            data = _load_single_file(file_path, participant_id=part_id)
            if data is not None:
                all_files_data.append(data)

    if not all_files_data:
        print("[ERRO]: Nenhum ficheiro foi carregado.")
        return None

    combined = np.vstack(all_files_data)
    print(f"Dados carregados com sucesso: {combined.shape[0]} linhas, {combined.shape[1]} colunas.")
    return combined

def calculate_magnitudes(features):
    # Vetorizado pelo numpy por defeito
    accel_mag = np.linalg.norm(features[:, 1:4], axis=1)
    gyro_mag  = np.linalg.norm(features[:, 4:7], axis=1)
    mag_mag   = np.linalg.norm(features[:, 7:10], axis=1)
    return accel_mag, gyro_mag, mag_mag

# ==================================================================================
#                                   GRÁFICOS E HELPERS
# ==================================================================================

def ask_sample_percent(default=100.0, context="gráfico", forced_percent=None):
    if forced_percent is not None:
        p = max(0.0, min(100.0, float(forced_percent)))
        if p < 100.0:
            print(f"→ Exibindo {p:.1f}% dos pontos no {context} (definido globalmente).")
        return p

    prompt = f"Quantos % dos dados usar para o {context}? (0-100, default={default}): "
    s = input(prompt).strip()
    if s == "":
        p = float(default)
    else:
        try:
            p = float(s)
        except ValueError:
            print("Entrada inválida — usando valor por omissão.")
            p = float(default)
    p = max(0.0, min(100.0, p))
    print(f"→ Exibindo {p:.1f}% dos pontos no {context} (os cálculos usam 100%).")
    return p

def sample_for_plot(data, labels=None, percent=100.0):
    n = len(data)
    if percent >= 100 or n <= 1:
        idx = np.arange(n)
        if labels is None: return data, None, idx
        return data, labels, idx

    m = max(1, int(n * (percent / 100.0)))
    idx = np.random.choice(n, m, replace=False)
    idx = np.sort(idx)
    if labels is None: return data[idx], None, idx
    return data[idx], labels[idx], idx

def plot_activity_boxplot(data_vector, labels, title, global_percent=None):
    percent = ask_sample_percent(100.0, f"boxplot '{title}'", forced_percent=global_percent)
    data_s, labels_s, _ = sample_for_plot(data_vector, labels, percent)

    if labels_s is None: return
    activity_ids = np.unique(labels_s).astype(int)
    data_by_activity = [data_s[labels_s == activity] for activity in activity_ids]

    plt.figure(figsize=(15, 8))
    plt.boxplot(data_by_activity, tick_labels=activity_ids)
    plt.title(title, fontsize=16)
    plt.xlabel("ID da Atividade", fontsize=12)
    plt.ylabel("Módulo do Vetor", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    print(f"A exibir: '{title}'.")
    plt.show()

def compute_iqr_density(activity_data):
    if activity_data.size == 0: return 0.0
    q1, q3 = np.percentile(activity_data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    no = np.count_nonzero((activity_data < lower) | (activity_data > upper))
    return (no / activity_data.size) * 100.0

def analyze_outlier_density(all_data):
    print("\n--- Requisito 3.2: Densidade de Outliers (Pulso Direito) ---")
    right_wrist_data = all_data[all_data[:, 0] == 2]
    if right_wrist_data.shape[0] == 0:
        print("Sem dados do device 2.")
        return
    
    features = right_wrist_data[:, :11]
    labels = right_wrist_data[:, 11]
    accel_mag, gyro_mag, mag_mag = calculate_magnitudes(features)
    magnitudes = [("Acelerómetro", accel_mag), ("Giroscópio", gyro_mag), ("Magnetómetro", mag_mag)]
    
    for name, mag_data in magnitudes:
        print(f"\nDensidade de Outliers para o {name}:")
        print("-" * 50)
        print(f"{'Atividade':<12} | {'Densidade (%)':<25}")
        print("-" * 50)
        for activity in np.unique(labels).astype(int):
            density = compute_iqr_density(mag_data[labels == activity])
            print(f"{activity:<12} | {density:>25.2f}%")
        print("-" * 50)

def identify_outliers_zscore(data, k=3.0):
    mean, std = np.mean(data), np.std(data)
    if std == 0: return np.array([])
    z_scores = np.abs((data - mean) / std)
    return np.where(z_scores > k)[0]

def analyze_outliers_with_zscore(all_features, all_labels, k=3.0):
    print(f"\n--- Requisito 3.3: Z-Score (k={k}) ---")
    accel_mag, gyro_mag, mag_mag = calculate_magnitudes(all_features)
    magnitudes = [("Acelerómetro", accel_mag), ("Giroscópio", gyro_mag), ("Magnetómetro", mag_mag)]
    for name, mag_data in magnitudes:
        print(f"\nOutliers no {name}:")
        print("-" * 50)
        print(f"{'Atividade':<12} | {'Amostras':<15} | {'Outliers':<10}")
        print("-" * 50)
        for activity in np.unique(all_labels).astype(int):
            activity_data = mag_data[all_labels == activity]
            outliers = identify_outliers_zscore(activity_data, k)
            print(f"{activity:<12} | {len(activity_data):<15} | {len(outliers):<10}")
        print("-" * 50)

def plot_outliers_by_zscore(data, labels, k, title, show_plot=True, global_percent=None):
    """
    Requisito 3.4: Realiza sempre o cálculo. Exibe apenas se show_plot=True.
    """
    # Cálculo para garantir execução (simulado pela contagem)
    _ = len(identify_outliers_zscore(data, k))
    
    if not show_plot:
        return

    percent = ask_sample_percent(100.0, f"gráfico de outliers '{title}'", forced_percent=global_percent)
    data_s, labels_s, _ = sample_for_plot(data, labels, percent)
    if labels_s is None: return

    activity_ids = np.unique(labels_s).astype(int)
    plt.figure(figsize=(15, 8))

    for activity in activity_ids:
        activity_data = data_s[labels_s == activity]
        outlier_idx = identify_outliers_zscore(activity_data, k)
        mask = np.ones_like(activity_data, dtype=bool)
        mask[outlier_idx] = False

        plt.scatter(np.full(np.sum(mask), activity), activity_data[mask], c='blue', s=10, alpha=0.6)
        plt.scatter(np.full(len(outlier_idx), activity), activity_data[outlier_idx], c='red', s=15, alpha=0.9)

    plt.title(f'{title} — Z-Score (k={k})', fontsize=16)
    plt.xlabel('ID da Atividade', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(['Normal', 'Outliers'], loc='upper right')
    print(f"A exibir: '{title}' (k={k}).")
    plt.show()

def kmeans_clustering(data, n_clusters, max_iters=100, batch_size=50000):
    n_samples, n_features = data.shape
    rng = np.random.default_rng()
    centroids = data[rng.choice(n_samples, n_clusters, replace=False)]
    labels = np.zeros(n_samples, dtype=int)

    for it in range(max_iters):
        old_centroids = centroids.copy()
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            chunk = data[start:end]
            dists = distance.cdist(chunk, centroids, metric='sqeuclidean')
            labels[start:end] = np.argmin(dists, axis=1)

        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centroids[k] = data[mask].mean(axis=0)
                
        if np.allclose(old_centroids, centroids): break

    chosen_centroids = centroids[labels]
    inertia = np.sum((data - chosen_centroids)**2)
    return centroids, labels, inertia

def plot_kmeans_results(data, labels, centroids, title, batch_size=50000, global_percent=None):
    percent = ask_sample_percent(100.0, f"gráfico k-means 2D '{title}'", forced_percent=global_percent)
    data_s, labels_s, _ = sample_for_plot(data, labels, percent)
    if labels_s is None or len(labels_s) == 0: return

    plt.ioff()
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.colormaps.get('viridis') or plt.get_cmap('viridis')
    denom = labels_s.max() if labels_s.max() > 0 else 1
    colors = cmap(labels_s / denom)

    n_points = len(data_s)
    for start in range(0, n_points, batch_size):
        end = start + batch_size
        ax.scatter(data_s[start:end, 0], data_s[start:end, 1],
                   c=colors[start:end], s=20, edgecolors='none', rasterized=True)

    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=120, linewidth=2, label='Centróides')
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.show(block=False)
    plt.ion()

def analyze_outliers_kmeans_3d(all_features, plot_mode, n_clusters_list=[3, 5, 7], batch_size=5000, global_percent=None):
    """
    Requisito 3.7: Calcula sempre usando K-Means otimizado. Plota conforme plot_mode.
    """
    print("\n--- Requisito 3.7: Outliers com K-Means (3D) ---")
    xyz = all_features[:, 1:4]

    plt.ioff()
    cmap = plt.colormaps.get('viridis') or plt.get_cmap('viridis')

    def make_legend(unique_labels, denom, has_outliers):
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(l / denom), markersize=8, label=f'Cluster {l}') for l in unique_labels]
        if has_outliers:
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Outliers'))
        handles.append(Line2D([0], [0], marker='x', color='black', markersize=8, label='Centróides'))
        return handles

    for i, n_clusters in enumerate(n_clusters_list):
        centroids, labels, _ = kmeans_clustering(xyz, n_clusters)
        distances = np.zeros(len(xyz))
        for start in range(0, len(xyz), batch_size):
            end = start + batch_size
            distances[start:end] = np.linalg.norm(xyz[start:end] - centroids[labels[start:end]], axis=1)
        threshold = np.mean(distances) + 3 * np.std(distances)
        outlier_mask = distances > threshold

        print(f"\nClusters: {n_clusters}")
        print(f"Outliers: {np.sum(outlier_mask)} de {len(xyz)} pontos")

        # Decisão Plot
        is_first = (i == 0)
        should_plot = check_plot_permission(plot_mode, f"K-Means 3D (n={n_clusters})", is_repetitive_variant=True, is_first_variant=is_first)
        
        if not should_plot: continue

        percent = ask_sample_percent(100.0, f"K-Means 3D (n={n_clusters})", forced_percent=global_percent)
        xyz_non = xyz[~outlier_mask]
        labels_non = labels[~outlier_mask]
        xyz_s, labels_s, _ = sample_for_plot(xyz_non, labels_non, percent)
        if labels_s is None: continue

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        denom = max(labels.max(), 1)
        unique_labels = np.unique(labels_s)
        cluster_colors = [cmap(l / denom) for l in unique_labels]

        for i_c, cluster_label in enumerate(unique_labels):
            mask = labels_s == cluster_label
            ax.scatter(xyz_s[mask, 0], xyz_s[mask, 1], xyz_s[mask, 2],
                       color=cluster_colors[i_c], s=4, alpha=0.65, edgecolors='none', rasterized=True) #type: ignore
        if np.any(outlier_mask):
            ax.scatter(xyz[outlier_mask, 0], xyz[outlier_mask, 1], xyz[outlier_mask, 2],
                       c='red', s=18, edgecolors='none', alpha=0.01) #type: ignore
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='x', s=80) #type: ignore
        ax.legend(handles=make_legend(unique_labels, denom, np.any(outlier_mask)), loc='best', fontsize=8)
        ax.set_title(f"(n={n_clusters}) clusters — Outliers em vermelho")
        fig.tight_layout()
        plt.show(block=False)

    plt.ion()

def KS_significance_test(all_features, all_labels):
    accel_mag, gyro_mag, mag_mag = calculate_magnitudes(all_features)
    sensors = {"Acelerómetro": accel_mag, "Giroscópio": gyro_mag, "Magnetómetro": mag_mag}
    results = {}

    for name, vec in sensors.items():
        acts = np.unique(all_labels).astype(int)
        groups = [vec[all_labels == a] for a in acts if len(vec[all_labels == a]) > 1]
        if not groups or len(groups) < 2: continue

        normals = []
        for g in groups:
            mu, sd = float(np.mean(g)), float(np.std(g) or 1e-12)
            _, pks = stats.kstest(g, 'norm', args=(mu, sd))
            normals.append(pks > 0.05)

        if all(normals):
            f, p = stats.f_oneway(*groups)
            test_name = "ANOVA"
        else:
            f, p = stats.kruskal(*groups)
            test_name = "Kruskal–Wallis"

        sig = p < 0.05
        print(f"[4.1] {name}: {test_name} → estat={f:.4f}, p={p:.3e}")
        print(f"     → {'Significativo' if sig else 'Não significativo'}")
        results[name] = {"test": test_name, "stat": f, "p_value": p, "significant": sig}
    return results

def extract_basic_features(all_features, all_labels, participants_all, fs=51.2, window_seconds=5, overlap=0.5, min_samples_per_activity=10, verbose=True):
    if verbose:print("\n[4.2] A iniciar extração de features vetorizada (12 Canais x 13 Stats)...")
    
    # 1. Preparar Dados
    # Tratar NaNs iniciais
    all_features_safe = np.nan_to_num(all_features)
    
    # Extrair Eixos Individuais (Raw Axes: AccXYZ, GyrXYZ, MagXYZ)
    raw_axes = all_features_safe[:, 1:10] # Shape (N, 9)
    
    # Calcular Magnitudes (Rotational Invariant)
    acc_mag = np.linalg.norm(all_features_safe[:, 1:4], axis=1)
    gyr_mag = np.linalg.norm(all_features_safe[:, 4:7], axis=1)
    mag_mag = np.linalg.norm(all_features_safe[:, 7:10], axis=1)
    
    # Stack de TUDO: 9 Eixos + 3 Magnitudes = 12 Canais
    # Ordem: AccX, AccY, AccZ, GyrX, GyrY, GyrZ, MagX, MagY, MagZ, AccMag, GyrMag, MagMag
    combined_signals = np.column_stack((raw_axes, acc_mag, gyr_mag, mag_mag))
    
    # Parâmetros Janela
    window_size = int(window_seconds * fs)
    step_size = int(window_size * (1 - overlap))
    
    # 2. Janelas Deslizantes (Efficient View)
    # Shape inicial: (N_janelas, Window_Size, 12_canais) porque axis=0
    windows = sliding_window_view(combined_signals, window_shape=window_size, axis=0)[::step_size]
    
    
    labels_windows = sliding_window_view(all_labels, window_shape=window_size, axis=0)[::step_size]
    parts_windows = sliding_window_view(participants_all, window_shape=window_size, axis=0)[::step_size]
    
    # 3. Filtragem (Consistência de Labels)
    valid_mask = np.all(labels_windows == labels_windows[:, [0]], axis=1)
    
    X_windows = windows[valid_mask]
    y_labels = labels_windows[valid_mask, 0]
    y_parts = parts_windows[valid_mask]
    
    # Filtro de min_samples
    unique, counts = np.unique(y_labels, return_counts=True)
    valid_activities = unique[counts >= min_samples_per_activity]
    activity_mask = np.isin(y_labels, valid_activities)
    
    X_windows = X_windows[activity_mask]
    y = y_labels[activity_mask].astype(int)
    y_parts = y_parts[activity_mask]
    
    participants_windowed = np.array([np.bincount(row.astype(int)).argmax() for row in y_parts], dtype=int)
    
    if verbose:print(f"Processando {X_windows.shape[0]} janelas com 12 Canais...")

    # 4. Cálculo de Features (13 stats por canal)    
    # --- TEMPO ---
    if verbose:print("  > [1/4] A calcular estatísticas temporais (Mean, Var, Skew, Kurtosis, ZCR)...")
    mean_val = np.mean(X_windows, axis=2)
    median_val = np.median(X_windows, axis=2)
    std_val  = np.std(X_windows, axis=2)
    var_val = np.var(X_windows, axis=2)
    energy   = np.mean(X_windows**2, axis=2)
    rms      = np.sqrt(energy)
    
    # IQR (Interquartile Range)
    q75, q25 = np.percentile(X_windows, [75 ,25], axis=2)
    iqr_val = q75 - q25
    
    # Mean Crossing Rate (MCR) - Cruzamentos da média (oscilação)
    centered = X_windows - mean_val[:, :, np.newaxis]
    # diff(sign) != 0 deteta cruzamento de zero do sinal centrado
    crossings = np.diff(np.sign(centered), axis=2) != 0
    mcr_val = np.mean(crossings, axis=2)
    
    # Skewness & Kurtosis (scipy.stats vetorizado)
    skew_val = skew(X_windows, axis=2)
    kurt_val = kurtosis(X_windows, axis=2)

    # --- FREQUÊNCIA ---
    if verbose:print("  > [2/4] A calcular FFT e domínio da frequência...")
    # Usar numpy.fft para compatibilidade Pylance
    yf = np.abs(np.fft.rfft(centered, axis=2)) 
    yf = yf[:, :, 1:] # Remover DC
    
    freqs = np.fft.rfftfreq(window_size, d=1/fs)[1:]
    
    idx_dom = np.argmax(yf, axis=2)
    freq_dom = freqs[idx_dom]
    spectral_energy = np.sum(yf**2, axis=2) / (yf.shape[2] + 1e-12)

    # --- ENTROPIA (Loop Híbrido) ---
    if verbose:print("  > [3/4] A calcular Entropia (Iterando por canais)...")
    def calc_entropy_row(row):
        hist, _ = np.histogram(row, bins=50, density=True)
        pk = hist[hist > 0]
        pk = pk / pk.sum() # Normalizar prob
        return -np.sum(pk * np.log2(pk))

    entropy_list = []
    # Iterar pelos 12 canais
    for s in range(X_windows.shape[1]):
        if verbose:print(f"      Processando Canal {s+1}/12...", end='\r')
        ent = np.array([calc_entropy_row(row) for row in X_windows[:, s, :]])
        entropy_list.append(ent)
    entropy = np.column_stack(entropy_list)
    if verbose:print("      Processando Canal 12/12... Feito.")

    # 5. Montagem Final (Stack das 13 estatísticas)
    if verbose:print("  > [4/4] A consolidar dataset final...")
    # Shape resultante do stack: (N, 12_canais, 13_features)
    features_stack = np.stack([
        mean_val, median_val, std_val, var_val, iqr_val, mcr_val, 
        skew_val, kurt_val, energy, rms, entropy, freq_dom, spectral_energy
    ], axis=2) 
    
    # Flatten: (N, 156)
    X = features_stack.reshape(X_windows.shape[0], -1)
    
    # Tratar eventuais NaNs gerados por skew/kurt em sinais constantes
    X = np.nan_to_num(X)

    # Nomes das Features
    base_feats = [
        "mean", "median", "std", "var", "iqr", "mcr", "skew", "kurt", 
        "energy", "rms", "entropy", "freq_dom", "spectral_energy"
    ]
    # Lista atualizada de canais (12)
    channel_names = [
        "Acc_x", "Acc_y", "Acc_z", 
        "Gyr_x", "Gyr_y", "Gyr_z", 
        "Mag_x", "Mag_y", "Mag_z",
        "Acc_mag", "Gyr_mag", "Mag_mag"
    ]
    
    feature_names = [f"{ch}_{f}" for ch in channel_names for f in base_feats]

    # Sumário (para debug/print)
    features_summary = {}
    n_feat_per_channel = len(base_feats)
    for act in np.unique(y):
        idx = np.where(y == act)[0]
        feats_mean = np.mean(X[idx, :], axis=0)
        features_summary[int(act)] = {}
        for i, ch_name in enumerate(channel_names):
            features_summary[int(act)][ch_name] = {
                fname: float(feats_mean[i*n_feat_per_channel + j]) for j, fname in enumerate(base_feats)
            }

    if verbose:print(f"Features extraídas com sucesso. Total features: {X.shape[1]} (Esperado: 156)")
    return features_summary, X, y, feature_names, participants_windowed

def perform_pca_analysis(X):
    if X is None or len(X) == 0:
        print("[4.3] Nenhum dado disponível para PCA.")
        return None, None, None, None
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-12
    X_norm = (X - X_mean) / X_std
    cov_matrix = np.cov(X_norm, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    explained_variance_ratio = eig_vals / np.sum(eig_vals)
    print(f"[4.3] PCA concluído: {len(eig_vals)} componentes principais extraídos.")
    return X_norm, eig_vals, eig_vecs, explained_variance_ratio

def analyze_pca_variance(eig_vals, explained_variance_ratio, threshold=0.75, show_plot=True):
    if eig_vals is None or explained_variance_ratio is None: return None, None
    cumulative_ratio = np.cumsum(explained_variance_ratio)
    num_components = np.argmax(cumulative_ratio >= threshold) + 1
    print("\n[4.4] Variância explicada acumulada por componente (Top 5):")
    for i, val in enumerate(cumulative_ratio[:5]):
        print(f"  Até PC{i+1}: {val*100:.2f}%")
    print(f"\n[4.4] Nº mínimo de componentes para explicar {threshold*100:.0f}% da variância: {num_components}")

    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(cumulative_ratio) + 1), cumulative_ratio * 100, marker='o')
        plt.axhline(threshold * 100, color='r', linestyle='--', label=f'{threshold*100:.0f}% limite')
        plt.xlabel('Número de componentes principais')
        plt.ylabel('Variância explicada (%)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
    return num_components, cumulative_ratio

def analyze_feature_selection(X, y, feature_names=None):
    if X is None or y is None:
        print("[4.5/4.6] Nenhum dado disponível.")
        return
    print("\n--- Requisito 4.5 / 4.6: Fisher Score e ReliefF ---")
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    fisher = fisher_score(X_norm, y)
    relief = reliefF(X, y)

    if feature_names is None:
        feature_names = [f"feat_{i+1}" for i in range(X.shape[1])]

    k = min(10, X.shape[1])
    top_fisher_idx = np.argsort(fisher)[-k:][::-1]
    top_relief_idx = np.argsort(relief)[-k:][::-1]

    print("\nTop {} Fisher Score:".format(k))
    for i in top_fisher_idx: print(f"{feature_names[i]:>30s} : {fisher[i]:.6f}")

    print("\nTop {} ReliefF:".format(k))
    for i in top_relief_idx: print(f"{feature_names[i]:>30s} : {relief[i]:.6f}")

    common = set(top_fisher_idx) & set(top_relief_idx)
    print("\nCommon features (indices):", sorted(list(common)))

def fisher_score(X, y):
    mean_total = np.mean(X, axis=0)
    unique_labels = np.unique(y)
    
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    
    for label in unique_labels:
        X_c = X[y == label]
        n_c = X_c.shape[0]
        if n_c == 0: continue
        
        mean_c = np.mean(X_c, axis=0)
        var_c = np.var(X_c, axis=0)
        
        num += n_c * (mean_c - mean_total) ** 2
        den += n_c * var_c

    return num / (den + 1e-12)

def reliefF(X, y, n_neighbors=10):
    """
    Implementação otimizada do ReliefF (Vetorizada).
    """
    # 1. Normalização Rápida (Z-Score)
    # Remove NaN e normaliza features para escala comparável
    X_norm = stats.zscore(X, axis=0, nan_policy='omit')
    X_norm = np.nan_to_num(X_norm)
    
    n_samples, n_features = X_norm.shape
    
    # 2. Matriz de Distâncias 
    dist_matrix = distance.cdist(X_norm, X_norm, metric='euclidean')
    np.fill_diagonal(dist_matrix, np.inf) 
    
    weights = np.zeros(n_features)
    sorted_indices_matrix = np.argsort(dist_matrix, axis=1)
    
    # 3. Iteração Otimizada
    for i in range(n_samples):
        sorted_indices = sorted_indices_matrix[i]
        sorted_labels = y[sorted_indices]
        current_label = y[i]
        
        hit_mask = (sorted_labels == current_label)
        miss_mask = (~hit_mask)
        
        hits = sorted_indices[hit_mask][:n_neighbors]
        misses = sorted_indices[miss_mask][:n_neighbors]
        
        if len(hits) == 0 or len(misses) == 0: continue
        
        hit_diff = np.mean(np.abs(X_norm[i] - X_norm[hits]), axis=0)
        miss_diff = np.mean(np.abs(X_norm[i] - X_norm[misses]), axis=0)
        
        weights += (miss_diff - hit_diff)
        
    return weights

# ==================================================================================
#                                      MAIN
# ==================================================================================

def main():
    try:
        base_path = os.path.dirname(__file__)
        all_data = load_all_data(base_path)
        if all_data is None: return
        
        print(f"\nDados carregados. Shape: {all_data.shape}")
        all_features, all_labels = all_data[:, :11], all_data[:, 11]
        accel_mag, gyro_mag, mag_mag = calculate_magnitudes(all_features)

        # SELEÇÃO DO MODO DE GRÁFICOS
        plot_mode = get_plot_mode()
        print(f"Modo selecionado: {plot_mode}")

        # DEFINIÇÃO GLOBAL DA PERCENTAGEM
        global_percent = None

        if plot_mode == 'all':
            global_percent = 100.0
        elif plot_mode == 'default':
            while True:
                val = input("\nQue percentagem de dados usar para TODOS os gráficos? (0-100, Enter=100): ").strip()
                if val == "":
                    global_percent = 100.0
                    break
                try:
                    p = float(val)
                    global_percent = max(0.0, min(100.0, p))
                    break
                except ValueError:
                    print("Valor inválido.")

        # ----------------- PARTE 3 -----------------
        if check_plot_permission(plot_mode, "[3.1] Mostrar boxplots das magnitudes?"):
            plot_activity_boxplot(accel_mag, all_labels, "Boxplot Acelerómetro", global_percent)
            plot_activity_boxplot(gyro_mag, all_labels, "Boxplot Giroscópio", global_percent)
            plot_activity_boxplot(mag_mag, all_labels, "Boxplot Magnetómetro", global_percent)

        # [3.2]
        do_32 = True
        if plot_mode == 'custom':
            do_32 = input("\n[3.2] Calcular densidade de outliers (Pulso Direito)? [s/n]: ").strip().lower() == 's'
        
        if do_32:
            analyze_outlier_density(all_data)

        # [3.3]
        do_33 = True
        if plot_mode == 'custom':
            do_33 = input("\n[3.3] Calcular outliers via Z-Score (Texto)? [s/n]: ").strip().lower() == 's'
            
        if do_33:
            analyze_outliers_with_zscore(all_features, all_labels)

        print("\n--- Executando 3.4 (Cálculos de Z-Score gráficos) ---")
        for sensor, mag in [("Acelerómetro", accel_mag), ("Giroscópio", gyro_mag), ("Magnetómetro", mag_mag)]:
            for i, k in enumerate([3.0, 3.5, 4.0]):
                is_first = (i == 0)
                should_plot = check_plot_permission(plot_mode, f"  Mostrar gráfico {sensor} (k={k})?", 
                                                    is_repetitive_variant=True, is_first_variant=is_first)
                plot_outliers_by_zscore(mag, all_labels, k, f'Outliers no {sensor}', show_plot=should_plot, global_percent=global_percent)

        print("\n--- Executando 3.6 (K-Means 2D) ---")
        for sensor, mag in [("Acelerómetro", accel_mag)]:
            data_2d = np.column_stack((mag, all_features[:, 1]))
            for i, n in enumerate([3, 5, 7]):
                centroids, labels, inertia = kmeans_clustering(data_2d, n)
                print(f"{sensor}: {n} clusters → Inércia = {inertia:.2f}")
                
                is_first = (i == 0)
                should_plot = check_plot_permission(plot_mode, f"  > Mostrar gráfico para k={n}?", 
                                                    is_repetitive_variant=True, is_first_variant=is_first)
                if should_plot:
                    plot_kmeans_results(data_2d, labels, centroids, f"{sensor} (k={n})", global_percent=global_percent)

        analyze_outliers_kmeans_3d(all_features, plot_mode, n_clusters_list=[3, 5, 7], global_percent=global_percent)
        
        # ----------------- PARTE 4 -----------------
        
        features_summary = None
        X, y, feature_names, participants_windowed = None, None, None, None
        eig_vals, explained_variance_ratio = None, None

        # [4.1]
        do_41 = True
        if plot_mode == 'custom':
            do_41 = input("\n[4.1] Avaliar significância estatística? [s/n]: ").strip().lower() == 's'
        
        if do_41:
            KS_significance_test(all_features, all_labels)
            
        # [4.2]
        print("\n4.2: Extração de Features")
        participants_all = all_data[:, -1]
        features_summary, X, y, feature_names, participants_windowed = extract_basic_features(all_features, all_labels, participants_all)
        print(f"Dataset X criado: {X.shape}")
        
        # [4.3]
        if X is not None:
            do_43 = True
            if plot_mode == 'custom':
                do_43 = input("\n[4.3] Executar PCA sobre o conjunto de features? [s/n]: ").strip().lower() == 's'
            
            if do_43:
                _, eig_vals, _, explained_variance_ratio = perform_pca_analysis(X)
        else:
            if plot_mode == 'custom':
                print("Nenhum feature set encontrado (Passo 4.2 ignorado).")
        
        # [4.4]
        if eig_vals is not None:
            show_pca_plot = True
            if plot_mode == 'none':
                show_pca_plot = False
            elif plot_mode == 'custom':
                show_pca_plot = input("\n[4.4] Mostrar gráfico de variância do PCA? [s/n]: ").strip().lower() == 's'
                
            analyze_pca_variance(eig_vals, explained_variance_ratio, threshold=0.75, show_plot=show_pca_plot)
        
        # [4.5/4.6] Feature Selection
        if X is not None:
            do_fs = True
            if plot_mode == 'custom':
                do_fs = input("\n[4.5/4.6] Calcular Fisher Score e ReliefF? [s/n]: ").strip().lower() == 's'
            
            if do_fs:
                analyze_feature_selection(X, y, feature_names)
        
        # Exportação
        if (X is not None) and (y is not None) and (participants_windowed is not None) and (feature_names is not None):
            output_path = os.path.join(os.path.dirname(__file__), "features_moduleA.npz")
            np.savez(output_path, X=X, y=y, participants=participants_windowed, feature_names=feature_names)
            print(f"\nFicheiro '{output_path}' guardado com sucesso!")
        else:
            print("\nNada exportado (features não extraídas).")

    except KeyboardInterrupt:
        print("\n[INFO] Execução interrompida pelo utilizador.")

if __name__ == "__main__":
    main()