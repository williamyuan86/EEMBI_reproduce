# =========================================================================
#  终极解决方案：在导入任何库之前，强制指定并初始化外部 R 环境
# =========================================================================
import os

# 只保留最核心的环境变量设置
R_HOME_PATH = 'C:/PROGRA~1/R/R-45~1.1'
R_USER_LIB_PATH = 'C:/Users/wyuan/AppData/Local/R/win-library/4.5'

# --- START OF NEW CODE ---
# 直接告诉 cdt Rscript.exe 的精确位置
# 请确保这个路径和您的实际安装路径一致
R_SCRIPT_EXECUTABLE = "C:/Program Files/R/R-4.5.1/bin/Rscript.exe"
os.environ['R_SCRIPT_PATH'] = R_SCRIPT_EXECUTABLE
# --- END OF NEW CODE ---

os.environ['R_HOME'] = R_HOME_PATH
os.environ['R_LIBS_USER'] = R_USER_LIB_PATH
r_bin_path = os.path.join(R_HOME_PATH, 'bin', 'x64')
os.environ['PATH'] = r_bin_path + os.pathsep + os.environ.get('PATH', '')


os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from cdt.data import load_dataset
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
import networkx as nx
import sys
# 这边请注意需要把KnnCMI.py文件和EEMBI.py文件放在同一个文件夹下
# sys.path.append('C:/code/codingai/EEMBI/sachs')
# from MINE_ICA import get_bss
from KnnCMI_cuda import cmi
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import conversion
import matplotlib.pyplot as plt
# 引入 tqdm 库用于显示进度条
from tqdm import tqdm

pcalg = importr('pcalg')
from cdt.metrics import SHD, SHD_CPDAG, precision_recall


def data_processing(data, data_type):
    data = np.array(data)
    if data_type == 'continuous':
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

    elif data_type == 'discrete':
        scaler = OrdinalEncoder()
        data = scaler.fit_transform(data)

    else:
        raise ImportError('Such data type is not involved')
    return data


def get_batch(data, batch_size):
    batch_data = []
    n = len(data)
    for i in range(n // batch_size):
        batch = data[i * batch_size:(i + 1) * batch_size]
        batch_data.append(batch)
    return batch_data


def IAMB(T, data, k=5, alpha=0.01):
    CMB = []
    CMB_copy = None
    n_nodes = data.shape[1]
    nodes = [i for i in range(n_nodes)]
    while CMB != CMB_copy:
        CMB_copy = CMB.copy()
        max_val = 0
        y = None
        for x in nodes:
            if x == T:
                continue
            val = cmi([T], [x], CMB, k, data=data)
            if val >= max_val:
                max_val = val
                y = x
        if max_val >= alpha:
            CMB.append(y)
            nodes.remove(y)
    CMB_copy = CMB.copy()
    for x in CMB_copy:
        CMB_x = CMB.copy()
        CMB_x.remove(x)
        if not CMB_x:  # 检查列表是否为空
            CMB_x = []
        val = cmi([T], [x], CMB_x, k, data=data)
        if val <= alpha:
            CMB = CMB_x
    return CMB


def get_MB(data, symmetry=None, k=5, alpha=0.01):
    MB = {}
    data = pd.DataFrame(data)
    n_nodes = data.shape[1]

    # 使用 tqdm 创建进度条来监控最耗时的循环
    iterator = tqdm(range(n_nodes), desc="Calculating Markov Blankets")

    if symmetry is None:
        for T in iterator:
            MB[T] = IAMB(T, data, k=k, alpha=alpha)
    elif symmetry == 'add':
        for T in iterator:
            MB[T] = IAMB(T, data, k=k, alpha=alpha)
            for X in MB[T]:
                if X < T and T not in MB[X]:
                    MB[X].append(T)
    elif symmetry == 'delete':
        for T in iterator:
            MB[T] = IAMB(T, data, k=k, alpha=alpha)

        # 第二个循环通常很快，可以不加进度条
        for T in range(n_nodes):
            MBT_copy = MB[T].copy()
            for X in MBT_copy:
                if T not in MB[X]:
                    MB[T].remove(X)

    print("\n--- Markov Blanket Calculation Finished ---")
    print("Final Markov Blankets:", MB)
    print("-" * 40)
    return MB


def get_MBM(MB):
    n_nodes = len(MB)
    MBM = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in MB[i]:
            if i == j:
                continue
            MBM[i, j] = 1
    return MBM


def get_exogenuous(data, method='FastICA'):
    if method == 'FastICA':
        n_nodes = data.shape[1]
        ica = FastICA(n_components=n_nodes, random_state=0, whiten='unit-variance', max_iter=1000)
        Exo = ica.fit_transform(data)
        return Exo
    else:
        raise ValueError('Method does not exist')


def metric(X):
    n_nodes = X.shape[1]
    MI_sum = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            x_i = X[:, i].reshape(-1, 1)
            x_j = X[:, j]
            MI_sum += mutual_info_regression(x_i, x_j)
    return MI_sum


def XE_match(E, data):
    n_nodes = data.shape[1]
    mutual_info_mat = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            x_i = data[:, i].reshape(-1, 1)
            e_j = E[:, j]
            mi_ij = mutual_info_regression(x_i, e_j)[0]
            mutual_info_mat[i, j] = mi_ij

    # 核心逻辑: 先将 MI 矩阵转换为成本矩阵
    cost_matrix = -mutual_info_mat

    # 修正2: 解决 ValueError
    # 找到原始 MI 接近于 0 的位置。在成本矩阵中，这些位置的值也接近 0。
    # 我们需要把这些位置的成本设置为一个非常大的正数，以确保算法不会选择它们。
    zero_mi_indices = mutual_info_mat < 1e-9
    cost_matrix[zero_mi_indices] = 1e9  # 赋一个很大的正成本

    # 现在 cost_matrix 是完全正确的：
    # - MI 大的地方 -> cost 是大的负数 (会被优先选择)
    # - MI 小的地方 -> cost 是大的正数 (会被避免)

    a, arrangement = linear_sum_assignment(cost_matrix)
    E_arrange = E[:, arrangement]
    return E_arrange


def IAMB_E(T, MB, Exo, data, k=5, alpha=0.0):
    CMB = [T]
    CMB_copy = None
    nodes = MB[T]
    e_T = Exo[:, T].reshape(-1, 1)
    new_data = pd.DataFrame(np.hstack((data, e_T))[:4000])
    while CMB != CMB_copy:
        CMB_copy = CMB.copy()
        max_val = 0
        y = None
        for x in nodes:
            if x == T:
                continue
            val = cmi([-1], [x], CMB, k=k, data=new_data)
            if val > max_val:
                max_val = val
                y = x
        if max_val > alpha:
            CMB.append(y)
            nodes.remove(y)
    for x in CMB:
        CMB_x = CMB.copy()
        CMB_x.remove(x)
        if not CMB_x:
            CMB_x = []
        val = cmi([-1], [x], CMB_x, k, data=new_data)
        if val <= alpha:
            CMB = CMB_x
    return CMB


def causal_learning(MB, Exo, data, k=5, alpha1=0.2, alpha2=0):
    n_nodes = data.shape[1]
    graph_pred = np.zeros((n_nodes, n_nodes))
    iterator = tqdm(range(n_nodes), desc="Running Causal Learning")
    for T in iterator:
        e_T = Exo[:, T].reshape(-1, 1)
        new_data = pd.DataFrame(np.hstack((data, e_T))[:3000])
        for X in MB[T]:
            if graph_pred[X, T] == 1 or graph_pred[T, X] == 1:
                continue
            val = cmi([-1], [X], [], k=k, data=new_data)
            val_con = cmi([-1], [X], [T], k=k, data=new_data)
            if val <= alpha1 and val_con > alpha2:
                graph_pred[T, X] = 1
            if val > alpha1 and val_con <= alpha2:
                graph_pred[X, T] = 1
            if val > alpha1 and val_con > alpha2:
                graph_pred[X, T] = 1
    graph_pred = np.array(pcalg.dag2cpdag(graph_pred))
    return graph_pred


def MB_intersection(MB, Exo, data, k=5, beta=0.01):
    n_nodes = data.shape[1]
    graph_pred = np.zeros((n_nodes, n_nodes))
    iterator = tqdm(range(n_nodes), desc="Running MB Intersection")
    for T in iterator:
        e_T = Exo[:, T].reshape(-1, 1)
        new_data = pd.DataFrame(np.hstack((data, e_T)))
        ECMB_T = []
        ECMB_T_copy = None
        MB_T = MB[T].copy()
        while ECMB_T != ECMB_T_copy:
            max_val = 0
            y = None
            ECMB_T_copy = ECMB_T.copy()
            for x in MB_T:
                val = cmi([-1], [x], ECMB_T, k, data=new_data)
                if val > max_val:
                    y = x
                    max_val = val
            if max_val > beta:
                ECMB_T.append(y)
                MB_T.remove(y)

        for x in ECMB_T.copy():
            ECMB_T_x = ECMB_T.copy()
            ECMB_T_x.remove(x)
            if not ECMB_T_x:
                ECMB_T_x = []
            val = cmi([-1], [x], ECMB_T_x, k, data=new_data)
            if val <= beta:
                ECMB_T = ECMB_T_x

        if T in ECMB_T:
            ECMB_T.remove(T)
        for X in ECMB_T:
            graph_pred[T, X] = 1
            graph_pred[X, T] = 1
    return graph_pred


def Find_Vstructure(graph_PC, data):
    from cdt.causality.graph import PC
    pc = PC()
    data = pd.DataFrame(data)
    graph_pred = pc.orient_directed_graph(data, nx.Graph(graph_PC))
    graph_pred = np.array(nx.adjacency_matrix(graph_pred).todense())
    return graph_pred


def EEMBI(MB, Exo, data, k=5, beta=0.01):
    n_nodes = data.shape[1]
    graph_pred = np.zeros((n_nodes, n_nodes))
    iterator = tqdm(range(n_nodes), desc="Running EEMBI")
    for T in iterator:
        e_T = Exo[:, T].reshape(-1, 1)
        new_data = pd.DataFrame(np.hstack((data, e_T)))
        ECMB_T = []
        ECMB_T_copy = None
        MB_T = MB[T].copy()
        while ECMB_T != ECMB_T_copy:
            max_val = 0
            y = None
            ECMB_T_copy = ECMB_T.copy()
            for x in MB_T:
                val = cmi([-1], [x], ECMB_T, k, data=new_data)
                if val > max_val:
                    y = x
                    max_val = val
            if max_val > beta:
                ECMB_T.append(y)
                MB_T.remove(y)

        for x in ECMB_T.copy():
            ECMB_T_x = ECMB_T.copy()
            ECMB_T_x.remove(x)
            if not ECMB_T_x:
                ECMB_T_x = []
            val = cmi([-1], [x], ECMB_T_x, k, data=new_data)
            if val <= beta:
                ECMB_T = ECMB_T_x

        if T in ECMB_T:
            ECMB_T.remove(T)

        for X in ECMB_T:
            if graph_pred[X, T] == 0 and graph_pred[T, X] == 0:
                graph_pred[X, T] = 1
    graph_pred = np.array(pcalg.dag2cpdag(np.array(graph_pred)))
    return graph_pred


def EEMBI_PC(MB, Exo, data, k=5, beta=0.01):
    print("\n[EEMBI-PC] Starting MB Intersection...")
    graph_PC = MB_intersection(MB, Exo, data, k, beta)
    print("[EEMBI-PC] Finished MB Intersection. Now finding V-structures...")
    graph_pred = Find_Vstructure(graph_PC, data)
    print("[EEMBI-PC] Finished finding V-structures.")
    return graph_pred


def visualize_graph(graph, title, ax):
    """绘制图形"""
    G = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=800, node_color='skyblue',
            font_size=10, font_weight='bold', arrows=True, ax=ax)
    ax.set_title(title)


if __name__ == '__main__':
    with conversion.localconverter(numpy2ri.converter) as cv:
        print("--- Starting script ---")

        print("\n[Step 1/7] Loading Sachs dataset...")
        data, true_graph_original = load_dataset('sachs')
        print("--- Finished loading dataset ---")

        print("\n[Step 2/7] Preprocessing data...")
        data_processed = data_processing(data.values, 'continuous')
        data_df = pd.DataFrame(data_processed, columns=data.columns)
        print("--- Finished preprocessing data ---")

        print("\n[Step 3/7] Preparing true graph...")
        true_graph = nx.to_numpy_array(true_graph_original)
        true_graph_cpdag = np.array(pcalg.dag2cpdag(true_graph))
        print("--- Finished preparing true graph ---")

        print("\n[Step 4/7] Starting Markov Blanket discovery (this may take a while)...")
        MB = get_MB(data_df, symmetry='delete', k=5, alpha=0.01)
        # 结果已在函数内部打印

        print("\n[Step 5/7] Extracting exogenous variables using FastICA...")
        Exo = get_exogenuous(data_processed, method='FastICA')
        Exo = XE_match(Exo, data_processed)
        print("--- Finished extracting exogenous variables ---")

        print("\n[Step 6/7] Running EEMBI and EEMBI-PC algorithms...")
        graph_pred_eembi = EEMBI(MB, Exo, data_processed, beta=0.05)
        graph_pred_eembi_pc = EEMBI_PC(MB, Exo, data_processed, beta=0.05)
        print("--- Finished running algorithms ---")

        print("\n[Step 7/7] Evaluating and printing results...")
        shd_eembi = SHD(true_graph_cpdag, graph_pred_eembi)
        pr_eembi = precision_recall(true_graph_cpdag, graph_pred_eembi)[0]
        print(f"EEMBI - SHD: {shd_eembi}, Precision-Recall: {pr_eembi}")

        shd_eembi_pc = SHD(true_graph_cpdag, graph_pred_eembi_pc)
        pr_eembi_pc = precision_recall(true_graph_cpdag, graph_pred_eembi_pc)[0]
        print(f"EEMBI-PC - SHD: {shd_eembi_pc}, Precision-Recall: {pr_eembi_pc}")
        print("--- Finished evaluation ---")

        print("\nVisualizing graphs...")
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        visualize_graph(true_graph, "True Graph", axes[0])
        visualize_graph(graph_pred_eembi, "Predicted Graph (EEMBI)", axes[1])
        visualize_graph(graph_pred_eembi_pc, "Predicted Graph (EEMBI-PC)", axes[2])

        plt.tight_layout()
        plt.show()
        print("--- Script finished ---")