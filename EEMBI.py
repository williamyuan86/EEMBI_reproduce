import numpy as np
import pandas as pd
import torch
from torch import nn,optim
from cdt.data import load_dataset
from sklearn.preprocessing import MinMaxScaler,OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
import networkx as nx
import sys
sys.path.append('D:/python_work/EEMBI/sachs')
from MINE_ICA import get_bss
from KnnCMI import cmi
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
pcalg=importr('pcalg')
numpy2ri.activate()
from cdt.metrics import SHD, SHD_CPDAG, precision_recall



def data_processing(data,data_type):
    data=np.array(data)
    if data_type=='continuous':
        scaler=MinMaxScaler()
        data=scaler.fit_transform(data)

    elif data_type=='discrete':
        scaler=OrdinalEncoder()
        data=scaler.fit_transform(data)

    else:
        raise ImportError('Such data type is not involved')
    return data


def get_batch(data,batch_size):
    batch_data=[]
    n=len(data)
    for i in range(n//batch_size):
        batch=data[i*batch_size:(i+1)*batch_size]
        batch_data.append(batch)
    return batch_data

# @jit(forceobj=True)
def IAMB(T,data,k=5,alpha=0.01):
    CMB=[]
    CMB_copy=None
    n_nodes=data.shape[1]
    nodes=[i for i in range(n_nodes)]
    while CMB!=CMB_copy:
        CMB_copy=CMB.copy()
        max_val=0
        y=None
        for x in nodes:
            if x==T:
                continue
            val=cmi([T],[x],CMB,k,data=data)
            if val>=max_val:
                max_val=val
                y=x
        if max_val>=alpha:
            CMB.append(y)
            nodes.remove(y)
    CMB_copy=CMB.copy()
    for x in CMB_copy:
        CMB_x=CMB.copy()
        CMB_x.remove(x)
        if CMB_x==None:
            CMB_x=[]
        val=cmi([T],[x],CMB_x,k,data=data)
        if val<=alpha:
            CMB=CMB_x
    return CMB


def get_MB(data,symmetry=None, k=5,alpha=0.01):
    MB={}
    data=pd.DataFrame(data)
    n_nodes=data.shape[1]
    if symmetry==None:
        for T in range(n_nodes):
            MB[T]=IAMB(T,data,k=k,alpha=alpha)
            # MB[T].append(T)
            print(MB)
    elif symmetry=='add':
        for T in range(n_nodes):
            MB[T]=IAMB(T, data, k=k, alpha=alpha)
            # MB[T].append(T)
            for X in MB[T]:
                if X<T and T not in MB[X]:
                    MB[X].append(T)
            print(MB)
    elif symmetry=='delete':
        for T in range(n_nodes):
            MB[T]=IAMB(T, data, k=k, alpha=alpha)
            # MB[T].append(T)
            print(T,MB[T])
        for T in range(n_nodes):
            MBT_copy=MB[T].copy()
            for X in MBT_copy:
                if T not in MB[X]:
                    MB[T].remove(X)
        print(MB)
    return MB


def get_MBM(MB):
    n_nodes=len(MB)
    MBM=np.zeros((n_nodes,n_nodes))
    for i in range(n_nodes):
        for j in MB[i]:
            if i==j:
                continue
            MBM[i,j]=1
    return MBM


def get_exogenuous(data, method='FastICA'):
    if method=='FastICA':
        n_nodes=data.shape[1]
        ica=FastICA(n_components=n_nodes, random_state=0, whiten='unit-variance', max_iter=1000)
        Exo=ica.fit_transform(data)
        return Exo
    elif method=='Mine_ICA':
        data=torch.Tensor(np.array(data))
        bss=get_bss(data)
        Exo=bss(data)
        Exo=Exo.detach().numpy()
        return Exo
    else:
        raise ValueError('Method does not exist')


def metric(X):
    n_nodes=X.shape[1]
    MI_sum=0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i==j:
                continue
            x_i=X[:,i].reshape(-1,1)
            x_j=X[:,j]
            MI_sum+=mutual_info_regression(x_i, x_j)
    return MI_sum


def XE_match(E,data):
    n_nodes=data.shape[1]
    mutual_info_mat=np.zeros((n_nodes,n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            x_i=data[:,i].reshape(-1,1)
            e_j=E[:,j]
            mi_ij=mutual_info_regression(x_i,e_j)
            mutual_info_mat[i,j]=mi_ij
            if mi_ij==0:
                mutual_info_mat[i,j]=int('inf')
    a,arrangement=linear_sum_assignment(-mutual_info_mat)
    # print(mutual_info_mat)
    # print(mutual_info_mat[a,arrangement].mean())
    E_arrange=E[:,arrangement]
    return E_arrange


def IAMB_E(T, MB, Exo, data, k=5, alpha=0.0):
    CMB=[T]
    CMB_copy=None
    nodes=MB[T]
    e_T=Exo[:,T].reshape(-1,1)
    new_data=pd.DataFrame(np.hstack((data,e_T))[:4000])
    while CMB!=CMB_copy:
        CMB_copy=CMB.copy()
        max_val=0
        y=None
        for x in nodes:
            if x==T:
                continue
            # val=con_ind_test(-1,x,CMB, data=new_data)
            val=cmi([-1],[x], CMB, k=k, data=new_data)
            print(x,val)
            if val>max_val:
                max_val=val
                y=x
        if max_val>alpha:
            CMB.append(y)
            nodes.remove(y)
    for x in CMB:
        CMB_x=CMB.copy()
        CMB_x.remove(x)
        if CMB_x==None:
            CMB_x=[]
        val=cmi([-1],[x],CMB_x,k,data=new_data)
        if val<=alpha:
            CMB=CMB_x
    return CMB


def causal_learning(MB, Exo,data, k=5, alpha1=0.2, alpha2=0):
    n_nodes=data.shape[1]
    graph_pred=np.zeros((n_nodes,n_nodes))
    for T in range(n_nodes):
        e_T=Exo[:,T].reshape(-1,1)
        new_data=pd.DataFrame(np.hstack((data,e_T))[:3000])
        for X in MB[T]:
            if graph_pred[X,T]==1 or graph_pred[T,X]==1:
                continue
            val=cmi([-1],[X], [], k=k, data=new_data)
            val_con=cmi([-1],[X], [T], k=k, data=new_data)
            print(T, X, val, val_con)
            if val<=alpha1 and val_con>alpha2:
                graph_pred[T,X]=1
            if val>alpha1 and val_con<=alpha2:
                graph_pred[X,T]=1
            if val>alpha1 and val_con>alpha2:
                graph_pred[X,T]=1
    graph_pred=np.array(pcalg.dag2cpdag(graph_pred))
    return graph_pred


def MB_intersection(MB, Exo, data, k=5, beta=0.01):
    n_nodes=data.shape[1]
    graph_pred=np.zeros((n_nodes,n_nodes))
    for T in range(n_nodes):
        print(T)
        e_T=Exo[:,T].reshape(-1,1)
        new_data=pd.DataFrame(np.hstack((data,e_T)))
        ECMB_T=[]
        ECMB_T_copy=None
        MB_T=MB[T].copy()
        while ECMB_T!=ECMB_T_copy:
            max_val=0
            y=None
            ECMB_T_copy=ECMB_T.copy()
            for x in MB_T:
                val=cmi([-1], [x], ECMB_T, k, data=new_data)
                if val>max_val:
                    y=x
                    max_val=val
            if max_val>beta:
                ECMB_T.append(y)
                MB_T.remove(y)


        for x in ECMB_T.copy():
            ECMB_T_x=ECMB_T.copy()
            ECMB_T_x.remove(x)
            if ECMB_T_x==None:
                ECMB_T_x=[]
            val=cmi([-1], [x], ECMB_T_x, k, data=new_data)
            if val<=beta:
                ECMB_T=ECMB_T_x


        if T in ECMB_T:
            ECMB_T.remove(T)
        for X in ECMB_T:
            graph_pred[T,X]=1
            graph_pred[X,T]=1
    # graph_pred=np.array(pcalg.dag2cpdag(graph_pred))
    return graph_pred


def Find_Vstructure(graph_PC, data):
    from cdt.causality.graph import PC
    pc=PC()
    data=pd.DataFrame(data)
    graph_pred=pc.orient_directed_graph(data, nx.Graph(graph_PC))
    graph_pred=np.array(nx.adjacency_matrix(graph_pred).todense())
    return graph_pred



def EEMBI(MB, Exo, data, k=5, beta=0.01):
    n_nodes=data.shape[1]
    graph_pred=np.zeros((n_nodes,n_nodes))
    for T in range(n_nodes):
        print(T)
        e_T=Exo[:,T].reshape(-1,1)
        new_data=pd.DataFrame(np.hstack((data,e_T)))
        ECMB_T=[]
        ECMB_T_copy=None
        MB_T=MB[T].copy()
        while ECMB_T!=ECMB_T_copy:
            max_val=0
            y=None
            ECMB_T_copy=ECMB_T.copy()
            for x in MB_T:
                val=cmi([-1], [x], ECMB_T, k, data=new_data)
                if val>max_val:
                    y=x
                    max_val=val
            if max_val>beta:
                ECMB_T.append(y)
                MB_T.remove(y)


        for x in ECMB_T.copy():
            ECMB_T_x=ECMB_T.copy()
            ECMB_T_x.remove(x)
            if ECMB_T_x==None:
                ECMB_T_x=[]
            val=cmi([-1], [x], ECMB_T_x, k, data=new_data)
            if val<=beta:
                ECMB_T=ECMB_T_x


        if T in ECMB_T:
            ECMB_T.remove(T)

        for X in ECMB_T:
            if graph_pred[X,T]==graph_pred[T,X]==0:
                graph_pred[X,T]=1
    graph_pred=np.array(pcalg.dag2cpdag(np.array(graph_pred)))
    return graph_pred



def EEMBI_PC(MB, Exo, data, k=5, beta=0.01):
    graph_PC=MB_intersection(MB, Exo, data, k, beta)
    graph_pred=Find_Vstructure(graph_PC, data)
    return graph_pred

if __name__=='__main__':
    data=data_processing(data, 'continuous')
    true_graph=np.array(pcalg.dag2cpdag(true_graph))
    
    MB=get_MB(data, symmetry='delete', k=5, alpha=0.01)
    Exo=get_exogenuous(data, method='FastICA')
    Exo=XE_match(Exo, data)
    graph_pred=EEMBI(MB, Exo, data, beta=0.05)
    graph_pred=EEMBI_PC(MB, Exo, data, beta=0.05)

    print(SHD(true_graph, graph_pred), precision_recall(true_graph, graph_pred)[0])