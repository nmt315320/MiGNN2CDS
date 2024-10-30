
import os
import dgl
import time
import torch
import seaborn
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from warnings import simplefilter
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import TensorDataset
# from load_data import load_data, load_graph, remove_graph
#     get_data_loaders,, topk_filtering
from load_data import topk_filtering
from model import Model
from utils import get_metrics, get_metrics_auc, set_seed, \
    plot_result_auc, plot_result_aupr, checkpoint
def load_data(dataset, k):
    # drug_drug = pd.read_csv('./dataset/{}/drug_drug.csv'.format(dataset), header=None)
    # drug_drug_link = topk_filtering(drug_drug.values, k)
    # disease_disease = pd.read_csv('./dataset/{}/disease_disease.csv'.format(dataset), header=None)
    # disease_disease_link = topk_filtering(disease_disease.values, k)
    # drug_disease = pd.read_csv('./dataset/{}/drug_disease.csv'.format(dataset), header=None)
    # drug_disease_link = np.array(np.where(drug_disease == 1)).T
    # disease_drug_link = np.array(np.where(drug_disease.T == 1)).T
    drug_drug = pd.read_csv('./dataset/{}/gene_sim.csv'.format(dataset), header=None)
    drug_drug_link = topk_filtering(drug_drug.values, k)
    disease_disease = pd.read_csv('./dataset/{}/drug_ssim.csv'.format(dataset), header=None)
    disease_disease_link = topk_filtering(disease_disease.values, k)
    drug_disease = pd.read_csv('./dataset/{}/drug_gene.csv'.format(dataset), header=None)
    drug_disease_link = np.array(np.where(drug_disease == 1)).T
    disease_drug_link = np.array(np.where(drug_disease.T == 1)).T
    links = {'drug-drug': drug_drug_link, 'drug-disease': drug_disease_link,
             'disease-disease': disease_disease_link}
    graph_data = {('drug', 'drug-drug', 'drug'): (torch.tensor(drug_drug_link[:, 0]),
                                                  torch.tensor(drug_drug_link[:, 1])),
                  ('drug', 'drug-disease', 'disease'): (torch.tensor(drug_disease_link[:, 0]),
                                                        torch.tensor(drug_disease_link[:, 1])),
                  ('disease', 'disease-drug', 'drug'): (torch.tensor(disease_drug_link[:, 0]),
                                                        torch.tensor(disease_drug_link[:, 1])),
                  ('disease', 'disease-disease', 'disease'): (torch.tensor(disease_disease_link[:, 0]),
                                                              torch.tensor(disease_disease_link[:, 1]))}
    g = dgl.heterograph(graph_data)
    drug_feature = np.hstack((drug_drug.values, np.zeros(drug_disease.shape)))
    dis_feature = np.hstack((np.zeros(drug_disease.T.shape), disease_disease.values))
    g.nodes['drug'].data['h'] = torch.from_numpy(drug_feature).to(torch.float32)
    g.nodes['disease'].data['h'] = torch.from_numpy(dis_feature).to(torch.float32)
    data = np.load('{}_temp_20k/data.npy'.format(dataset, k))
    label = np.load('{}_temp_20k/label.npy'.format(dataset, k))
    if '{}_temp_{}k'.format(dataset, k) in os.listdir():
        print('Load data and label(It takes time)...')

        data = np.load('{}_temp_{}k/data.npy'.format(dataset, k))
        label = np.load('{}_temp_{}k/label.npy'.format(dataset, k))
    print('Data prepared !')
    return g, data, label

#%%

print('Data ')
# Some settings to load the pre-trained model with accurate path
seed = 0
batch_size = 2048
k = 5
nfold = 5
aggregate_type = 'mean'
hidden_feats = 128
learning_rate = 0.001
epoch = 1000
topk = 3
num_layer = 2
dropout = 0.4
batch_norm = False
path = 'result/B-dataset'
# Define the query dataset, drug, and disease
dataset = 'B-dataset'
drug_id = 40
disease_id = 140

#%%

# Load data
g, data, label = load_data(dataset, k)
data = torch.tensor(data)
label = torch.tensor(label)
feature = {'drug': g.nodes['drug'].data['h'],
           'disease': g.nodes['disease'].data['h']}
# Load pre-trained MilGNet model
model_d = []
attns = []

for idx in range(nfold):

    model = Model(g.etypes,
            {'drug'
             : feature['drug'].shape[1],
             'disease': feature['disease'].shape[1]},
                      hidden_feats=hidden_feats,
                      num_emb_layers=num_layer,
                      agg_type=aggregate_type,
                      dropout=dropout,
                      bn=batch_norm,
                      k=topk)

    # model.load_state_dict(
    #     torch.load('result/B-dataset/'
    #                f'k{k}_topk{topk}_nl{num_layer}_ep{epoch}_'
    #                f'lr{learning_rate}_dp{dropout}_bs{batch_size}_'
    #                f'hf{hidden_feats}_{seed}/model_{idx+1}.pkl'
    #               ,map_location='cpu' ))
    model.load_state_dict(
        torch.load('result/B-dataset/saved_421/'
                   f'model_{idx+1}.pkl'
                  ,map_location='cpu' ))

    model_d.append(model)

# Define the data to be used for generating the instance attentions
print((drug_id)*g.num_nodes('disease')+
                 disease_id)
print(data[5])
pred_data = data[(drug_id)*g.num_nodes('disease')+
                 disease_id].unsqueeze(dim=0)

i=0
attns = np.zeros(k*k+2*k+1)

for model in model_d:
    model.eval()

    pred, attn = model(g, feature, pred_data)

    a=attn.detach().numpy().squeeze()
    print(type(attn))
    print(len(attn))
    print(len(attn.detach().numpy().squeeze()))
    print(type(attn.detach().numpy().squeeze()))
    df=pd.DataFrame(a)

    print(type(df))
    df.to_excel('result/B-dataset/saved_421/'
                   f'{i+1}.xlsx',index=False)
    # print(111)
    i =i+1
    # attns += attn.detach().numpy().squeeze()
    # attns=attns/ nfold
    # print(attn.detach().numpy().squeeze()/nfold)
    print(i)

# Visualize the attention coefficient distribution

# seaborn.set(style='white')
# seaborn.histplot(attn.detach().numpy()[0], palette='Set2')
# plt.xlabel('Attention Coefficients')
# plt.legend()
# plt.show()
# Inferences for the topK meta-path instances
topk_idx = np.argsort(attns)[-5:][::-1]
print('Top5 Attentions: {}'.format(attns[topk_idx]))
print('Top5 Meta-path InstanIces:')
metapath = pred_data.numpy().squeeze()
attncoef = attn.detach().numpy()[0]
pred_data.numpy().squeeze()[topk_idx]

#%%

# Ensemble all generated meta-path instances
# and their attention attention coefficients
dr = pd.read_csv(f'dataset/{dataset}/gene_map.csv')
di = pd.read_csv(f'dataset/{dataset}/drug_pubchem_map.csv')
# dr = pd.read_csv(f'dataset/{dataset}/drug.csv')
# di = pd.read_csv(f'dataset/{dataset}/disease.csv')
mpac = []
for i in range(len(metapath)):
    new = []
    for j in range(2):
        new.append(dr.loc[metapath[i, j]].values[1])
    for j in range(2, 4):
        new.append(di.loc[metapath[i, j]].values[1])
    new.append(attncoef[i][0])
    mpac.append(new)
df_mp = pd.DataFrame(np.array(mpac),
                     columns=['CircRNA1',
                              'CircRNA2', 'Drug1',
                              'Drug2', 'Attn'])
df1=pd.DataFrame(df_mp)

df1.to_excel('df_mp.xlsx',index=False)
df_mp['Attn'] = df_mp['Attn'].astype(float)
df_mp = df_mp.sort_values(by='Attn', ascending=False)
df_mp


#%%


