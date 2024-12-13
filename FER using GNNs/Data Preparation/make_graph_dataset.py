import torch
import pandas as pd
import os
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt

landmarks_train_src = "E:\\GNN\\datasets\\landmarks\\train"
landmarks_validation_src = "E:\\GNN\\datasets\\landmarks\\validation"

graph_train_dst = "E:\\GNN\\datasets\\graphs\\train"
graph_validation_dst = "E:\\GNN\\datasets\\graphs\\validation"

edges_path = "E:\\GNN\\datasets\\landmarks\\EDGES.txt"

"""
for category in os.listdir(landmarks_train_src):
    file_path = os.path.join(landmarks_train_src, category)
    
    for file in os.listdir(file_path):
        # print(file)
        filename = file.split('.')
        label = filename[0] + "_label." + filename[-1]
        filename_path = os.path.join(file_path, file)
        label_path = os.path.join(file_path, label)
        # print(filename_path, label_path)
        
        df_file = pd.read_csv(filename_path, delimiter=',', header=None)
        df_file.index += 1
        
        df_label = pd.read_csv(label_path, delimiter=',', header=None)
        df_label.index += 1
        
        df_edge = pd.read_csv(edges_path, delimiter=',', header=None)
        df_edge.index += 1
        
        x = torch.tensor(df_file.to_numpy(), dtype=torch.float)
        edge_idx = torch.tensor(df_edge.to_numpy().transpose(), dtype=torch.int64)
        y = torch.tensor(df_label.to_numpy().reshape(1,), dtype=torch.int64)
        graph = Data(x=x, edge_index=edge_idx, y=y)
        
        torch.save(graph, os.path.join(graph_train_dst, category, f'{filename[0]}.pt'))
        
        # print(df_file.head())
        # print()
        # print(df_label.head())
        
        '''
        print(graph)
        data = graph    
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        
        vis = to_networkx(graph)
        plt.figure(1, figsize=(8, 8))
        nx.draw(vis, cmap=plt.get_cmap('Set3'), node_size=70, linewidths=6)
        plt.show()
        '''
    print(f'finished: {file_path}')
"""

def make_graph(src, dst):
    for idx, category in enumerate(os.listdir(src)):
        file_path = os.path.join(src, category)
        
        for file in os.listdir(file_path):
            if len(file.split('_')) == 3:
                continue
            filename = file.split('.')
            filename_path = os.path.join(file_path, file)
            label_path = os.path.join(file_path, f'label_{idx}.csv')
            print(filename_path, label_path)
            
            df_file = pd.read_csv(filename_path, delimiter=',', header=None)
            df_file.index += 1
            
            df_label = pd.read_csv(label_path, delimiter=',', header=None)
            df_label.index += 1
            
            df_edge = pd.read_csv(edges_path, delimiter=',', header=None)
            df_edge.index += 1
            
            x = torch.tensor(df_file.to_numpy(), dtype=torch.float)
            edge_idx = torch.tensor(df_edge.to_numpy().transpose(), dtype=torch.int64)
            y = torch.tensor(df_label.to_numpy().reshape(1,), dtype=torch.int64)
            graph = Data(x=x, edge_index=edge_idx, y=y)
            
            torch.save(graph, os.path.join(dst, category, f'{filename[0]}.pt'))
            
            # print(df_file.head())
            # print()
            # print(df_label.head())
            
            '''
            print(graph)
            data = graph    
            print(f'Has isolated nodes: {data.has_isolated_nodes()}')
            print(f'Has self-loops: {data.has_self_loops()}')
            print(f'Is undirected: {data.is_undirected()}')
            
            vis = to_networkx(graph)
            plt.figure(1, figsize=(8, 8))
            nx.draw(vis, cmap=plt.get_cmap('Set3'), node_size=70, linewidths=6)
            plt.show()
            '''
            
        print(f'Finished: {file_path}')
        
    
print(f"Training Graph")
make_graph(landmarks_train_src, graph_train_dst)
print(f"training graph finished!!!")

print('='*40)

print(f"Validation Graph")
make_graph(landmarks_validation_src, graph_validation_dst)
print(f"validation graph finished!!!")
