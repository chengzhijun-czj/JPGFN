from dgl.data import FraudYelpDataset, FraudAmazonDataset, RedditDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
from utils import *
import random
import pickle
from torch_geometric.data import Data


class Dataset:
    def __init__(self, name='tfinance', homo=True):
        self.name = name
        graph = None
        if name == 'tfinance':
            graph, label_dict = load_graphs('./data/tfinance.mat')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

        elif name == 'yelp':
            def load_yelp_data(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                x = data['x'].clone().detach().to(torch.float)
                edge_index = data['edge_index'].clone().detach().to(torch.long)
                y = data['y'].clone().detach().to(torch.long)
                train_mask = data['train_mask'].clone().detach().to(torch.bool)
                val_mask = data['val_mask'].clone().detach().to(torch.bool)
                test_mask = data['test_mask'].clone().detach().to(torch.bool)
                graph_data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask,
                                  test_mask=test_mask)
                return graph_data

            file_path = './data/yelp.dat'
            data = load_yelp_data(file_path)
            print(data)
            graph = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
            graph.ndata['feature'] = data.x
            graph.ndata['label'] = data.y
            graph.ndata['train_mask'] = data.train_mask
            graph.ndata['val_mask'] = data.val_mask
            graph.ndata['test_mask'] = data.test_mask
            if homo:
                graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                # graph = dgl.add_self_loop(graph)
            if homo:
                graph = dgl.add_self_loop(graph)

        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                # graph = dgl.add_self_loop(graph)
            if homo:
                graph = dgl.add_self_loop(graph)

        elif name == 'elliptic':
            def load_yelp_data(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                x = data['x'].clone().detach().to(torch.float)
                edge_index = data['edge_index'].clone().detach().to(torch.long)
                y = data['y'].clone().detach().to(torch.long)
                train_mask = data['train_mask'].clone().detach().to(torch.bool)
                val_mask = data['val_mask'].clone().detach().to(torch.bool)
                test_mask = data['test_mask'].clone().detach().to(torch.bool)
                graph_data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask,
                                  test_mask=test_mask)
                return graph_data
            file_path = './data/elliptic.dat'
            data = load_yelp_data(file_path)
            print(data)
            graph = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
            graph.ndata['feature'] = data.x
            graph.ndata['label'] = data.y
            graph.ndata['train_mask'] = data.train_mask
            graph.ndata['val_mask'] = data.val_mask
            graph.ndata['test_mask'] = data.test_mask
            if homo:
                graph = dgl.to_homogeneous(graph,ndata=['feature', 'label', 'train_mask', 'val_mask','test_mask'])
            if homo:
                graph = dgl.add_self_loop(graph)


        elif name == 'weibo':
            file_path = './data/weibo.pt'
            data = torch.load(file_path)
            graph = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
            graph.ndata['feature'] = data.x
            graph.ndata['label'] = data.y
            if homo:
                graph = dgl.to_homogeneous(graph, ndata=['feature', 'label'])
            if homo:
                graph = dgl.add_self_loop(graph)
            
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph



    
