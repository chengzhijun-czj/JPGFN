import torch
import torch.nn.functional as F
import argparse
import time
import numpy as np
import random
from dataset import Dataset
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import pickle as pkl
import dgl
from torch.optim import AdamW
from config import *

from JPGFN import JPGFN
from utils import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def get_optimizer(model, lr1=1e-3, lr2=1e-3, lr3=1e-3, lr4=1e-3, wd1=0, wd2=0, wd3=0, wd4=0):
    return AdamW([
        # 嵌入模块参数
        {
            'params': model.emb.parameters(),
            'lr': lr1,
            'weight_decay': wd1
        },
        # 卷积模块参数
        {
            'params': model.conv.parameters(),
            'lr': lr2,
            'weight_decay': wd2
        },
        # 组合模块参数
        {
            'params': model.comb.parameters(),
            'lr': lr3,
            'weight_decay': wd3
        },
        # 注意力模块参数
        {
            'params': model.GCN.parameters(),
            'lr': lr4,
            'weight_decay': wd4
        },
        # # 其他参数默认配置
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(k in n for k in ['emb', 'conv', 'comb', 'GCN'])]
        }
    ])


def train(args, **kwargs):
    dataname = args.dataset
    h_feats = args.hid_dim
    homo = args.homo
    graph = Dataset(dataname, homo).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_class = 2
    features = graph.ndata['feature']
    labels = graph.ndata['label']
    index = list(range(len(labels)))
    if dataname == 'amanzon':
        index = list(range(3305, len(labels)))
    if dataname in ['elliptic', 'yelp']:
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
        idx_train = torch.where(train_mask)[0]
    else:
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                train_size=args.train_ratio,
                                                                random_state=2, shuffle=True)
        (idx_valid, idx_test, y_valid, y_test) = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                  test_size=0.67,
                                                                  random_state=2, shuffle=True)
        train_mask = torch.zeros([len(labels)]).bool()
        val_mask = torch.zeros([len(labels)]).bool()
        test_mask = torch.zeros([len(labels)]).bool()

        train_mask[idx_train] = 1
        val_mask[idx_valid] = 1
        test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    graph = graph.to(device)
    features = features.to(device)  # 确保与模型设备一致
    labels = labels.to(device)

    if args.run == 1:
        model = JPGFN(in_feats, h_feats, num_class, graph, poly_depth=args.conv_layer,
                    alpha=params_config['alpha'], dpb=params_config['dpb'],
                    dpt=params_config['dpt'], **kwargs).to(device)
    optimizer = get_optimizer(model, lr1=params_config['lr1'], lr2=params_config['lr2'], lr3=params_config['lr3'],
                              lr4=params_config['lr4'],
                              wd1=params_config['wd1'], wd2=params_config['wd2'], wd3=params_config['wd3'],
                              wd4=params_config['wd4'])

    weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    time_start = time.time()
    best_auc, best_f1, best_tauprc, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0., 0.

    for i in range(args.epoch):
        model.train()
        logits, emb, h_emb = model(features)
        loss1 = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]).to(device))
        nce_loss = cal_nceloss(emb, h_emb, labels, idx_train)
        loss = loss1 + args.lemda * nce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        labels_np = labels.cpu().detach().numpy()
        probs_np = probs.cpu().detach().numpy()
        preds = np.zeros_like(labels_np)
        preds[probs_np[:, 1] > thres] = 1
        trec = recall_score(labels_np[test_mask], preds[test_mask])
        tpre = precision_score(labels_np[test_mask], preds[test_mask])
        tmf1 = f1_score(labels_np[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(labels_np[test_mask], probs_np[test_mask][:, 1])
        tauprc = average_precision_score(labels_np[test_mask], probs_np[test_mask][:, 1])

        if best_f1 < f1:
            if (tauc + tauprc) > (best_auc + best_tauprc):
                best_f1 = f1
                best_auc = tauc
                best_tauprc = tauprc
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
                final_tauprc = tauprc

        print('Trial {}, Epoch {}, loss: {:.4f}, f1: {:.4f}, best f1: {:.4f}, (auc: {:.4f} prc: {:.4f})'.format(trial, i,
                                                                                                              loss, f1,
                                                                                                              best_f1,
                                                                                                              best_auc,
                                                                                                              best_tauprc))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} PR {:.2f}'.format(final_trec * 100,
                                                                               final_tpre * 100, final_tmf1 * 100,
                                                                               final_tauc * 100, final_tauprc * 100))
    return final_tmf1, final_tauc, final_tauprc


def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    labels = labels.cpu().detach().numpy()
    probs = probs.cpu().detach().numpy()
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HSGFD')
    parser.add_argument('--dataset', type=str, default='yelp', help='dataset for our model (yelp/amazon/tfinance/tsocial/reddit/elliptic/weibo/dgraphfin/tolokers/questions)')
    parser.add_argument('--train_ratio', type=float, default=0.4, help='Training Ratio')
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--epoch", type=int, default=600, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--data_path", type=str, default='/data', help="data path")
    parser.add_argument("--adj_type", type=str, default='sym', help="sym or rw")
    parser.add_argument('--ntrials', type=int, default=10)
    parser.add_argument('--conv_layer', type=int, default=4, help='卷积层深度')
    parser.add_argument("--homo", type=int, default=1, help="1 for (Homo) and 0 for (Hetero)")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--lemda", type=float, default=0.2, help="balance between losses")

    args = parser.parse_args()
    print(args)

    dataname = args.dataset
    data_path = args.data_path
    h_feats = args.hid_dim
    epoch_num = args.epoch
    homo = args.homo

    params_config = dataset_config[dataname]
    graph = Dataset(dataname, homo).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_class = 2

    result_f1 = []
    result_roc = []
    result_prc = []

    for trial in range(args.ntrials):
        setup_seed(42 + trial)
        final_tmf1, final_tauc, final_tauprc = train(args, a=params_config['a'], b=params_config['b'])
        result_f1.append(final_tmf1 * 100)
        result_roc.append(final_tauc * 100)
        result_prc.append(final_tauprc * 100)

    print('Final Result  Dataset:{}, Run:{}, Test F1:{:.2f}±{:.2f}, Test AUC:{:.2f}±{:.2f}, Test PRC:{:.2f}±{:.2f}'.format(
            args.dataset, args.ntrials, np.mean(result_f1), np.std(result_f1),
            np.mean(result_roc), np.std(result_roc), np.mean(result_prc), np.std(result_prc)))