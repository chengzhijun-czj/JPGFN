import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl.function as fn
import sympy
import scipy
from dgl.nn import GraphConv
from torch_geometric.utils import get_laplacian, degree
from torch_geometric.utils import add_self_loops
from scipy.special import comb
from torch_geometric.nn import MessagePassing
import dgl
from torch import Tensor
import torch.sparse
from typing import Iterable
import math
from dgl.nn.functional import edge_softmax
from torch_geometric.nn import AGNNConv, GCNConv, GATv2Conv, TransformerConv, GATConv, SAGEConv, Node2Vec
from torch_scatter import scatter_mean


class Seq(nn.Module):
    '''
    An extension of nn.Sequential.
    Args:
        modlist an iterable of modules to add.
    '''

    def __init__(self, modlist: Iterable[nn.Module]):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, g):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hid_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hid_dim, in_dim, allow_zero_in_degree=True)
        self.act = nn.ReLU()
        self.g = g

    def forward(self, in_feat):
        g = self.g
        h = self.conv1(g, in_feat)
        h = self.act(h)
        h = self.conv2(g, h)
        return h

class DimensionWiseGCN(nn.Module):
    def __init__(self, dim, g, num_splines=5):
        super().__init__()
        self.att_layers = nn.ModuleList([
            GCN(1, num_splines, g)
            for _ in range(dim)
        ])
        # 组合层：把 dim 个 GCN 结果组合成 out_dim
        self.combine = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [N, dim]
        outs = []
        for i in range(x.size(1)):
            xi = x[:, i:i + 1]  # 取第i维
            out_i = self.att_layers[i](xi)  # 每维做一次 GCN 聚合
            outs.append(out_i)
        h = torch.cat(outs, dim=1)
        return self.combine(h)

class FSTNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_splines=5):
        super().__init__()
        self.functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, num_splines),
                # nn.Tanh(),
                nn.ReLU(),
                nn.Linear(num_splines, out_dim)
            ) for _ in range(in_dim)
        ])
        self.combine = nn.Linear(in_dim * out_dim, out_dim)
    def forward(self, x):
        # x: [batch_size, in_dim]
        outputs = []
        for i in range(x.size(1)):
            xi = x[:, i:i + 1]  # 取第i个特征维度 [batch_size, 1]
            fi = self.functions[i](xi)
            outputs.append(fi)  # 每个特征独立变换
        h = torch.cat(outputs, dim=1)  # [batch_size, in_dim * out_dim]
        h = self.combine(h)
        return h


class JPGFN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, graph, poly_depth=3, alpha=1.5, dpb=0.2, dpt=0.2, sole=False,
                 **kwargs):
        super(JPGFN, self).__init__()
        self.g = graph
        jacobi_a = kwargs.pop('a', -1.0)
        jacobi_b = kwargs.pop('b', 2.0)
        self.conv = JacobiPolyConv(depth=poly_depth, aggr="gcn", alpha=alpha, fixed=False, a=jacobi_a, b=jacobi_b)
        # self.conv = Bern_prop(K=poly_depth)
        self.emb = self._build_embedding_module(in_dim, hid_dim, dpb=dpb, dpt=dpt)
        self.output_layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )
        self.comb = Combination(in_dim, poly_depth + 1, sole=sole)
        self.graph_new = dgl.remove_self_loop(self.g)
        self.GCN = DimensionWiseGCN(in_dim, self.graph_new)

    def _build_embedding_module(self, in_dim, hid_dim, dpb, dpt):
        return nn.Sequential(
            nn.Dropout(p=dpb),
            nn.BatchNorm1d(in_dim),  # 归一化输入
            FSTNN(in_dim, hid_dim),
            nn.Dropout(p=dpt)
        )
    # def _build_embedding_module(self, in_dim, hid_dim, dpb, dpt):
    #     return nn.Sequential(
    #         nn.Dropout(p=dpb),
    #         nn.Linear(in_dim, hid_dim),
    #         nn.ReLU(),
    #         nn.Linear(hid_dim, hid_dim),
    #         nn.Dropout(p=dpt)
    #     )

    def forward(self, in_feat):
        h_emb = self.emb(in_feat)
        edge_index = torch.stack(self.g.edges())
        h_poly = self.conv(h_emb, edge_index, torch.ones(self.g.num_edges(), device=h_emb.device))
        h_combined = self.comb(h_poly)
        h = self.output_layer(h_combined)
        emb = self.GCN(h_emb)
        return h, emb, h_emb


def JacobiConv(L, xs, adj, alphas, a=1.75, b=-0.5, l=-1.0, r=1.0):
    '''
    Jacobi Bases. Please refer to our paper for the form of the bases.
    '''
    if L == 0: return xs[0]
    if L == 1:
        coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
        coef1 *= alphas[0]
        coef2 = (a + b + 2) / (r - l)
        coef2 *= alphas[0]
        return coef1 * xs[-1] + coef2 * (adj @ xs[-1])
    coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
    coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
    coef_lm1_2 = (2 * L + a + b - 1) * (a ** 2 - b ** 2)
    coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
    tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
    tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
    tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)
    tmp1_2 = tmp1 * (2 / (r - l))
    tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
    nx = tmp1_2 * (adj @ xs[-1]) - tmp2_2 * xs[-1]
    nx -= tmp3 * xs[-2]
    return nx


class JacobiPolyConv(nn.Module):
    def __init__(self, depth=3, aggr="gcn", cached=True, alpha=1.0, fixed=False,
                 a=-1.0, b=2.0, l=-1.0, r=1.0):
        super(JacobiPolyConv, self).__init__()
        self.jacobi_params = dict(a=a, b=b, l=l, r=r)

        def wrapped_jacobi(L, xs, adj, alphas):
            return JacobiConv(L, xs, adj, alphas, **self.jacobi_params)
            # return LegendreConv(L, xs, adj, alphas)

        self.poly_conv = PolyConvFrame(conv_fn=wrapped_jacobi,
                                       depth=depth,
                                       aggr=aggr,
                                       cached=cached,
                                       alpha=alpha,
                                       fixed=fixed)

    def forward(self, x, edge_index, edge_attr):
        return self.poly_conv(x, edge_index, edge_attr)


class PolyConvFrame(nn.Module):
    '''
    A framework for polynomial graph signal filter.
    Args:
        conv_fn: the filter function, like PowerConv, LegendreConv,...
        depth (int): the order of polynomial.
        cached (bool): whether or not to cache the adjacency matrix.
        alpha (float):  the parameter to initialize polynomial coefficients.
        fixed (bool): whether or not to fix to polynomial coefficients.
    '''

    def __init__(self,
                 conv_fn,
                 depth: int = 3,
                 aggr: int = "gcn",
                 cached: bool = True,
                 alpha: float = 1.0,
                 fixed: float = False):
        super().__init__()
        self.depth = depth
        self.basealpha = alpha
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(min(1 / alpha, 1))),
                         requires_grad=not fixed) for i in range(depth + 1)
        ])
        self.cached = cached
        self.aggr = aggr
        self.adj = None
        self.conv_fn = conv_fn

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        '''
        Args:
            x: node embeddings. of shape (number of nodes, node feature dimension)
            edge_index and edge_attr: If the adjacency is cached, they will be ignored.
        '''
        if self.adj is None or not self.cached:
            n_node = x.shape[0]
            self.adj = buildAdj(edge_index, edge_attr, x, n_node, self.aggr)
        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas]
        xs = [self.conv_fn(0, [x], self.adj, alphas)]
        for L in range(1, self.depth + 1):
            tx = self.conv_fn(L, xs, self.adj, alphas)
            xs.append(tx)
        xs = [x.unsqueeze(1) for x in xs]
        x = torch.cat(xs, dim=1)
        return x


def buildAdj(edge_index, edge_attr, x, n_node, aggr="gcn"):
    """
    根据边索引和边属性构建邻接矩阵
    Args:
        edge_index: 边的索引，形状为 [2, num_edges]
        edge_attr: 边的权重，形状为 [num_edges]
        n_node: 节点数量
        aggr: 聚合方式，支持 "gcn" 或 "mean"
    Returns:
        稀疏邻接矩阵
    """
    row, col = edge_index
    if aggr == "gcn":
        # GCN归一化: D^-1/2 A D^-1/2
        deg = torch.zeros(n_node, device=edge_index.device)
        deg.scatter_add_(0, row, edge_attr)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        val = deg_inv_sqrt[row] * edge_attr * deg_inv_sqrt[col]

        # 构建稀疏邻接矩阵
    adj = torch.sparse_coo_tensor(
        edge_index,
        val,
        size=(n_node, n_node))

    return adj.coalesce()


class Combination(nn.Module):
    '''
    A mod combination the bases of polynomial filters.
    Args:
        channels (int): number of feature channels.
        depth (int): number of bases to combine.
        sole (bool): whether or not use the same filter for all output channels.
    '''

    def __init__(self, channels: int, depth: int, sole=False):
        super().__init__()
        if sole:
            self.comb_weight = nn.Parameter(torch.empty((1, depth, 1)))
        else:
            self.comb_weight = nn.Parameter(torch.empty((1, depth, channels)))
        nn.init.kaiming_normal_(self.comb_weight, mode='fan_out')
        # nn.init.xavier_normal_(self.comb_weight)

    def forward(self, x: Tensor):
        '''
        x: node features filtered by bases, of shape (number of nodes, depth, channels).
        '''
        weights = torch.softmax(self.comb_weight, dim=1)
        x = x * weights
        x = torch.sum(x, dim=1)
        return x


def ChebyshevConv(L, xs, adj, alphas):
    '''
    Chebyshev Bases. Please refer to our paper for the form of the bases.
    '''
    if L == 0: return xs[0]
    nx = (2 * alphas[L - 1]) * (adj @ xs[-1])
    if L > 1:
        nx -= (alphas[L - 1] * alphas[L - 2]) * xs[-2]
    return nx


def LegendreConv(L, xs, adj, alphas):
    '''
    Legendre bases. Please refer to our paper for the form of the bases.
    '''
    if L == 0: return xs[0]
    nx = (alphas[L - 1] * (2 - 1 / L)) * (adj @ xs[-1])
    if L > 1:
        nx -= (alphas[L - 1] * alphas[L - 2] * (1 - 1 / L)) * xs[-2]
    return nx


class Bern_prop(MessagePassing):
    # Bernstein polynomial filter from the `"BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation" paper.
    # Copied from the official implementation.
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index,
                                           edge_weight,
                                           normalization='sym',
                                           dtype=x.dtype,
                                           num_nodes=x.size(0))
        # 2I-L
        edge_index2, norm2 = add_self_loops(edge_index1,
                                            -norm1,
                                            fill_value=2.,
                                            num_nodes=x.size(0))

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = [(comb(self.K, 0) / (2 ** self.K)) * tmp[self.K]]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)

            out.append((comb(self.K, i + 1) / (2 ** self.K)) * x)
        return torch.stack(out, dim=1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class TensorMod(nn.Module):
    '''
    An mod which forwards a Tensor
    Args:
        x: Tensor
    '''

    def __init__(self, x: Tensor):
        super().__init__()
        self.x = nn.parameter.Parameter(x, requires_grad=False)

    def forward(self, *args, **kwargs):
        return self.x


