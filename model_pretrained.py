import torch
import torch_sparse
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, global_add_pool, global_mean_pool, global_max_pool, HypergraphConv, MessagePassing
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_atom_type = 119
num_chirality_tag = 3
num_bond_type = 5
num_bond_direction = 3

def gcn_norm(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class GCNConv_pre(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv_pre, self).__init__()
        self.emb_dim = emb_dim
        self.aggr = aggr
        self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))
        self.bias = Parameter(torch.Tensor(emb_dim))
        self.reset_parameters()
        self.edge_embedding1 = nn.Embedding(num_bond_type, 1)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, 1)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        edge_index, __ = gcn_norm(edge_index)
        x = x @ self.weight
        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=None)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j, edge_attr):
        return x_j if edge_attr is None else edge_attr + x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

class GCN(nn.Module):
    def __init__(self, task='classification', num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean'):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = task
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GCNConv_pre(emb_dim, aggr="add"))
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)
        if self.task == 'classification':
            self.pred_head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.Softplus(),
                nn.Linear(self.feat_dim // 2, 2)
            )
        elif self.task == 'regression':
            self.pred_head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.Softplus(),
                nn.Linear(self.feat_dim // 2, 1)
            )

    def forward(self, drug_feature, drug_adj, edge_attr, ibatch):
        x = drug_feature
        edge_index = drug_adj
        edge_attr = edge_attr
        batch = ibatch
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        h = self.pool(h, batch)
        h = self.feat_lin(h)
        return h

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

class LapPEEncoder(nn.Module):
    def __init__(self, in_dim=16, out_dim=512):
        super(LapPEEncoder, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, pe):
        x = self.proj(pe)
        x = self.bn(x)
        return x

class HgnnEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HgnnEncoder, self).__init__()
        self.conv1 = HypergraphConv(in_channels, 256)
        self.batch1 = nn.BatchNorm1d(256)
        self.conv2 = HypergraphConv(256, out_channels)
        self.batch2 = nn.BatchNorm1d(out_channels)
        self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.drop_out = nn.Dropout(0.2)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, edge):
        h1 = self.act(self.batch1(self.conv1(x, edge)))
        h1 = self.drop_out(h1)
        h2 = self.act(self.batch2(self.conv2(h1, edge)))
        res = self.res_proj(x)
        if res.shape[-1] == h2.shape[-1]:
            h2 = h2 + res
        return h2

class BioEncoder(nn.Module):
    def __init__(self, pretrained_gnn, dim_cellline, output, use_GMP=True):
        super(BioEncoder, self).__init__()
        self.use_GMP = use_GMP
        self.druglayer = pretrained_gnn
        self.fc_cell1 = nn.Linear(dim_cellline, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.fc_cell2 = nn.Linear(128, output)
        self.drop_out = nn.Dropout(0.2)
        self.act = nn.ReLU()

    def forward(self, drug_feature, drug_adj, edge_attr, ibatch, gexpr_data, drug_embed=None):
        if drug_embed is not None:
            x_drug = drug_embed
        else:
            x_drug = self.druglayer(drug_feature, drug_adj, edge_attr, ibatch)
        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.drop_out(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug, x_cellline

class Subgraph(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, heads=8, concat=True):
        super(Subgraph, self).__init__()
        self.gat1 = GATv2Conv(in_channels, hidden // heads, heads=heads, concat=True, add_self_loops=True)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.gat2 = GATv2Conv(hidden, out_channels // heads, heads=heads, concat=True, add_self_loops=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.res = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels)
        )
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()
        self.drop_out = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        identity = self.res(x)
        h1 = self.act(self.bn1(self.gat1(x, edge_index)))
        h1 = self.drop_out(h1)
        h2 = self.bn2(self.gat2(h1, edge_index))
        out = self.act(h2 + identity)
        out = self.bn3(out + self.ffn(out))
        return out

class SymmetricDecoder(torch.nn.Module):
    def __init__(self, in_channels=256, hidden=256, dropout=0.2):
        super(SymmetricDecoder, self).__init__()
        # Input features: [d1 + d2, d1 * d2, (d1 + d2) * c, c] -> 4 * in_channels = 1024
        self.fc1 = nn.Linear(in_channels * 4, hidden)
        self.batch1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.batch2 = nn.BatchNorm1d(hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, 1)
        self.drop_out = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, graph_embed, druga_id, drugb_id, cellline_id):
        d1 = graph_embed[druga_id]
        d2 = graph_embed[drugb_id]
        c = graph_embed[cellline_id]
        
        f_sum = d1 + d2
        f_prod = d1 * d2
        f_cell = (d1 + d2) * c
        feat = torch.cat([f_sum, f_prod, f_cell, c], dim=-1)
        
        h = self.act(self.batch1(self.fc1(feat)))
        h = self.drop_out(h)
        h = self.act(self.batch2(self.fc2(h)))
        h = self.drop_out(h)
        out = self.fc3(h).squeeze(dim=1)
        return out

Decoder = SymmetricDecoder

class GraphSynergy(torch.nn.Module):
    def __init__(self, bio_encoder, graph_encoder, subgraph, decoder, flat_target_indices, batch_edge_index, root_indices_in_batch, target_root_ids, pe_dim=0):
        super(GraphSynergy, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.subgraph = subgraph
        self.decoder = decoder
        self.gate = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Sigmoid()
        )
        self.pe_encoder = LapPEEncoder(in_dim=pe_dim, out_dim=512) if pe_dim > 0 else None
        self.register_buffer('flat_target_indices', flat_target_indices)
        self.register_buffer('batch_edge_index', batch_edge_index)
        self.register_buffer('root_indices_in_batch', root_indices_in_batch)
        self.register_buffer('target_root_ids', target_root_ids)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.decoder)
        reset(self.subgraph)
        reset(self.graph_encoder)
        if hasattr(self.gate, 'reset_parameters'):
            self.gate.reset_parameters()
        if self.pe_encoder is not None:
            reset(self.pe_encoder)

    def forward(self, drug_feature, drug_adj, edge_attr, ibatch, gexpr_data, adj, druga_id, drugb_id, cellline_id, pe=None, return_embed=False, drug_embed=None):
        drug_embed, cellline_embed = self.bio_encoder(drug_feature, drug_adj, edge_attr, ibatch, gexpr_data, drug_embed=drug_embed)
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        
        if pe is not None and self.pe_encoder is not None:
            merge_embed = merge_embed + self.pe_encoder(pe)
        
        sub_x = merge_embed[self.flat_target_indices]
        sub_out = self.subgraph(sub_x, self.batch_edge_index)
        active_refined = sub_out[self.root_indices_in_batch]
        active_initial = merge_embed[self.target_root_ids]
        
        cross_gate = self.gate(torch.cat([active_initial, active_refined], dim=-1))
        refined = cross_gate * active_refined + (1.0 - cross_gate) * active_initial
        
        merge_embed = merge_embed.clone()
        merge_embed[self.target_root_ids] = refined
        
        graph_embed = self.graph_encoder(merge_embed, adj)
        res = self.decoder(graph_embed, druga_id, drugb_id, cellline_id)
        if return_embed:
            return res, graph_embed
        return res

def reset(nn_mod):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()
    if nn_mod is not None:
        if hasattr(nn_mod, 'children') and len(list(nn_mod.children())) > 0:
            for item in nn_mod.children():
                _reset(item)
        else:
            _reset(nn_mod)