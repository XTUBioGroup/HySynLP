import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from scipy.stats import pearsonr, spearmanr
from torch_scatter import scatter_add
from torch_geometric import data as DATA
from torch_geometric.data import Data
from collections import defaultdict, Counter
from torch_geometric.data import InMemoryDataset, Batch
from sklearn.metrics import mean_squared_error, r2_score
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def load_pre_trained_weights(devices, model):
    try:
        state_dict = torch.load('./pretrained/gcn_model.pth', map_location=devices)
        model.load_my_state_dict(state_dict)
        print("Loaded pre-trained model with success.")
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")
    model.to(devices)
    return model

def build_index(hypergraph):
    nodes, hyperedge_ids = hypergraph
    node_to_hyperedges = defaultdict(set)
    hyperedge_to_nodes = defaultdict(set)
    for i in range(len(nodes)):
        if i % 3 != 2:
            node_to_hyperedges[nodes[i]].add(hyperedge_ids[i])
            hyperedge_to_nodes[hyperedge_ids[i]].add(nodes[i])
    return node_to_hyperedges, hyperedge_to_nodes

def sample_hypergraph_neighborhood(hypergraph, target_nodes):
    node_to_hyperedges, hyperedge_to_nodes = build_index(hypergraph)
    neighborhoods = {target: set() for target in target_nodes}
    for target in target_nodes:
        target_hyperedges = node_to_hyperedges[target]
        for edge_id in target_hyperedges:
            neighbors = hyperedge_to_nodes[edge_id]
            neighborhoods[target].update(neighbors)
        neighborhoods[target].discard(target)
    for target in neighborhoods:
        neighborhoods[target] = list(neighborhoods[target])
    return neighborhoods

def get_subgraph(synergy_graph, num_targets=None):
    nodes = synergy_graph[0]
    drug_nodes = [nodes[i] for i in range(len(nodes)) if i % 3 != 2]
    drug_counter = Counter(drug_nodes)
    if num_targets is not None:
        target_drug_nodes = [drug[0] for drug in drug_counter.most_common(num_targets)]
    else:
        target_drug_nodes = [drug[0] for drug in drug_counter.most_common()]
    sampled_neighborhoods = sample_hypergraph_neighborhood([synergy_graph[0], synergy_graph[1]], target_drug_nodes)
    edge_adj = []
    all_target_neighbors = []
    for target_node, neighbors in sampled_neighborhoods.items():
        all_nodes = [target_node] + neighbors
        all_target_neighbors.append(all_nodes)
        new_index_mapping = {old: new for new, old in enumerate(all_nodes)}
        edge_index_list = []
        for neighbor in neighbors:
            edge_index_list.append([new_index_mapping[target_node], new_index_mapping[neighbor]])
            edge_index_list.append([new_index_mapping[neighbor], new_index_mapping[target_node]])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_adj.append(edge_index)
    return edge_adj, all_target_neighbors

def compute_hypergraph_pe(hg, num_nodes, k=16, device_target=device):
    nodes = hg[0].cpu().numpy()
    edges = hg[1].cpu().numpy()
    num_edges = int(edges.max() + 1)
    H = sp.coo_matrix((np.ones(len(nodes)), (nodes, edges)), shape=(num_nodes, num_edges)).tocsr()
    d_v = np.array(H.sum(axis=1)).flatten()
    d_e = np.array(H.sum(axis=0)).flatten()
    d_v[d_v == 0] = 1.0
    d_e[d_e == 0] = 1.0
    D_v_inv_sqrt = sp.diags(1.0 / np.sqrt(d_v))
    D_e_inv = sp.diags(1.0 / d_e)
    L_sym = sp.eye(num_nodes) - D_v_inv_sqrt @ H @ D_e_inv @ H.T @ D_v_inv_sqrt
    try:
        vals, vecs = spla.eigsh(L_sym, k=k+1, which='SM')
        pe_vecs = vecs[:, 1:k+1]
    except Exception:
        dense_l = L_sym.toarray()
        vals, vecs = np.linalg.eigh(dense_l)
        pe_vecs = vecs[:, 1:k+1]
    pe_tensor = torch.tensor(pe_vecs, dtype=torch.float32, device=device_target)
    return pe_tensor

def differentiable_pcc_loss(pred, target, eps=1e-8):
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    pred_res = pred - pred_mean
    target_res = target - target_mean
    cov = torch.sum(pred_res * target_res)
    var_pred = torch.sum(pred_res ** 2)
    var_target = torch.sum(target_res ** 2)
    pcc = cov / (torch.sqrt(var_pred * var_target) + eps)
    return 1.0 - pcc

def variance_matching_loss(pred, target):
    std_pred = torch.std(pred)
    std_target = torch.std(target)
    return torch.abs(std_pred - std_target)

def margin_ranking_loss_sampled(pred, target, num_samples=2048, margin=0.05):
    n = pred.size(0)
    idx1 = torch.randint(0, n, (num_samples,), device=pred.device)
    idx2 = torch.randint(0, n, (num_samples,), device=pred.device)
    diff_target = target[idx1] - target[idx2]
    diff_pred = pred[idx1] - pred[idx2]
    target_sign = torch.sign(diff_target)
    valid_mask = target_sign != 0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.margin_ranking_loss(diff_pred[valid_mask], torch.zeros_like(diff_pred[valid_mask]), target_sign[valid_mask], margin=margin)

def prepare_batched_subgraphs(sub_edges, targets, synergy_train=None, use_triad_closure=True, device_target=device):
    drug_pairs = set()
    drug_cell_pairs = set()
    if use_triad_closure and synergy_train is not None:
        pos_edge = [t for t in synergy_train if t[3] >= 0]
        for row in pos_edge:
            d1, d2, c = int(row[0]), int(row[1]), int(row[2])
            drug_pairs.add((d1, d2))
            drug_pairs.add((d2, d1))
            drug_cell_pairs.add((d1, c))
            drug_cell_pairs.add((c, d1))
            drug_cell_pairs.add((d2, c))
            drug_cell_pairs.add((c, d2))

    flat_targets = []
    root_indices = []
    target_roots = []
    data_list = []
    current_offset = 0

    for i, target in enumerate(targets):
        t_list = [int(x) for x in target]
        flat_targets.extend(t_list)
        root_indices.append(current_offset)
        target_roots.append(t_list[0])
        current_offset += len(t_list)

        orig_edge = sub_edges[i]
        if use_triad_closure and (len(drug_pairs) > 0 or len(drug_cell_pairs) > 0):
            mapping = {old: new for new, old in enumerate(t_list)}
            existing_edges = set((orig_edge[0, k].item(), orig_edge[1, k].item()) for k in range(orig_edge.size(1)))
            new_edges = list(existing_edges)
            for u_idx, u in enumerate(t_list):
                for v_idx in range(u_idx + 1, len(t_list)):
                    v = t_list[v_idx]
                    if (u, v) in drug_pairs or (u, v) in drug_cell_pairs:
                        if (mapping[u], mapping[v]) not in existing_edges:
                            new_edges.append((mapping[u], mapping[v]))
                            new_edges.append((mapping[v], mapping[u]))
                            existing_edges.add((mapping[u], mapping[v]))
                            existing_edges.add((mapping[v], mapping[u]))
            edge_tensor = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        else:
            edge_tensor = orig_edge

        dummy_x = torch.zeros((len(t_list), 1))
        data_list.append(Data(x=dummy_x, edge_index=edge_tensor))

    batch_data = Batch.from_data_list(data_list)
    return (
        torch.tensor(flat_targets, dtype=torch.long, device=device_target),
        batch_data.edge_index.to(device_target),
        torch.tensor(root_indices, dtype=torch.long, device=device_target),
        torch.tensor(target_roots, dtype=torch.long, device=device_target)
    )

def regression_metric(ytrue, ypred):
    rmse = np.sqrt(mean_squared_error(y_true=ytrue, y_pred=ypred))
    r2 = r2_score(y_true=ytrue, y_pred=ypred)
    r, p = pearsonr(ytrue, ypred)
    scc, _ = spearmanr(ypred, ytrue)
    return rmse, r2, r, scc

def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def creat_hg(synergy_train):
    synergy_train_tmp = np.copy(synergy_train)
    for data in synergy_train_tmp:
        data[3] = 1 if data[3] >= 0 else 0
    pos_edge = np.array([t for t in synergy_train_tmp if t[3] != 0])
    edge_data = pos_edge[:, 0:3]
    synergy_edge = edge_data.reshape(1, -1)
    index_num = np.expand_dims(np.arange(len(edge_data)), axis=-1)
    synergy_num = np.concatenate((index_num, index_num, index_num), axis=1).reshape(1, -1)
    synergy_graph = np.concatenate((synergy_edge, synergy_num), axis=0)
    synergy_graph = torch.from_numpy(synergy_graph).type(torch.LongTensor).to(device)
    return synergy_graph

def drug_feature_extract(drug_data):
    drug_data = pd.DataFrame(drug_data).T
    drug_feat = [[] for _ in range(len(drug_data))]
    for i in range(len(drug_feat)):
        feat_mat, adj_list = drug_data.iloc[i]
        drug_feat[i] = calculate_graph_feat(feat_mat, adj_list)
    return drug_feat

def calculate_graph_feat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]

def data_split(synergy, rd_seed=0):
    synergy_pos = pd.DataFrame([i for i in synergy])
    train_size = 0.9
    synergy_cv_data, synergy_test = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),
                                             [int(train_size * len(synergy_pos))])
    np.random.shuffle(synergy_cv_data)
    np.random.shuffle(synergy_test)
    test_label = torch.from_numpy(np.array(synergy_test[:, 3], dtype='float32')).to(device)
    test_ind = torch.from_numpy(np.array(synergy_test[:, 0:3])).long().to(device)
    return synergy_cv_data, test_ind, test_label

def getData(dataset):
    dataset_dir = os.path.join('./Datasets', dataset)
    if dataset == 'DrugComb':
        drug_smiles_file = os.path.join(dataset_dir, 'drug_smiles.csv')
        cline_feature_file = os.path.join(dataset_dir, 'cell_expression.csv')
        drug_synergy_file = os.path.join(dataset_dir, 'drug_synergy_processed.csv')
        drug = pd.read_csv(drug_smiles_file, sep=',', header=0, index_col=[0])
        smiles_col = 'SMILES'
        gene_data = pd.read_csv(cline_feature_file, sep=',', header=None, index_col=[0])
    elif dataset in ['ALMANAC', 'ONEIL']:
        drug_smiles_file = os.path.join(dataset_dir, 'drug_smiles.csv')
        cline_feature_file = os.path.join(dataset_dir, 'cell line_gene_expression.csv')
        drug_synergy_file = os.path.join(dataset_dir, 'drug_synergy_processed.csv')
        drug = pd.read_csv(drug_smiles_file, sep=',', header=0)
        smiles_col = 'isosmiles'
        gene_data = pd.read_csv(cline_feature_file, sep=',', header=0, index_col=[0])
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    drug_fea = []
    for s in drug[smiles_col]:
        mol = Chem.MolFromSmiles(s)
        mol = Chem.AddHs(mol)
        type_idx = []
        chirality_idx = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        drug_fea.append([x, edge_index, edge_attr])
    cline_fea = np.array(gene_data, dtype='float32')
    synergy = pd.read_csv(drug_synergy_file, sep=',', header=0)
    return cline_fea, drug_fea, synergy

def load_data(dataset):
    cline_fea, drug_fea, synergy = getData(dataset)
    cline_fea = torch.from_numpy(cline_fea).to(device)
    return drug_fea, cline_fea, synergy

class GraphDataset(InMemoryDataset):
    def __init__(self, root='.', dataset='_', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for data_mol in graphs_dict:
            features = data_mol[0].to(device)
            edge_index = data_mol[1].to(device)
            edge_attr = data_mol[2].to(device)
            GCNData = DATA.Data(x=features, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA.to(device)
