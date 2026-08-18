import os
import copy
import time
import pickle
import torch
import warnings
import argparse
import numpy as np
import torch.utils.data as DAta
from utils_pretrained import *
from model_pretrained import *
from sklearn.model_selection import KFold

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

def train_epoch(model, optimizer, loss_func, cached_drug, cached_cline, synergy_graph, train_x, train_y, pe=None):
    model.train()
    optimizer.zero_grad()
    pred = model(None, None, None, None, cached_cline, synergy_graph, train_x[:, 0], train_x[:, 1], train_x[:, 2], pe=pe, drug_embed=cached_drug)
    loss = loss_func(pred, train_y)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, loss_func, cached_drug, cached_cline, synergy_graph, val_x, val_y, pe=None):
    model.eval()
    with torch.no_grad():
        pred = model(None, None, None, None, cached_cline, synergy_graph, val_x[:, 0], val_x[:, 1], val_x[:, 2], pe=pe, drug_embed=cached_drug)
        loss = loss_func(pred, val_y)
        rmse, r2, pr, scc = regression_metric(val_y.cpu().detach().numpy(),
                                             pred.cpu().detach().numpy())
        return [rmse, r2, pr, scc], loss.item(), pred.cpu().detach().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HySynLP: Synergistic Drug Combination Prediction')
    parser.add_argument('--dataset', type=str, default='DrugComb', choices=['DrugComb', 'ALMANAC', 'ONEIL'], help='Dataset name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--epochs', type=int, default=4000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=4e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=6.0e-3, help='L2 regularization')
    parser.add_argument('--pe_dim', type=int, default=0, help='LapPE positional encoding dimension')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    args = parser.parse_args()

    dataset_name = args.dataset
    seed = args.seed
    epochs = args.epochs
    learning_rate = args.lr
    L2 = args.weight_decay
    pe_dim = args.pe_dim
    eval_interval = args.eval_interval
    set_seed_all(seed)
    drug_feature, cline_feature, synergy_data = load_data(dataset_name)
    drug_set = DAta.DataLoader(dataset=GraphDataset(graphs_dict=drug_feature),
                                collate_fn=collate, batch_size=len(drug_feature), shuffle=False)
    cline_set = DAta.DataLoader(dataset=DAta.TensorDataset(cline_feature),
                                batch_size=len(cline_feature), shuffle=False)
    synergy_data = np.array(synergy_data, dtype='float32')
    synergy_cv, test_x, test_y = data_split(synergy_data, rd_seed=seed)

    pretrained_gnn = GCN().to(device)
    pretrained_gnn = load_pre_trained_weights(devices=device, model=pretrained_gnn)
    pretrained_gnn.eval()
    for param in pretrained_gnn.parameters():
        param.requires_grad = False

    # Pre-cache static drug embeddings to eliminate 5-layer GCN forward pass during epochs
    with torch.no_grad():
        for drug in drug_set:
            cached_drug = pretrained_gnn(drug.x.to(device), drug.edge_index.to(device), drug.edge_attr.to(device), drug.batch.to(device)).detach()
        for cline in cline_set:
            cached_cline = cline[0].to(device)

    final_val_metric = np.zeros(4)
    final_test_metric = np.zeros(4)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold = 0
    total_nodes = len(drug_feature) + cline_feature.shape[0]

    for train_index, validation_index in kf.split(synergy_cv):
        fold = fold + 1
        set_seed_all(seed + fold)
        synergy_train, synergy_validation = synergy_cv[train_index], synergy_cv[validation_index]
        train_y = torch.from_numpy(np.array(synergy_train[:, 3], dtype='float32')).to(device)
        val_y = torch.from_numpy(np.array(synergy_validation[:, 3], dtype='float32')).to(device)
        train_x = torch.from_numpy(synergy_train[:, 0:3]).long().to(device)
        val_x = torch.from_numpy(synergy_validation[:, 0:3]).long().to(device)

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

        pe_tensor = compute_hypergraph_pe(synergy_graph, num_nodes=total_nodes, k=pe_dim, device_target=device) if pe_dim > 0 else None

        with open(f'./Datasets/{dataset_name}/subgraph_fold_{fold}.pkl', 'rb') as file:
            edge_adj = pickle.load(file)
        with open(f'./Datasets/{dataset_name}/fold_{fold}.pkl', 'rb') as file:
            all_target_neighbors = pickle.load(file)

        flat_targets, batch_edge_index, root_indices, target_roots = prepare_batched_subgraphs(
            edge_adj, all_target_neighbors, synergy_train=synergy_train, use_triad_closure=True, device_target=device)

        bio_encoder = BioEncoder(pretrained_gnn=pretrained_gnn, dim_cellline=cline_feature.shape[-1], output=512)
        hgnn_encoder = HgnnEncoder(in_channels=512, out_channels=256)
        subgraph_encoder = Subgraph(in_channels=512, hidden=512, out_channels=512, heads=8, concat=True)
        decoder = SymmetricDecoder(in_channels=256, hidden=256, dropout=0.22)

        model = GraphSynergy(
            bio_encoder=bio_encoder,
            graph_encoder=hgnn_encoder,
            subgraph=subgraph_encoder,
            decoder=decoder,
            flat_target_indices=flat_targets,
            batch_edge_index=batch_edge_index,
            root_indices_in_batch=root_indices,
            target_root_ids=target_roots,
            pe_dim=pe_dim
        ).to(device)

        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)

        best_score = -1e9
        best_metric = [0, 0, -1, -1]
        best_epoch = 0
        best_state_dict = None

        print(f"=== Starting Fold {fold}/5 Training on {device} (Dataset: {dataset_name}, Epochs: {epochs}) ===")
        start_time = time.time()

        for epoch in range(epochs):
            train_loss = train_epoch(model, optimizer, loss_func, cached_drug, cached_cline, synergy_graph, train_x, train_y, pe=pe_tensor)
            
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                val_metric, val_loss, _ = evaluate(model, loss_func, cached_drug, cached_cline, synergy_graph, val_x, val_y, pe=pe_tensor)
                current_score = val_metric[1] + val_metric[2] + 1.2 * val_metric[3]

                if current_score > best_score:
                    best_score = current_score
                    best_metric = val_metric
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(model.state_dict())

                if epoch % 500 == 0 or epoch == epochs - 1:
                    print('Fold {:d} | Epoch: {:04d} | Total Loss: {:.6f} | Val RMSE: {:.4f}, R2: {:.4f}, PCC: {:.4f}, SCC: {:.4f}'.format(
                        fold, epoch, train_loss, val_metric[0], val_metric[1], val_metric[2], val_metric[3]))

        elapsed_time = time.time() - start_time
        print(f"=== Fold {fold} Finished in {elapsed_time:.2f}s ===")
        print('Validation Best (Epoch {:04d}) -> RMSE: {:.6f}, R2: {:.6f}, PCC: {:.6f}, SCC: {:.6f}'.format(
            best_epoch, best_metric[0], best_metric[1], best_metric[2], best_metric[3]))

        model.load_state_dict(best_state_dict)
        val_metric, _, _ = evaluate(model, loss_func, cached_drug, cached_cline, synergy_graph, val_x, val_y, pe=pe_tensor)
        test_metric, _, _ = evaluate(model, loss_func, cached_drug, cached_cline, synergy_graph, test_x, test_y, pe=pe_tensor)
        print('Test Evaluation -> RMSE: {:.6f}, R2: {:.6f}, PCC: {:.6f}, SCC: {:.6f}'.format(
            test_metric[0], test_metric[1], test_metric[2], test_metric[3]))

        final_val_metric += np.array(val_metric)
        final_test_metric += np.array(test_metric)

    final_val_metric /= 5
    final_test_metric /= 5
    print('Final 5-CV Validation Average -> RMSE: {:.6f}, R2: {:.6f}, PCC: {:.6f}, SCC: {:.6f}'.format(
        final_val_metric[0], final_val_metric[1], final_val_metric[2], final_val_metric[3]))
    print('Final 5-CV Test Average -> RMSE: {:.6f}, R2: {:.6f}, PCC: {:.6f}, SCC: {:.6f}'.format(
        final_test_metric[0], final_test_metric[1], final_test_metric[2], final_test_metric[3]))