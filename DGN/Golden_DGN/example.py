import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.data import DataLoader
from torch_geometric.nn import BatchNorm, global_mean_pool

import numpy as np
import json
import struct


from pna import PNAConvSimple

dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=128, shuffle=False,num_workers = 1)
val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=128, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=128, shuffle=False)

# Compute in-degree histogram over training data.
deg = torch.zeros(10, dtype=torch.long)
for data in dataset[split_idx['train']]:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.node_emb = AtomEncoder(emb_dim=80)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConvSimple(in_channels=80, out_channels=80, aggregators=aggregators,
                                 scalers=scalers, deg=deg, post_layers=1)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(80))
        self.mlp = Sequential(Linear(80, 40), ReLU(), Linear(40, 20), ReLU(), Linear(20, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            k = batch_norm(conv(x, edge_index, edge_attr))
            h = F.relu(k)
            x = h + x  # residual#
            x = F.dropout(x, 0.3, training=self.training)

        x = global_mean_pool(x, batch)
        return self.mlp(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=3e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, min_lr=0.0001)


def train(epoch):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        #清零梯度
        out = model(data.x, data.edge_index, None, data.batch)
        #前向传播
        loss = torch.nn.BCEWithLogitsLoss()(out.to(torch.float32), data.y.to(torch.float32))
        #计算loss
        loss.backward()
        #反向传播
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        #更新训练参数
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    evaluator = Evaluator(name='ogbg-molhiv')
    list_pred = []
    list_labels = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, None, data.batch)
        list_pred.append(out)
        list_labels.append(data.y)
    epoch_test_ROC = evaluator.eval({'y_pred': torch.cat(list_pred),
                                     'y_true': torch.cat(list_labels)})['rocauc']
    return epoch_test_ROC


if __name__ == '__main__':
    best = (0, 0)
    for epoch in range(1, 10):
        loss = train(epoch)
        val_roc = test(val_loader)
        test_roc = test(test_loader)
        scheduler.step(val_roc)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_roc:.4f}, 'f'Test: {test_roc:.4f}')
        if val_roc > best[0]:
           best = (val_roc, test_roc)
    torch.save(model, 'pna_ep1_dim80.pt')

#val_roc = test(val_loader)
#test_roc = test(test_loader)
#scheduler.step(val_roc)
#print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_roc:.4f}, 'f'Test: {test_roc:.4f}')

    #if val_roc > best[0]:
        #best = (val_roc, test_roc)

#print(f'Best epoch val: {best[0]:.4f}, test: {best[1]:.4f}')
#print model's state_dict


#print('Model.state_dict:')
#for param_tensor in model.state_dict():
    #打印 key value字典
    #print(param_tensor,'\t',model.state_dict()[param_tensor].size())
# all embeddings

#
# nd_emb_0 = model.state_dict()['node_emb.atom_embedding_list.0.weight']
# nd_emb_1 = model.state_dict()['node_emb.atom_embedding_list.1.weight']
# nd_emb_2 = model.state_dict()['node_emb.atom_embedding_list.2.weight']
# nd_emb_3 = model.state_dict()['node_emb.atom_embedding_list.3.weight']
# nd_emb_4 = model.state_dict()['node_emb.atom_embedding_list.4.weight']
# nd_emb_5 = model.state_dict()['node_emb.atom_embedding_list.5.weight']
# nd_emb_6 = model.state_dict()['node_emb.atom_embedding_list.6.weight']
# nd_emb_7 = model.state_dict()['node_emb.atom_embedding_list.7.weight']
# nd_emb_8 = model.state_dict()['node_emb.atom_embedding_list.8.weight']
# nd_all = torch.cat((nd_emb_0, nd_emb_1, nd_emb_2, nd_emb_3, nd_emb_4, nd_emb_5, nd_emb_6, nd_emb_7, nd_emb_8), dim=0)
#
# data = list(nd_all.view(-1).numpy())
# f = open('PNA_ep1_nd_embed_dim100.bin', 'wb')
# packed = struct.pack('f'*len(data), *data)
# f.write(packed)
# f.close()
#
#
# bn_eps = 0.00001
#
#
#
# conv_weight = model.state_dict()['convs.0.post_nn.0.weight']
# conv_bias = model.state_dict()['convs.0.post_nn.0.bias']
# running_mean = model.state_dict()['batch_norms.0.module.running_mean']
# running_var = model.state_dict()['batch_norms.0.module.running_var']
# bn_weight = model.state_dict()['batch_norms.0.module.weight']
# bn_bias = model.state_dict()['batch_norms.0.module.bias']
#
# conv_weight = conv_weight.t()
# conv_weight = (torch.div(conv_weight, torch.sqrt(running_var + bn_eps)) * bn_weight).t()
# conv_bias = torch.div((conv_bias - running_mean), torch.sqrt(running_var + bn_eps)) * bn_weight + bn_bias
# print(model.state_dict()['convs.0.post_nn.0.weight'])
# print('compare:\n')
# print(conv_weight)
# print('\n')
# print('\n')
# print(model.state_dict()['convs.1.post_nn.0.weight'])
#
#
# conv_weight = model.state_dict()['convs.1.post_nn.0.weight']
# conv_bias = model.state_dict()['convs.1.post_nn.0.bias']
# running_mean = model.state_dict()['batch_norms.0.module.running_mean']
# running_var = model.state_dict()['batch_norms.0.module.running_var']
# bn_weight = model.state_dict()['batch_norms.0.module.weight']
# bn_bias = model.state_dict()['batch_norms.0.module.bias']
#
# conv_weight = conv_weight.t()
# conv_weight = (torch.div(conv_weight, torch.sqrt(running_var + bn_eps)) * bn_weight).t()
# conv_bias = torch.div((conv_bias - running_mean), torch.sqrt(running_var + bn_eps)) * bn_weight + bn_bias
#
#
#
# conv_weight = model.state_dict()['convs.2.post_nn.0.weight']
# conv_bias = model.state_dict()['convs.2.post_nn.0.bias']
# running_mean = model.state_dict()['batch_norms.0.module.running_mean']
# running_var = model.state_dict()['batch_norms.0.module.running_var']
# bn_weight = model.state_dict()['batch_norms.0.module.weight']
# bn_bias = model.state_dict()['batch_norms.0.module.bias']
#
# conv_weight = conv_weight.t()
# conv_weight = (torch.div(conv_weight, torch.sqrt(running_var + bn_eps)) * bn_weight).t()
# conv_bias = torch.div((conv_bias - running_mean), torch.sqrt(running_var + bn_eps)) * bn_weight + bn_bias
#
#
# conv_weight = model.state_dict()['convs.3.post_nn.0.weight']
# conv_bias = model.state_dict()['convs.3.post_nn.0.bias']
# running_mean = model.state_dict()['batch_norms.3.module.running_mean']
# running_var = model.state_dict()['batch_norms.3.module.running_var']
# bn_weight = model.state_dict()['batch_norms.3.module.weight']
# bn_bias = model.state_dict()['batch_norms.3.module.bias']
#
# conv_weight = conv_weight.t()
# conv_weight = (torch.div(conv_weight, torch.sqrt(running_var + bn_eps)) * bn_weight).t()
# conv_bias = torch.div((conv_bias - running_mean), torch.sqrt(running_var + bn_eps)) * bn_weight + bn_bias