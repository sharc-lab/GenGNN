from ogb.graphproppred import Evaluator

import torch

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.HIV_graph_classification.dgn_net import DGNNet
from data.HIV import HIVDataset  # import dataset

# from example import Net

import numpy as np
import json
import struct

from data.HIV import HIVDataset
from nets.HIV_graph_classification.dgn_net import DGNNet


def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_ROC = 0
    with torch.no_grad():
        list_scores = []
        list_labels = []
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_labels = batch_labels.to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            list_scores.append(batch_scores.detach())
            list_labels.append(batch_labels.detach().unsqueeze(-1))

        epoch_test_loss /= (iter + 1)
        evaluator = Evaluator(name='ogbg-molhiv')
        epoch_test_ROC = evaluator.eval({'y_pred': torch.cat(list_scores),
                                           'y_true': torch.cat(list_labels)})['rocauc']

    y_true = torch.cat(list_labels).numpy()
    y_pred = torch.cat(list_scores).numpy()
    return epoch_test_loss, epoch_test_ROC,y_true,y_pred


if __name__ == '__main__':
    ############ Device and test_loader #############
    device = torch.device("cpu")
    DATASET_NAME = 'HIV'
    dataset = HIVDataset(DATASET_NAME, pos_enc_dim=int(0), norm='none')
    testset = dataset.test
    test_loader = DataLoader(testset, batch_size=int(1), shuffle=False, collate_fn=dataset.collate, pin_memory=True)
    gid = 1
    for g in testset.graph_lists:
        key = 'g' + str(gid)
        f_txt = open('eig/' + key + '.txt', 'w+')
        gid += 1
        f_txt.write(str(g.ndata['eig']) + '\n')
        f_txt.close()
    #
    # ########### Load the pre-trained GIN model ###########
    print('Load the pretrained GNN model')
    model = torch.load('dgn_ep1_dim100.pt')
    print('Evaluating...')
    epoch_val_loss, epoch_test_roc,ytrue,ypred = evaluate_network(model, device, test_loader, 2)
    print('test_loss, roC of the original model:')
    print(epoch_val_loss,epoch_test_roc)

    ########### Collect all the golden outputs from Pytorch ###########
    graph_id = 1
    all_result = {}
    f_txt = open('Pytorch_output_dim100.txt', 'w+')
    for yt, yp in zip(ytrue, ypred):
        # print(yt[0], yp[0])
        key = 'g' + str(graph_id)
        all_result[key] = {}
        all_result[key]['true'] = float(yt[0])
        all_result[key]['predict'] = float(yp[0])
        graph_id += 1
        f_txt.write(key + ': ' + str(yp[0]) + '\n')
    f_txt.close()

    ############ This part removes the BatchNorm inside MLP ###############

    f = open('configs/molecules_graph_classification_DGN_HIV.json')
    config = json.load(f)
    print('Removing the BatchNorm in the model')

    print('Model.state_dict:')
    params = config['params']
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    net_params['L'] = 4
    net_params['hidden_dim'] = 100
    net_params['out_dim'] = 100
    net_params['residual'] = True
    net_params['edge_feat'] = False
    net_params['readout'] = 'mean'
    net_params['in_feat_dropout'] = float(0)
    net_params['dropout'] = float(0.3)
    net_params['aggregators'] = 'mean dir1-dx'
    net_params['scalers'] = "identity"
    net_params['towers'] = 5
    D = torch.cat([torch.sparse.sum(g.adjacency_matrix(transpose=True), dim=-1).to_dense() for g in dataset.train.graph_lists])
    net_params['avg_d'] = dict(lin=torch.mean(D),
                                   exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                                   log=torch.mean(torch.log(D + 1)))
    # net_params['divide_input_first'] = config['divide_input_first']
    # net_params['divide_input_last'] = ''
    # net_params['edge_dim'] = args.edge_dim
    # net_params['pretrans_layers'] = args.pretrans_layers
    # net_params['posttrans_layers'] = args.posttrans_layers
    # net_params['type_net'] = args.type_net
    net_params['pos_enc_dim'] = 0
    model_noBN = DGNNet(net_params).to(device)
    for param_tensor in model_noBN.state_dict():
    # 打印 key value字典
        print(param_tensor,'\t',model_noBN.state_dict()[param_tensor].size())
    model_noBN.training = False
    bn_eps = 0.00001

    model_noBN.state_dict()['embedding_h.atom_embedding_list.0.weight'].copy_(
        model.state_dict()['embedding_h.atom_embedding_list.0.weight'])
    model_noBN.state_dict()['embedding_h.atom_embedding_list.1.weight'].copy_(
        model.state_dict()['embedding_h.atom_embedding_list.1.weight'])
    model_noBN.state_dict()['embedding_h.atom_embedding_list.2.weight'].copy_(
        model.state_dict()['embedding_h.atom_embedding_list.2.weight'])
    model_noBN.state_dict()['embedding_h.atom_embedding_list.3.weight'].copy_(
        model.state_dict()['embedding_h.atom_embedding_list.3.weight'])
    model_noBN.state_dict()['embedding_h.atom_embedding_list.4.weight'].copy_(
        model.state_dict()['embedding_h.atom_embedding_list.4.weight'])
    model_noBN.state_dict()['embedding_h.atom_embedding_list.5.weight'].copy_(
        model.state_dict()['embedding_h.atom_embedding_list.5.weight'])
    model_noBN.state_dict()['embedding_h.atom_embedding_list.6.weight'].copy_(
        model.state_dict()['embedding_h.atom_embedding_list.6.weight'])
    model_noBN.state_dict()['embedding_h.atom_embedding_list.7.weight'].copy_(
        model.state_dict()['embedding_h.atom_embedding_list.7.weight'])
    model_noBN.state_dict()['embedding_h.atom_embedding_list.8.weight'].copy_(
        model.state_dict()['embedding_h.atom_embedding_list.8.weight'])
    #print(model_noBN.state_dict()['embedding_h.atom_embedding_list.0.weight'])

    conv_weight = model.state_dict()['layers.0.posttrans.fully_connected.0.linear.weight']
    conv_bias = model.state_dict()['layers.0.posttrans.fully_connected.0.linear.bias']
    running_mean = model.state_dict()['layers.0.batchnorm_h.running_mean']
    running_var = model.state_dict()['layers.0.batchnorm_h.running_var']
    bn_weight = model.state_dict()['layers.0.batchnorm_h.weight']
    bn_bias = model.state_dict()['layers.0.batchnorm_h.bias']

    conv_weight = conv_weight.t()
    conv_weight = (torch.div(conv_weight, torch.sqrt(running_var + bn_eps)) * bn_weight).t()
    conv_bias = torch.div((conv_bias - running_mean), torch.sqrt(running_var + bn_eps)) * bn_weight + bn_bias

    model_noBN.state_dict()['layers.0.posttrans.fully_connected.0.linear.weight'].copy_(conv_weight)
    model_noBN.state_dict()['layers.0.posttrans.fully_connected.0.linear.bias'].copy_(conv_bias)

    conv_weight = model.state_dict()['layers.1.posttrans.fully_connected.0.linear.weight']
    conv_bias = model.state_dict()['layers.1.posttrans.fully_connected.0.linear.bias']
    running_mean = model.state_dict()['layers.1.batchnorm_h.running_mean']
    running_var = model.state_dict()['layers.1.batchnorm_h.running_var']
    bn_weight = model.state_dict()['layers.1.batchnorm_h.weight']
    bn_bias = model.state_dict()['layers.1.batchnorm_h.bias']

    conv_weight = conv_weight.t()
    conv_weight = (torch.div(conv_weight, torch.sqrt(running_var + bn_eps)) * bn_weight).t()
    conv_bias = torch.div((conv_bias - running_mean), torch.sqrt(running_var + bn_eps)) * bn_weight + bn_bias

    model_noBN.state_dict()['layers.1.posttrans.fully_connected.0.linear.weight'].copy_(conv_weight)
    model_noBN.state_dict()['layers.1.posttrans.fully_connected.0.linear.bias'].copy_(conv_bias)

    conv_weight = model.state_dict()['layers.2.posttrans.fully_connected.0.linear.weight']
    conv_bias = model.state_dict()['layers.2.posttrans.fully_connected.0.linear.bias']
    running_mean = model.state_dict()['layers.2.batchnorm_h.running_mean']
    running_var = model.state_dict()['layers.2.batchnorm_h.running_var']
    bn_weight = model.state_dict()['layers.2.batchnorm_h.weight']
    bn_bias = model.state_dict()['layers.2.batchnorm_h.bias']

    conv_weight = conv_weight.t()
    conv_weight = (torch.div(conv_weight, torch.sqrt(running_var + bn_eps)) * bn_weight).t()
    conv_bias = torch.div((conv_bias - running_mean), torch.sqrt(running_var + bn_eps)) * bn_weight + bn_bias

    model_noBN.state_dict()['layers.2.posttrans.fully_connected.0.linear.weight'].copy_(conv_weight)
    model_noBN.state_dict()['layers.2.posttrans.fully_connected.0.linear.bias'].copy_(conv_bias)

    conv_weight = model.state_dict()['layers.3.posttrans.fully_connected.0.linear.weight']
    conv_bias = model.state_dict()['layers.3.posttrans.fully_connected.0.linear.bias']
    running_mean = model.state_dict()['layers.3.batchnorm_h.running_mean']
    running_var = model.state_dict()['layers.3.batchnorm_h.running_var']
    bn_weight = model.state_dict()['layers.3.batchnorm_h.weight']
    bn_bias = model.state_dict()['layers.3.batchnorm_h.bias']

    conv_weight = conv_weight.t()
    conv_weight = (torch.div(conv_weight, torch.sqrt(running_var + bn_eps)) * bn_weight).t()
    conv_bias = torch.div((conv_bias - running_mean), torch.sqrt(running_var + bn_eps)) * bn_weight + bn_bias

    model_noBN.state_dict()['layers.3.posttrans.fully_connected.0.linear.weight'].copy_(conv_weight)
    model_noBN.state_dict()['layers.3.posttrans.fully_connected.0.linear.bias'].copy_(conv_bias)

    model_noBN.state_dict()['MLP_layer.FC_layers.0.weight'].copy_(model.state_dict()['MLP_layer.FC_layers.0.weight'])
    model_noBN.state_dict()['MLP_layer.FC_layers.1.weight'].copy_(model.state_dict()['MLP_layer.FC_layers.1.weight'])
    model_noBN.state_dict()['MLP_layer.FC_layers.2.weight'].copy_(model.state_dict()['MLP_layer.FC_layers.2.weight'])

    model_noBN.state_dict()['MLP_layer.FC_layers.0.bias'].copy_(model.state_dict()['MLP_layer.FC_layers.0.bias'])
    model_noBN.state_dict()['MLP_layer.FC_layers.1.bias'].copy_(model.state_dict()['MLP_layer.FC_layers.1.bias'])
    model_noBN.state_dict()['MLP_layer.FC_layers.2.bias'].copy_(model.state_dict()['MLP_layer.FC_layers.2.bias'])

    print('Evaluating...')
    epoch_val_loss, epoch_test_roc,ytrue,ypred = evaluate_network(model_noBN, device, test_loader, 2)
    print("ROC of the model with no BatchNorm:")
    print(epoch_test_roc)
    for i in model_noBN.named_parameters():
        print(i)
    torch.save(model_noBN, 'dgn_ep1_noBN_dim100.pt')

    ############# Collect all the weights without BatchNorm for golden C ##############

    print("collecting weights for the golden C")

    weights_dict = {}
    weights_data = []
    offset = 0
    for key in model_noBN.state_dict():
        # print(key)
        # print(model_noBN.state_dict()[key].shape)
        # print(model_noBN.state_dict()[key].view(-1).numpy().shape)

        weights_dict[key] = {}
        weights_dict[key]['shape'] = list(model_noBN.state_dict()[key].shape)
        weights_dict[key]['key'] = key
        weights_dict[key]['offset'] = offset
        data = list(model_noBN.state_dict()[key].view(-1).numpy())
        data_length = len(data)
        #print(data)
        weights_dict[key]['length'] = data_length
        offset += data_length
        weights_data += data

    f = open('dgn_ep1_noBN_dim100.weights.dict.json', 'w')
    json.dump(weights_dict, f)
    f.close()

    f = open('dgn_ep1_noBN_dim100.weights.all.bin', 'wb')
    packed = struct.pack('f' * len(weights_data), *weights_data)
    f.write(packed)
    f.close()

    ########### collecting weights for the accelerator #######################

    ### merge all the embedding tables into one
    ### Ahhh this is cumbersome
    nd_emb_0 = model.state_dict()['embedding_h.atom_embedding_list.0.weight']
    nd_emb_1 = model.state_dict()['embedding_h.atom_embedding_list.1.weight']
    nd_emb_2 = model.state_dict()['embedding_h.atom_embedding_list.2.weight']
    nd_emb_3 = model.state_dict()['embedding_h.atom_embedding_list.3.weight']
    nd_emb_4 = model.state_dict()['embedding_h.atom_embedding_list.4.weight']
    nd_emb_5 = model.state_dict()['embedding_h.atom_embedding_list.5.weight']
    nd_emb_6 = model.state_dict()['embedding_h.atom_embedding_list.6.weight']
    nd_emb_7 = model.state_dict()['embedding_h.atom_embedding_list.7.weight']
    nd_emb_8 = model.state_dict()['embedding_h.atom_embedding_list.8.weight']

    nd_all = torch.cat((nd_emb_0, nd_emb_1, nd_emb_2, nd_emb_3, nd_emb_4, nd_emb_5, nd_emb_6, nd_emb_7, nd_emb_8),
                       dim=0)
    # print(nd_all.shape)

    data = list(nd_all.view(-1).numpy())
    f = open('dgn_ep1_nd_embed_dim100.bin', 'wb')
    packed = struct.pack('f' * len(data), *data)
    f.write(packed)
    f.close()

    conv_0_weight = model.state_dict()['layers.0.posttrans.fully_connected.0.linear.weight'].numpy()
    conv_1_weight = model.state_dict()['layers.1.posttrans.fully_connected.0.linear.weight'].numpy()
    conv_2_weight = model.state_dict()['layers.2.posttrans.fully_connected.0.linear.weight'].numpy()
    conv_3_weight = model.state_dict()['layers.3.posttrans.fully_connected.0.linear.weight'].numpy()
    conv_weight_all = torch.tensor([conv_0_weight, conv_1_weight, conv_2_weight, conv_3_weight])
    data = list(conv_weight_all.view(-1).numpy())
    f = open('dgn_conv_weights_dim100.bin', 'wb')
    packed = struct.pack('f' * len(data), *data)
    f.write(packed)
    f.close()

    conv_0_bias = model.state_dict()['layers.0.posttrans.fully_connected.0.linear.bias'].numpy()
    conv_1_bias = model.state_dict()['layers.1.posttrans.fully_connected.0.linear.bias'].numpy()
    conv_2_bias = model.state_dict()['layers.2.posttrans.fully_connected.0.linear.bias'].numpy()
    conv_3_bias = model.state_dict()['layers.3.posttrans.fully_connected.0.linear.bias'].numpy()
    conv_bias_all = torch.tensor([conv_0_bias, conv_1_bias, conv_2_bias, conv_3_bias])
    data = list(conv_bias_all.view(-1).numpy())
    f = open('dgn_conv_bias_dim100.bin', 'wb')
    packed = struct.pack('f' * len(data), *data)
    f.write(packed)
    f.close()

    mlp_0_weight = model.state_dict()['MLP_layer.FC_layers.0.weight'].numpy()
    mlp_2_weight = model.state_dict()['MLP_layer.FC_layers.1.weight'].numpy()
    mlp_4_weight = model.state_dict()['MLP_layer.FC_layers.2.weight'].numpy()

    mlp_0_bias = model.state_dict()['MLP_layer.FC_layers.0.bias'].numpy()
    mlp_2_bias = model.state_dict()['MLP_layer.FC_layers.1.bias'].numpy()
    mlp_4_bias = model.state_dict()['MLP_layer.FC_layers.2.bias'].numpy()


    # mlp_all = torch.tensor([mlp_0_weight,mlp_2_weight,mlp_4_weight,mlp_0_bias,mlp_2_bias,mlp_4_bias])
    # data = list(mlp_all.view(-1).numpy())
    # f = open('dgn_mlp_dim100.bin', 'wb')
    # packed = struct.pack('f'*len(data), *data)
    # f.write(packed)
    # f.close()
