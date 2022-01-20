In order to run the preapare_weights.py, you should install the reqirement cpu environments from the dgn repository https://github.com/Saro00/DGN, I have changed some of the origin code of the net to remove the batchNorm layer, so if you want to recreate a model file with batchNorm layer, you should run the following scripts with the origin file.

```
python -m main_HIV --weight_decay=3e-6 --L=4 --type_net="simple" --hidden_dim=100 --out_dim=100 --residual=True --edge_feat=False --readout=mean --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=False --batch_norm=True --aggregators="mean dir1-dx" --scalers="identity" --dataset HIV --gpu_id 0 --config "configs/molecules_graph_classification_DGN_HIV.json" --epochs=200 --init_lr=0.01 --lr_reduce_factor=0.5 --lr_schedule_patience=20 --min_lr=0.0001
```

 the full state_dicts of the no batchNorm layer model are listed below:

`embedding_h.atom_embedding_list.0.weight 	 torch.Size([119, 100])`
`embedding_h.atom_embedding_list.1.weight 	 torch.Size([4, 100])`
`embedding_h.atom_embedding_list.2.weight 	 torch.Size([12, 100])`
`embedding_h.atom_embedding_list.3.weight 	 torch.Size([12, 100])`
`embedding_h.atom_embedding_list.4.weight 	 torch.Size([10, 100])`
`embedding_h.atom_embedding_list.5.weight 	 torch.Size([6, 100])`
`embedding_h.atom_embedding_list.6.weight 	 torch.Size([6, 100])`
`embedding_h.atom_embedding_list.7.weight 	 torch.Size([2, 100])`
`embedding_h.atom_embedding_list.8.weight 	 torch.Size([2, 100])`
`layers.0.posttrans.fully_connected.0.linear.weight 	 torch.Size([100, 100])`
`layers.0.posttrans.fully_connected.0.linear.bias 	 torch.Size([100])`
`layers.1.posttrans.fully_connected.0.linear.weight 	 torch.Size([100, 100])`
`layers.1.posttrans.fully_connected.0.linear.bias 	 torch.Size([100])`
`layers.2.posttrans.fully_connected.0.linear.weight 	 torch.Size([100, 100])`
`layers.2.posttrans.fully_connected.0.linear.bias 	 torch.Size([100])`
`layers.3.posttrans.fully_connected.0.linear.weight 	 torch.Size([100, 100])`
`layers.3.posttrans.fully_connected.0.linear.bias 	 torch.Size([100])`
`MLP_layer.FC_layers.0.weight 	 torch.Size([50, 100])`
`MLP_layer.FC_layers.0.bias 	 torch.Size([50])`
`MLP_layer.FC_layers.1.weight 	 torch.Size([25, 50])`
`MLP_layer.FC_layers.1.bias 	 torch.Size([25])`
`MLP_layer.FC_layers.2.weight 	 torch.Size([1, 25])`
`MLP_layer.FC_layers.2.bias 	 torch.Size([1])`

