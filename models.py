import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, ModuleList, Linear, ReLU, BatchNorm1d, Dropout, LogSoftmax

from torch_geometric.utils import to_dense_batch, add_self_loops

from utils import *


from resnet import ResnetEncoderModel

import torch
from torch_geometric.nn import MessagePassing

from functools import partial


num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

num_degree = 11 # suppose from allowable features
num_formal_charge=11
num_hybrid=7
num_aromatic=2


class DTI_model(nn.Module):
    def __init__(self, chem_pretrained, protein_descriptor, 
                frozen, frozen_list,device,
                model, batch_size):
               
        super(DTI_model, self).__init__()
        
        self.protein_descriptor = protein_descriptor
        self.frozen = frozen
       
        self.batch_size = batch_size


        self.ligandEmbedding = GNN(num_layer=5,
                                   emb_dim=300,
                                   JK='last',
                                   drop_ratio=0.5,
                                   gnn_type='gin')
                                   
        if chem_pretrained =='chemble-alone':
            print('------------loading pretrained ContextPred on ChEMBLE alone')
            contextpred_file='data/pretrained_contextpred/chemblFiltered_pretrained_model_with_contextPred.pth'
            self.ligandEmbedding.load_state_dict(torch.load(contextpred_file))
        else:
            print('----------- CotextPred unpretrained')
        #          protein decriptor
        
        self.proteinEmbedding  = model
        if protein_descriptor=='DISAE':
            prot_embed_dim = 256

        if frozen=='partial':
            prot_embed_dim = 256
            ct = 0
            for m in self.proteinEmbedding.modules():
                ct += 1
                if ct in frozen_list:
                    print('frozen module ', ct)
                    for param in m.parameters():
                        param.requires_grad = False
                else:
                    for param in m.parameters():
                        param.requires_grad = True
                # else:
        self.resnet = ResnetEncoderModel(1)
        print('plus Resnet!')

        #        interaction
        self.attentive_interaction_pooler = AttentivePooling(300, )
        self.interaction_pooler = EmbeddingTransform(300 + prot_embed_dim, 128, 64,
                                                     0.1)
        self.binary_predictor = EmbeddingTransform(64, 64, 2, 0.2)

    
        self.attentive_interaction_pooler = self.attentive_interaction_pooler.to(device)
        self.interaction_pooler = self.interaction_pooler.to(device)
        self.binary_predictor = self.binary_predictor.to(device)
        self.ligandEmbedding = self.ligandEmbedding.to(device)
        self.proteinEmbedding = self.proteinEmbedding.to(device)



    def forward(self, batch_protein_tokenized,batch_chem_graphs, **kwargs):
        # ---------------protein embedding ready -------------
        if self.protein_descriptor=='DISAE':
            if self.frozen == 'whole':
                with torch.no_grad():
                    batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]
            else:
                batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]

            batch_protein_repr_resnet = self.resnet(batch_protein_repr.unsqueeze(1)).reshape(self.batch_size,1,-1)#(batch_size,1,256)

        # ---------------ligand embedding ready -------------
        node_representation = self.ligandEmbedding(batch_chem_graphs.x, batch_chem_graphs.edge_index,
                                                   batch_chem_graphs.edge_attr)
        batch_chem_graphs_repr_masked, mask_graph = to_dense_batch(node_representation, batch_chem_graphs.batch)
        batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1).unsqueeze(1)  # (batch_size,1,300)
        # ---------------interaction embedding ready -------------
        ((chem_vector, chem_score), (prot_vector, prot_score)) = self.attentive_interaction_pooler(  batch_chem_graphs_repr_pooled,
                                                                                                     batch_protein_repr_resnet)  # same as input dimension


        interaction_vector = self.interaction_pooler(
            torch.cat((chem_vector.squeeze(), prot_vector.squeeze()), 1))  # (batch_size,64)
        logits = self.binary_predictor(interaction_vector)  # (batch_size,2)
        return logits


class EmbeddingTransform(nn.Module):

    def __init__(self, input_size, hidden_size, out_size,
                 dropout_p=0.1):
        super(EmbeddingTransform, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size),
            nn.BatchNorm1d(out_size)
        )

    def forward(self, embedding):
        embedding = self.dropout(embedding)
        hidden = self.transform(embedding)
        return hidden


class AttentivePooling(nn.Module):
    """ Attentive pooling network according to https://arxiv.org/pdf/1602.03609.pdf """
    def __init__(self, chem_hidden_size=128,prot_hidden_size=256):
        super(AttentivePooling, self).__init__()
        self.chem_hidden_size = chem_hidden_size
        self.prot_hidden_size = prot_hidden_size
        self.param = nn.Parameter(torch.zeros(chem_hidden_size, prot_hidden_size))

    def forward(self, first, second):
        """ Calculate attentive pooling attention weighted representation and
        attention scores for the two inputs.
        Args:
            first: output from one source with size (batch_size, length_1, hidden_size)
            second: outputs from other sources with size (batch_size, length_2, hidden_size)
        Returns:
            (rep_1, attn_1): attention weighted representations and attention scores
            for the first input
            (rep_2, attn_2): attention weighted representations and attention scores
            for the second input
        """
        # logging.debug("AttentivePooling first {0}, second {1}".format(first.size(), second.size()))
        param = self.param.expand(first.size(0), self.chem_hidden_size,self.prot_hidden_size)
        # logging.debug("AttentivePooling params: {0}".format(param.size()))
        wm1 = torch.tanh(torch.bmm(second,param.transpose(1,2)))
        wm2 = torch.tanh(torch.bmm(first,param))
        # logging.debug("Wm1 {}, Wm2 {} before softmax".format(wm1.size(),wm2.size()))
        score_m1 = F.softmax(wm1,dim=2)
        score_m2 = F.softmax(wm2,dim=2)
        # logging.debug("score_m1 {}, score_m2 {}".format(score_m1.size(),score_m2.size()))
        rep_first = first*score_m1
        rep_second = second*score_m2
        # logging.debug("AttentivePooling reps: {0}, {1}".format(rep_first.size(), rep_second.size()))

        return ((rep_first, score_m1), (rep_second, score_m2))

class GINConv(MessagePassing): # from ContextPred
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module): # from Yang
        """


        Args:
            num_layer (int): the number of GNN layers
            emb_dim (int): dimensionality of embeddings
            JK (str): last, concat, max or sum.
            max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
            drop_ratio (float): dropout rate
            gnn_type: gin, gcn, graphsage, gat

        Output:
            node representations

        """

        def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
            super(GNN, self).__init__()
            self.num_layer = num_layer
            self.drop_ratio = drop_ratio
            self.JK = JK

            if self.num_layer < 2:
                raise ValueError("Number of GNN layers must be greater than 1.")

            self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
            self.x_embedding2 = torch.nn.Embedding(num_degree, emb_dim)
            self.x_embedding3 = torch.nn.Embedding(num_formal_charge, emb_dim)
            self.x_embedding4 = torch.nn.Embedding(num_hybrid, emb_dim)
            self.x_embedding5 = torch.nn.Embedding(num_aromatic, emb_dim)
            self.x_embedding6 = torch.nn.Embedding(num_chirality_tag, emb_dim)

            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding4.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding5.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding6.weight.data)

            ###List of MLPs
            self.gnns = torch.nn.ModuleList()
            for layer in range(num_layer):
                if gnn_type == "gin":
                    self.gnns.append(GINConv(emb_dim, aggr="add"))
                elif gnn_type == "gcn":
                    self.gnns.append(GCNConv(emb_dim))
                elif gnn_type == "gat":
                    self.gnns.append(GATConv(emb_dim))
                elif gnn_type == "graphsage":
                    self.gnns.append(GraphSAGEConv(emb_dim))

            ###List of batchnorms
            self.batch_norms = torch.nn.ModuleList()
            for layer in range(num_layer):
                self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        # def forward(self, x, edge_index, edge_attr):
        def forward(self, *argv):
            if len(argv) == 3:
                x, edge_index, edge_attr = argv[0], argv[1], argv[2]
            elif len(argv) == 1:
                data = argv[0]
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            else:
                raise ValueError("unmatched number of arguments.")

            x = self.x_embedding1(x[:, 0].type(torch.long)) + \
                self.x_embedding2(x[:, 1].type(torch.long)) + \
                self.x_embedding3(x[:, 2].type(torch.long)) + \
                self.x_embedding4(x[:, 3].type(torch.long)) + \
                self.x_embedding5(x[:, 4].type(torch.long)) + \
                self.x_embedding6(x[:, 5].type(torch.long))
            #         x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

            h_list = [x]
            for layer in range(self.num_layer):
                h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
                h = self.batch_norms[layer](h)
                # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
                if layer == self.num_layer - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                h_list.append(h)

            ### Different implementations of Jk-concat
            if self.JK == "concat":
                node_representation = torch.cat(h_list, dim=1)
            elif self.JK == "last":
                node_representation = h_list[-1]
            elif self.JK == "max":
                h_list = [h.unsqueeze_(0) for h in h_list]
                node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
            elif self.JK == "sum":
                h_list = [h.unsqueeze_(0) for h in h_list]
                node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

            return node_representation

        def from_pretrained(self, model_file):
            #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
            self.gnn.load_state_dict(torch.load(model_file))




class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None   
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
        

        
class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
    
class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[32, 32, 64, 64], depths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=(5,7), stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=(4,6), stride=(2,3), padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x
    


def ResnetEncoderModel(in_channels,blocks_sizes=[16,32,64,32,16], depths=[2,2,2,2,2], activation='relu'):
    return ResNetEncoder(in_channels,blocks_sizes=blocks_sizes,depths=depths,activation=activation)
