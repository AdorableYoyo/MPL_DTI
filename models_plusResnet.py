import numpy as np
#--------------------------
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, ModuleList, Linear, ReLU, BatchNorm1d, Dropout, LogSoftmax
#--------------------------
from torch_geometric.utils import to_dense_batch
from model_Yang import *
from data_tool_box import *

#--------------------------
from resnet import ResnetEncoderModel
    
class DTI_model(nn.Module):
    def __init__(self, all_config=None,
                 contextpred_config = {
                            'num_layer':5,
                            'emb_dim':300,
                            'JK':'last',
                            'drop_ratio':0.5,
                            'gnn_type':'gin'
                 },
                 model=None):
        super(DTI_model, self).__init__()
        # -------------------------------------------
        #         hyper-parameter
        # -------------------------------------------
        self.use_cuda = all_config['use_cuda']
        self.contextpred_config= contextpred_config
        self.all_config = all_config
        # self.tape_related = tape_related
        # -------------------------------------------
        #         model components
        # -------------------------------------------

       #          chemical decriptor
        self.ligandEmbedding = GNN(num_layer=contextpred_config['num_layer'],
                                   emb_dim=contextpred_config['emb_dim'],
                                   JK=contextpred_config['JK'],
                                   drop_ratio=contextpred_config['drop_ratio'],
                                   gnn_type=contextpred_config['gnn_type'])
        if all_config['chem_pretrained'] =='chemble-alone':
            print('------------loading pretrained ContextPred on ChEMBLE alone')
            contextpred_file='data/pretrained_contextpred/chemblFiltered_pretrained_model_with_contextPred.pth'
            self.ligandEmbedding.load_state_dict(torch.load(all_config['cwd']+contextpred_file))
        else:
            print('----------- CotextPred unpretrained')
        #          protein decriptor
        proteinEmbedding =model
        self.proteinEmbedding  = proteinEmbedding
        if all_config['protein_descriptor']=='DISAE':
            prot_embed_dim = 256

        if all_config['frozen']=='partial':
            prot_embed_dim = 256
            ct = 0
            for m in self.proteinEmbedding.modules():
                ct += 1
                if ct in all_config['DISAE']['frozen_list']:
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
        self.attentive_interaction_pooler = AttentivePooling(contextpred_config['emb_dim'], )
        self.interaction_pooler = EmbeddingTransform(contextpred_config['emb_dim'] + prot_embed_dim, 128, 64,
                                                     0.1)
        self.binary_predictor = EmbeddingTransform(64, 64, 2, 0.2)

        if self.use_cuda and torch.cuda.is_available():
            self.attentive_interaction_pooler = self.attentive_interaction_pooler.to('cuda')
            self.interaction_pooler = self.interaction_pooler.to('cuda')
            self.binary_predictor = self.binary_predictor.to('cuda')
            self.ligandEmbedding = self.ligandEmbedding.to('cuda')
            self.proteinEmbedding = self.proteinEmbedding.to('cuda')



    def forward(self, batch_protein_tokenized,batch_chem_graphs, **kwargs):
        # ---------------protein embedding ready -------------
        if self.all_config['protein_descriptor']=='DISAE':
            if self.all_config['frozen'] == 'whole':
                with torch.no_grad():
                    batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]
            else:
                batch_protein_repr = self.proteinEmbedding(batch_protein_tokenized)[0]

            batch_protein_repr_resnet = self.resnet(batch_protein_repr.unsqueeze(1)).reshape(self.all_config['batch_size'],1,-1)#(batch_size,1,256)

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

