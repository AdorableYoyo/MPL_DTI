"""
DISAE -plus
protein descriptor: ESM vs DISAE-plus
chemical descriptor: contextPred
data: chemble interaction
"""
# ------------- admin
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import numpy as np
import torch
# -------------  my work
from models_plusResnet import *
from trainer_plusResnet_L import *
from utils import  *
from data_tool_box import *

#-------------------------------------------
#      set hyperparameters
#-------------------------------------------

parser = argparse.ArgumentParser("DISAE-plus")
# ---------- args for admin
parser.add_argument('--cwd', type=str, default='',help='current working directory')
parser.add_argument('--debug_ratio', type=float, default=1.0)
parser.add_argument('--exp', default='global_step_based_pfam_based_splitting/',help='Path to the train/dev/test dataset.')
#---------- args for protein descriptor
parser.add_argument('--protein_descriptor', type=str, default='DISAE',help='choose from [DISAE, TAPE,ESM ]')
parser.add_argument('--ALBERT_raw', type=str2bool, nargs='?',const=True, default=False)
parser.add_argument('--frozen', type=str, default='partial',help='choose from {whole, none,partial}')
#---------- args for ContextPred
parser.add_argument('--chem_pretrained',type=str,default='nope',help='chose from {chemble-alone,nope}')
####---------- args for model training and optimization
parser.add_argument('--global_step', default=30, type=int, help='Number of training epoches ')
parser.add_argument('--eval_at', default=10, type=int, help='')
parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
parser.add_argument('--use_cuda',type=str2bool, nargs='?',const=True, default=True, help='use cuda.')
parser.add_argument('--lr', type=float, default=2e-5, help="Initial learning rate")
#----------

opt = parser.parse_args()
# -------------------------------------------
#         set admin
# -------------------------------------------
all_config = load_json(opt.cwd + 'DTI_config.json')
checkpoint_dir = set_up_exp_folder(opt.cwd)

seed = all_config['opt_config']['random_seed']
torch.manual_seed(seed)
if opt.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.set_num_threads(all_config['opt_config']['num_threads'])

all_config.update(vars(opt))

config_file = checkpoint_dir + 'config.json'
save_json(all_config,
          config_file)  # combine config files with the most frequently tuned ones and save again just in case
if opt.use_cuda == False:
    print('not using GPU')

# ------------- protein descriptor
if all_config['protein_descriptor']=='DISAE':
    print('using DISAE+')
    from transformers import BertTokenizer
    from transformers.configuration_albert import AlbertConfig
    from transformers.modeling_albert import AlbertForMaskedLM
    from transformers.modeling_albert import load_tf_weights_in_albert


#-------------------------------------------
#         main
#-------------------------------------------
if __name__ == '__main__':
    print(all_config['exp'])
    # -------------------------------------------
    #      set up DTI models
    # -------------------------------------------
    # Load protein descriptor
    if all_config['protein_descriptor'] == 'DISAE':
        albertconfig = AlbertConfig.from_pretrained(all_config['cwd']+all_config['DISAE']['albertconfig'])
        m = AlbertForMaskedLM(config=albertconfig)
        if all_config['ALBERT_raw']==False:
            m = load_tf_weights_in_albert(m, albertconfig,
                                                        all_config['cwd']+all_config['DISAE']['albert_pretrained_checkpoint'])
        else:
            print('DISAE not pretrained!')
        prot_descriptor = m.albert
        prot_tokenizer = BertTokenizer.from_pretrained(all_config['cwd']+all_config['DISAE']['albertvocab'])



    model = DTI_model( all_config = all_config ,model= prot_descriptor)


    # -------------------------------------------
    #      set up trainer and evaluator
    # -------------------------------------------
    
    trainer = Trainer_byStep(model=model, tokenizer=prot_tokenizer,
                      all_config = all_config,checkpoint_dir=checkpoint_dir )

    # -------------------------------------------
    #      training and evaluating
    # -------------------------------------------

    trainer.train()
    print('Finished training! Experiment log at: ', checkpoint_dir)

