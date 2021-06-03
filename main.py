import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import numpy as np
import torch
import time
from sklearn import metrics
from tensorboardX import SummaryWriter
import shutil
from tqdm import tqdm 
from torch.nn import functional as F
import logging
import math 
from torch import optim

from models import *
from utils import  *
from ligand_graph_features import *

from transformers import BertTokenizer
from transformers.configuration_albert import AlbertConfig
from transformers.modeling_albert import AlbertForMaskedLM
from transformers.modeling_albert import load_tf_weights_in_albert

#torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

def train(teacher_model,student_model,tokenizer,
            chem_dict,protein_dict,t_optimizer, t_scheduler, s_optimizer,s_scheduler,
              criterion, args):

    fname = ( "TFlogs/"+ str(args.runseed) + "/" + args.filename)

    if os.path.exists(fname):
        shutil.rmtree(fname)
        print("removed the existing file.")
    writer = SummaryWriter(fname)
    
    train_l, dev, test = load_training_data(
      args.chem_path + 'interaction/'
        + args.exp,
        args.debug_ratio)

    train_un = pd.read_csv('unlabeled_data.csv')
    print(f'unlabel train. size : {train_un.shape[0]}')
    
    
    for step in range(0, args.global_step):
        if step % args.eval_at == 0 :
            pbar = tqdm(range(args.eval_at))
            s_losses = AverageMeter()
            t_losses = AverageMeter()
   
        teacher_model.train()
        student_model.train()
            
        x_un = train_un.sample(args.batch_size*args.mu)
        #chem_un, pro_un = get_repr_DTI(x_un, tokenizer, chem_dict, protein_dict,args.protein_descriptor)
        
        x_l = train_l.sample(args.batch_size)
        x = pd.concat([x_l,x_un])
        y_l = torch.LongTensor(x_l['Activity'].values).to(args.device)
        chem, pro = get_repr_DTI(x,tokenizer, chem_dict, protein_dict, args.protein_descriptor)
        #chem_l, pro_l = get_repr_DTI(x_l, tokenizer, chem_dict, protein_dict,args.protein_descriptor)
        #chem_l = chem_l.to(args.device)
        #pro_l = pro_l.to(args.device)
        #chem_un = chem_un.to(args.device)
        #pro_un = pro_un.to(args.device)
        chem = chem.to(args.device)
        pro = pro.to(args.device)
        t_logits = teacher_model(pro,chem)
        t_logits_l = t_logits[:args.batch_size]
        t_logits_un = t_logits[args.batch_size:]
        #t_logits_un = teacher_model(pro_un, chem_un)
        #t_logits_l = teacher_model(pro_l, chem_l)
        del t_logits
        t_loss_l = criterion(t_logits_l, y_l)
        #t_loss_l = criterion(t_logits_l, y_l)
        
        _,hard_pseudo_label = torch.max(t_logits_un.detach(), dim=1)
        #ratio = ((hard_pseudo_label==1).sum(dim=0))/((hard_pseudo_label==0).sum(dim=0))
        #s_logits_l = student_model(pro_l, chem_l)
        #s_logits_un = student_model(pro_un, chem_un)

        s_logits = student_model(pro,chem)
        s_logits_l = s_logits[:args.batch_size]
        s_logits_un = s_logits[args.batch_size:]
        del s_logits
        s_loss_l_old = F.cross_entropy(s_logits_l.detach(), y_l)
        s_loss = criterion(s_logits_un, hard_pseudo_label)

        s_optimizer.zero_grad()
        s_loss.backward()
        s_optimizer.step()
        s_scheduler.step()


 
        #with amp.autocast(enabled=args.amp):   
        #t_logits_un = teacher_model(pro_un, chem_un)
     
        
        with torch.no_grad():
            s_logits_l = student_model(pro,chem)[:args.batch_size]

        s_loss_l_new = F.cross_entropy(s_logits_l.detach(), y_l)

        dot_product = s_loss_l_old - s_loss_l_new

        #_, hard_pseudo_label = torch.max(t_logits_un.detach(), dim=1)
        #ratio = ((hard_pseudo_label==1).sum(dim=0))/((hard_pseudo_label==0).sum(dim=0))
        #t_loss =  dot_product * t_loss_l
        t_loss_un =  dot_product * F.cross_entropy(t_logits_un,hard_pseudo_label) 
        t_loss = t_loss_un + t_loss_l
      #  print(f'if there is NAN in s loss : {t_loss.isnan().any()}')
        t_optimizer.zero_grad()
        t_loss.backward()
        t_optimizer.step()
        t_scheduler.step()
       
        teacher_model.zero_grad()
        student_model.zero_grad()

        t_losses.update(t_loss.item())
        s_losses.update(s_loss.item())
        
        
        pbar.set_description(
            f"S_loss: {s_losses.avg:.4f}. "
            f"T_loss: {t_losses.avg:.4f}. "
          )
        pbar.update()
       
        args.num_eval = step//args.eval_at
        if(step+1) % args.eval_at ==0:
            
            pbar.close()
            
            writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
            writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)

            dev_eval = dev.sample(1000)
            test_eval = test.sample(1000)

            print('dev eval number:', dev_eval.shape[0])
            print('test eval number:', test_eval.shape[0])

            dev_f1, dev_auc, dev_aupr  = evaluate(dev_eval,
                                    args, tokenizer,chem_dict,protein_dict,student_model,
                                    datatype='dev')
            test_f1, test_auc, test_aupr = evaluate(test_eval,
                                    args, tokenizer,chem_dict,protein_dict,student_model,
                                    datatype='test')
   
            writer.add_scalar("dev/1.f1", dev_f1, args.num_eval)
            writer.add_scalar("dev/2.auc",dev_auc, args.num_eval)
            writer.add_scalar("dev/3.aupr",dev_aupr, args.num_eval)
            writer.add_scalar("test/1.f1", test_f1, args.num_eval)
            writer.add_scalar("test/2.auc", test_auc, args.num_eval)
            writer.add_scalar("test/3.aupr", test_aupr,args.num_eval)

    writer.close()

    return



def evaluate_binary_predictions(label, predprobs):
    probs = np.array(predprobs)
    predclass = np.argmax(probs, axis=1)
    f1 = metrics.f1_score(label, predclass, average='weighted')
    fpr, tpr, thresholds = metrics.roc_curve(label, probs[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    prec, reca, thresholds = metrics.precision_recall_curve(label, probs[:, 1], pos_label=1)
    aupr = metrics.auc(reca, prec)
    return (f1, auc, aupr)


def evaluate(data, args, tokenizer, chem_dict, protein_dict, model, datatype='dev'):
    model.eval()

    collected_logits = []
    collected_labels = []
    with torch.no_grad():
        for i in range(int(data.shape[0] / args.batch_size)):

            #x = data.sample(args.batch_size)
            x = data[i * args.batch_size:(i + 1) * args.batch_size]
            y = torch.LongTensor(x['Activity'].values).to(args.device)


            chem, prot = get_repr_DTI(x, tokenizer, chem_dict, protein_dict,
                                                                args.protein_descriptor)
            chem = chem.to(args.device)
            prot = prot.to(args.device)
            
            logits = model(prot, chem)
            logits = logits.detach().cpu()
            y = y.detach().cpu()
            
            collected_logits.append(logits)
            collected_labels.append(y)

        collected_logits = np.concatenate(collected_logits, axis=0)
        collected_labels = np.concatenate(collected_labels, axis=0)

        f1, auc, aupr = evaluate_binary_predictions(collected_labels, collected_logits)

    #print("{}\t{:.5f}\t{:.5f}\t{:.5f}".format(datatype, metric[0], metric[1], metric[2]))
    return f1, auc, aupr


def main():
    parser = argparse.ArgumentParser("MPL_DTI")

    parser.add_argument('--debug_ratio', type=float, default=1.0)
    parser.add_argument('--exp', default='global_step_based_pfam_based_splitting/',help='Path to the train/dev/test dataset.')
    parser.add_argument('--protein_descriptor', type=str, default='DISAE',help='choose from [DISAE, TAPE,ESM ]')
    parser.add_argument('--ALBERT_raw', type=str2bool, nargs='?',const=True, default=False)
    parser.add_argument('--frozen', type=str, default='whole',help='choose from {whole, none,partial}')
    parser.add_argument('--global_step', default=20, type=int, help='Number of training epoches ')
    parser.add_argument('--eval_at', default=10, type=int, help='')
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size")
    parser.add_argument('--lr', type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument('--runseed',type=int, default=42,help='random seed')
    parser.add_argument("--device", type=int, default=2, help="which gpu to use if any (default: 0)")
    parser.add_argument("--filename", type=str, default="0524test", help="output filename")
    parser.add_argument("--chem_path", type=str, default= "data/ChEMBLE26/" )
    parser.add_argument("--protein_dict_path", type=str, default= 'protein/' + 'unipfam2triplet.pkl' )
    parser.add_argument("--amp", action="store_true", default =True, help="use 16-bit (mixed) precision")
    parser.add_argument('--label_smoothing', default=0, type=float, help='label smoothing alpha')
    parser.add_argument('--l2',type=int, default=0.0001, help='weight decay')
    parser.add_argument('--exp_name', type=str, default= 'undefined_exp', help='exp name')
    parser.add_argument('--albertconfig',type=str, default= "data/albertdata/DISAE_plus/albert_config_tiny_google.json",help='albert config')
    parser.add_argument("--albert_pretrained_checkpoint", type=str, default= "data/albertdata/DISAE_plus/model.ckpt-3000000",help='load pretrained albert')
    parser.add_argument("--albertvocab" , type=str, default= "data/albertdata/DISAE_plus/pfam_vocab_triplets.txt") 
    parser.add_argument('--chem_pretrained', type=str, default= 'chemble-alone', help='chemble-alone: load contextPred')
    parser.add_argument('--frozen_list',type=list, default=[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    parser.add_argument('--momentum', default=0.5, type=float, help='SGD Momentum')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov')
    parser.add_argument('--grad_clip', default=0.0, type=float, help='gradient norm clipping')
    parser.add_argument('--mu', default=2, type=int, help='coefficient of unlabeled batch size')
    args = parser.parse_args()

    print(f"show all arguments configuration.....{args}")
    
    
    args.device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    torch.cuda.manual_seed(args.runseed) 

    np.random.seed(args.runseed)
    torch.set_num_threads(8)
   
    logger.info(dict(args._get_kwargs()))


    if args.protein_descriptor == 'DISAE':
        albertconfig = AlbertConfig.from_pretrained(args.albertconfig)
        m = AlbertForMaskedLM(config=albertconfig)
        if args.ALBERT_raw==False:
            m = load_tf_weights_in_albert(m, albertconfig,
                                                        args.albert_pretrained_checkpoint)
        else:
            print('DISAE not pretrained!')
        prot_descriptor = m.albert
        prot_tokenizer = BertTokenizer.from_pretrained(args.albertvocab)

    
    protein_dict = pd.Series(load_pkl(args.chem_path
                                               + args.protein_dict_path))
    chem_dict = pd.Series(
            load_json(args.chem_path
                      + 'chemical/ikey2smiles_ChEMBLE.json'))

    
    teacher_model = DTI_model( chem_pretrained=args.chem_pretrained , protein_descriptor=args.protein_descriptor, 
                frozen=args.frozen, frozen_list=args.frozen_list ,device=args.device ,
                model = prot_descriptor)

    student_model = DTI_model( chem_pretrained=args.chem_pretrained , protein_descriptor=args.protein_descriptor, 
                frozen=args.frozen, frozen_list=args.frozen_list ,device=args.device ,
                model = prot_descriptor)
                

    teacher_model = teacher_model.to(args.device)
    student_model = student_model.to(args.device)

    loss_fn = torch.nn.CrossEntropyLoss()  

    teacher_model.zero_grad()
    student_model.zero_grad()

    t_parameters = list(teacher_model.parameters())
    t_optimizer = torch.optim.Adam(t_parameters, lr=args.lr, weight_decay=args.l2)
    t_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t_optimizer, T_max=10)

    s_parameters = list(student_model.parameters())
    s_optimizer = torch.optim.Adam(s_parameters, lr=args.lr, weight_decay=args.l2)
    s_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(s_optimizer, T_max=10)

    train(teacher_model, student_model, prot_tokenizer,
            chem_dict,protein_dict,t_optimizer,t_scheduler, 
            s_optimizer,s_scheduler, 
             loss_fn, args)
    print('Finished training! ')

if __name__ == '__main__':

    main()