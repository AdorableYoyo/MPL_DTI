import os
import time
import numpy as np
import torch
from sklearn import metrics
from data_tool_box import *
from tensorboardX import SummaryWriter
import shutil



class Trainer_byStep():
    def __init__(self, model=None, tokenizer=None,all_config =None,checkpoint_dir=None):
        # ----------------------------------
        #    hyper-parameter/ config
        # ----------------------------------
        self.checkpoint_dir = checkpoint_dir
        self.opt_config= all_config['opt_config']
        self.admin_config=all_config['admin_config']
        self.all_config=all_config
        # ----------------------------------
        #       model
        # ----------------------------------
        if self.all_config['use_cuda'] and torch.cuda.is_available():
            model = model.to('cuda')
        self.model = model
        self.tokenizer =  tokenizer
        # ----------------------------------
        #       input data
        # ----------------------------------
        self.chem_dict = pd.Series(
            load_json(self.all_config['cwd']
                      + self.admin_config['chemble_path']
                      + 'chemical/ikey2smiles_ChEMBLE.json'))
        if all_config['protein_descriptor'] =='DISAE':
            protein_dict_path  = 'protein/' + 'unipfam2triplet.pkl'

        self.protein_dict = pd.Series(load_pkl(self.all_config['cwd']+ self.admin_config['chemble_path']
                                               + protein_dict_path))


    def train(self):
        # ----------------------------------
        #    input data
        # ----------------------------------
        traindf, devdf, testdf = load_training_data(
            self.all_config['cwd']
            + self.admin_config['chemble_path'] + 'interaction/'
            + self.all_config['exp'],
            self.all_config['debug_ratio'])
        # ----------------------------------
        #    training setup
        # ---------------------------------
        parameters = list(self.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.all_config['lr'], weight_decay=self.opt_config['l2'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        loss_fn = torch.nn.CrossEntropyLoss()

        best_target_AUC =-np.inf
        best_epoch = 0
        loss_train = []
        train_metrics_by_epoch,dev_metrics_by_epoch,test_metrics_by_epoch = [],[],[]
        print("Data\tF1\tAUC\tAUPR")
        # ----------------------------------
        #           training
        # ----------------------------------
        train_f1 = []
        train_auc = []
        train_aupr = []
        dev_f1 = []
        dev_auc = []
        dev_aupr = []
        test_f1 = []
        test_auc = []
        test_aupr = []
        

        tbpath = (
            "TFlogs/0505/")
                
        if os.path.exists(tbpath):
            shutil.rmtree(tbpath)
            print("removed the existing file.")
        writer = SummaryWriter(tbpath)
        stime=time.time()
        for step in range(self.all_config['global_step']):

            self.model.train()

            batch_logits,batch_labels  = core_batch_prediction(traindf,step,self.all_config,self.tokenizer,
                                                               self.chem_dict,self.protein_dict,self.model,
                                                               detach=False)
            loss = loss_fn(batch_logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_train.append(loss.detach().cpu().numpy())

            # ----------------------------------
            #           evaluation
            # ----------------------------------
            if step%self.all_config['eval_at'] ==0:
                print('------------------------global step: ', step)
                # stime_eval = time.time()
                self.model.eval()
                traindf_eval = traindf.sample(frac=0.002) # not evalutate all training data
                devdf_eval = devdf.sample(frac=0.006)
                testsize =min(int(testdf.shape[0]*0.03),2000)
                testdf_eval = testdf.sample(testsize)
                print('train eval number:', traindf_eval.shape[0])
                print('dev eval number:', devdf_eval.shape[0])
                print('test eval number:', testdf_eval.shape[0])
                trainmetrics=evaluate(traindf_eval,
                                      self.all_config, self.tokenizer,self.chem_dict,self.protein_dict,self.model,
                                      datatype='train')
                devmetrics  = evaluate(devdf_eval,
                                       self.all_config, self.tokenizer,self.chem_dict,self.protein_dict,self.model,
                                       datatype='dev')
                testmetrics=evaluate(testdf_eval,
                                     self.all_config, self.tokenizer,self.chem_dict,self.protein_dict,self.model,
                                     datatype='test')


                train_f1.append(trainmetrics[0])
                train_auc.append(trainmetrics[1])
                train_aupr.append(trainmetrics[2])
                dev_f1.append(devmetrics[0])
                dev_auc.append(devmetrics[1])
                dev_aupr.append(devmetrics[2])
                test_f1.append(testmetrics[0])
                test_auc.append(testmetrics[1])
                test_aupr.append(testmetrics[2])
                
                for i,val in enumerate(train_f1):
                    writer.add_scalar("data/train f1", val, step)
                for i,val in enumerate(train_auc):
                    writer.add_scalar("data/train auc", val, step)
                for i,val in enumerate (train_aupr):
                    writer.add_scalar("data/train aupr", val, step)
                for i,val in enumerate(dev_f1):
                    writer.add_scalar("data/dev f1", val, step)
                for i,val in enumerate(dev_auc):
                    writer.add_scalar("data/dev auc", val, step)
                for i,val in enumerate (dev_aupr):
                    writer.add_scalar("data/dev aupr", val, step)

                for i,val in enumerate(test_f1):
                    writer.add_scalar("data/test f1", val, step)
                for i,val in enumerate(test_auc):
                    writer.add_scalar("data/test auc", val, step)
                for i,val in enumerate (test_aupr):
                    writer.add_scalar("data/test aupr", val, step)



                print("")


                writer.close()

                train_metrics_by_epoch.append(trainmetrics)
                dev_metrics_by_epoch.append(devmetrics)
                test_metrics_by_epoch.append(testmetrics)
                np.save(self.checkpoint_dir + 'loss_train.npy', loss_train)
                np.save(self.checkpoint_dir + 'trainmetrics_by_epoch.npy', train_metrics_by_epoch)
                np.save(self.checkpoint_dir + 'devmetrics_by_epoch.npy', dev_metrics_by_epoch)
                np.save(self.checkpoint_dir + 'testmetrics_by_epoch.npy', test_metrics_by_epoch)

                # ----------------------------------
                #           save weights
                # ----------------------------------
                print('time cost of the episode: ', time.time() - stime)
                stime= time.time()
                if testmetrics[1] > best_target_AUC:
                    best_target_AUC=  testmetrics[1]
                    best_epoch = step
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'model.dat'))
        print("Best test AUC {:.6f} at epoch {}".format(best_target_AUC,best_epoch))

def core_batch_prediction(traindf, i, all_config, tokenizer, chem_dict, protein_dict, model,by_epoch=False,detach=True):
    # ----------------------------------
    #           process input
    # ----------------------------------
    if by_epoch:
        batch_data = traindf[i * all_config['batch_size']:(i + 1) * all_config['batch_size']]
    else:
        batch_data = traindf.sample(all_config['batch_size'])

    batch_chem_graphs, batch_protein_tokenized = get_repr_DTI(batch_data, tokenizer, chem_dict, protein_dict,
                                                              all_config['protein_descriptor'])

    if all_config['use_cuda'] and torch.cuda.is_available():
        batch_protein_tokenized = batch_protein_tokenized.to('cuda')
        batch_chem_graphs = batch_chem_graphs.to('cuda')
    # ----------------------------------
    #       get prediction score
    # ----------------------------------
    batch_logits = model(batch_protein_tokenized, batch_chem_graphs)
    # ----------------------------------
    #            loss
    # ----------------------------------
    batch_labels = torch.LongTensor(batch_data['Activity'].values)
    if all_config['use_cuda'] and torch.cuda.is_available():
        batch_labels = batch_labels.to('cuda')
    if detach == True:
        batch_logits = batch_logits.detach().cpu()
        batch_labels = batch_labels.detach().cpu()

    return batch_logits, batch_labels


def evaluate_binary_predictions(label, predprobs):
    probs = np.array(predprobs)
    predclass = np.argmax(probs, axis=1)
    f1 = metrics.f1_score(label, predclass, average='weighted')
    fpr, tpr, thresholds = metrics.roc_curve(label, probs[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    prec, reca, thresholds = metrics.precision_recall_curve(label, probs[:, 1], pos_label=1)
    aupr = metrics.auc(reca, prec)
    return f1, auc, aupr


def evaluate(df, all_config, tokenizer, chem_dict, protein_dict, model, datatype='dev', detach=True):
    collected_logits = []
    collected_labels = []
    for i in range(int(df.shape[0] / all_config['batch_size'])):
        # print(i)
        batch_logits, batch_labels = core_batch_prediction(df, i, all_config, tokenizer, chem_dict,
                                                           protein_dict, model,by_epoch=True,
                                                           detach=True)
        collected_logits.append(batch_logits)
        collected_labels.append(batch_labels)
    collected_labels = np.concatenate(collected_labels, axis=0)
    collected_logits = np.concatenate(collected_logits, axis=0)
    metric = evaluate_binary_predictions(collected_labels, collected_logits)
    print("{}\t{:.5f}\t{:.5f}\t{:.5f}".format(datatype, metric[0], metric[1], metric[2]))
    return metric
