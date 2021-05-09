import pandas as pd 
from numpy.random import choice
import logging

def create_unlabeled(ratio):
    
    s = pd.read_csv("/raid/home/yoyowu/DTI_wholepfam/data/ChEMBLE26/interaction/global_step_based_pfam_based_splitting/train.csv")
    s[['uniprot','pfam']] = s['uniprot+pfam'].str.split("|", expand=True)
    m=len(s)
    prob = s['pfam'].value_counts(normalize=True)
    c = s['pfam'].value_counts()
    unipf = list(c.index)
    unipf.reverse()
    pro=choice(unipf,p=prob,size=m*ratio)

    w = pd.read_csv("data/ChEMBLE26/interaction/whole_usable_pfam_based_splitting/train.csv")

    chem = [i for i in w['InChIKey'] if i not in s['InChIKey']]
    che=choice(chem,size=m*ratio)
    unipro=[choice(s[(s.pfam==i)]['uniprot+pfam']) for i in pro]
    logger.info(f'--------------done unipro process-----------')

    assert(len(che)==len(unipro))

    d= {'InChIKey': pd.Series(che),'uniprot+pfam' : pd.Series(unipro)}
    df=pd.DataFrame(d)
    df.to_csv("processed_data.csv")
    return df.head()

logger = logging.getLogger(__name__)
create_unlabeled(ratio=2)