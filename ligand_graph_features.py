from rdkit import Chem
import torch
from torch_geometric.data import Data
import numpy as np

from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import os 


import pickle
from itertools import repeat, chain


from torch.utils import data
import pandas as pd

import networkx as nx

from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from utils import load_json
from torch_geometric.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# allowable node and edge features
# allowable_features = {
#     'possible_atomic_num_list' : list(range(1, 119)),
#     'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
#     'possible_chirality_list' : [
#         Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#         Chem.rdchem.ChiralType.CHI_OTHER
#     ],
#     'possible_hybridization_list' : [
#         Chem.rdchem.HybridizationType.S,
#         Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#         Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
#     ],
#     'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
#     'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
#     'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'possible_bonds' : [
#         Chem.rdchem.BondType.SINGLE,
#         Chem.rdchem.BondType.DOUBLE,
#         Chem.rdchem.BondType.TRIPLE,
#         Chem.rdchem.BondType.AROMATIC
#     ],
#     'possible_bond_dirs' : [ # only for double bond stereo information
#         Chem.rdchem.BondDir.NONE,
#         Chem.rdchem.BondDir.ENDUPRIGHT,
#         Chem.rdchem.BondDir.ENDDOWNRIGHT
#     ]
# }

allowable_features = { # from Yang
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_hybridization_list": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_aromatic_list": [True, False],
    "possible_bond_dirs": [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}



def mol_to_graph_data_obj_simple(mol): # from Yang
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    # num_atom_features = 6  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = (
            [allowable_features["possible_atomic_num_list"].index(atom.GetAtomicNum())]
            + [allowable_features["possible_degree_list"].index(atom.GetDegree())]
            + [allowable_features["possible_formal_charge_list"].index(atom.GetFormalCharge())]
            + [
                allowable_features["possible_hybridization_list"].index(
                    atom.GetHybridization()
                )
            ]
            + [allowable_features["possible_aromatic_list"].index(atom.GetIsAromatic())]
            + [allowable_features["possible_chirality_list"].index(atom.GetChiralTag())]
        )
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features["possible_bonds"].index(bond.GetBondType())
            ] + [allowable_features["possible_bond_dirs"].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        # data = None,
        # slices = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset="chembl26",
        empty=False,
    ):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        self.transform, self.pre_transform, self.pre_filter = (
            transform,
            pre_transform,
            pre_filter,
        )

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        raise NotImplementedError(
            "Must indicate valid location of raw data. " "No download allowed"
        )

    def process(self):
   
        data_list = []

        if self.dataset == "chembl26":
            chem_dict = pd.Series(load_json("data/ChEMBLE26/chemical/ikey2smiles_ChEMBLE.json"))
            input_path = self.raw_paths[0]
            input_df = pd.read_csv(input_path)
            smiles_list = chem_dict[input_df['InChIKey'].values.tolist()].values.tolist()
         
            for i in tqdm(range(len(smiles_list))):
                #                 print(i, end="\r")
                s = smiles_list[i]
                # each example contains a single species
                try:
                    rdkit_mol = AllChem.MolFromSmiles(s)
                    if rdkit_mol is not None:  # ignore invalid mol objects
                        # # convert aromatic bonds to double bonds
                        # Chem.SanitizeMol(rdkit_mol,
                        #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                        data = mol_to_graph_data_obj_simple(rdkit_mol)
                        # manually add mol id
                        id = i
                        data.id = torch.tensor(
                            [id]
                        )  # id here is zinc id value, stripped of
                        # leading zeros
                        data_list.append(data)
      
                except AttributeError:
                    continue

        else:
            raise ValueError("Invalid dataset name")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_repr_DTI(batch_data,tokenizer,chem_dict,protein_dict,prot_descriptor_choice):
    #  . . . .  chemicals  . . . .
    chem_smiles = chem_dict[batch_data['InChIKey'].values.tolist()].values.tolist()
    chem_graph_list = []
    for smiles in chem_smiles:
        mol = Chem.MolFromSmiles(smiles)
        graph = mol_to_graph_data_obj_simple(mol)
        chem_graph_list.append(graph)
    chem_graphs_loader = DataLoader(chem_graph_list, batch_size=batch_data.shape[0],
                                    shuffle=False)
    for batch in chem_graphs_loader:
        chem_graphs = batch
    #  . . . .  proteins  . . . .
    if prot_descriptor_choice =='DISAE':
        uniprot_list = batch_data['uniprot+pfam'].values.tolist()
        protein_tokenized = torch.tensor([tokenizer.encode(protein_dict[uni]) for uni in uniprot_list  ])

    elif prot_descriptor_choice == 'TAPE':
        uniprot_list = batch_data['uniprot+pfam'].values.tolist()
        protein_tokenized_unpad = [torch.tensor(tokenizer.encode(protein_dict[uni])[1:-1]) for uni in uniprot_list]
        protein_tokenized_pad = pad_sequence(protein_tokenized_unpad, padding_value=0)
        # protein_tokenized = protein_tokenized_pad[:, :-1]
        protein_tokenized = protein_tokenized_pad.T
    else:
        batch_seq = list(zip(list(protein_dict[batch_data['uniprot+pfam']].index),
                             protein_dict[batch_data['uniprot+pfam']].values.tolist()))
        batch_labels, batch_strs, protein_tokenized = tokenizer(batch_seq)
    return chem_graphs, protein_tokenized
