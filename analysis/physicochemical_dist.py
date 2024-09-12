import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')   
import random
random.seed(0)
from rdkit.Chem import MACCSkeys
import umap
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from matplotlib.colors import to_rgba
import os
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from sklearn.manifold import TSNE
import selfies as sf

akt1 = pd.read_csv("/home/atabey/Prot2Mol/data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False/test_CHEMBL4282.csv")
akt1["smiles"] = akt1["Compound_SELFIES"].apply(lambda x: sf.decoder(x))
akt_fp = pd.DataFrame(columns=["fp"])
akt_fp["fp"] = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024).ToBitString() for x in akt1["smiles"] if Chem.MolFromSmiles(x) is not None]
akt_fp = akt_fp.assign(model_name = "akt1")

drd4 = pd.read_csv("/home/atabey/Prot2Mol/data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False/test_CHEMBL219.csv")
drd4["smiles"] = drd4["Compound_SELFIES"].apply(lambda x: sf.decoder(x))
drd4_fp = pd.DataFrame(columns=["fp"])
drd4_fp["fp"] = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024).ToBitString() for x in drd4["smiles"] if Chem.MolFromSmiles(x) is not None]
drd4_fp = drd4_fp.assign(model_name = "drd4")

prot2mol_akt1 = pd.read_csv("/home/atabey/Prot2Mol/saved_mols/prot_t5/papyrus_CHEMBL4282_TARGET_CHEMBL4282_lr_1e-05_bs_64_ep_50_wd_0.0005_prot_t5/10000_mols.csv")
prot2mol_akt1_fp = pd.DataFrame(columns=["fp"])
prot2mol_akt1_fp["fp"] = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024).ToBitString() for x in prot2mol_akt1["smiles"] if Chem.MolFromSmiles(x) is not None]
prot2mol_akt1_fp = prot2mol_akt1_fp.assign(model_name = "prot2mol_akt1")

prot2mol_drd4 = pd.read_csv("/home/atabey/Prot2Mol/saved_mols/prot_t5/papyrus_CHEMBL219_TARGET_CHEMBL219_lr_1e-05_bs_64_ep_50_wd_0.0005_prot_t5/10000_mols.csv")
prot2mol_drd4_fp = pd.DataFrame(columns=["fp"])
prot2mol_drd4_fp["fp"] = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024).ToBitString() for x in prot2mol_drd4["smiles"] if Chem.MolFromSmiles(x) is not None]
prot2mol_drd4_fp = prot2mol_drd4_fp.assign(model_name = "prot2mol_drd4")


# distribution of (A) molecular weight, (B) number of heavy atoms, (C) AlogP, (D) polar surface area (PSA), (E) hydrogen bond acceptor (HBA), (F) hydrogen bond donor (HBD), (G) rotatable bonds, and (H) aromatic rings

from rdkit.Chem import Descriptors

def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return pd.Series({
            'MW': Descriptors.ExactMolWt(mol),
            'HeavyAtoms': Descriptors.HeavyAtomCount(mol),
            'ALogP': Descriptors.MolLogP(mol),
            'PSA': Descriptors.TPSA(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol)
        })
    return pd.Series()

xtick_intervals = {"MW": [0,1000],
                   "HeavyAtoms": [0,80],
                   "ALogP": [-1,10],
                   "PSA": [0,175],
                   "HBA": [0,11],
                   "HBD": [0,5],
                   "RotatableBonds": [0,15],
                   "AromaticRings": [0,7]}

datasets = {
    'Real Molecules (AKT1 Fine-tuning Data)': akt1,
    'Real Molecules (DRD4 Fine-tuning Data)': drd4,
    'De novo AKT1 (Fine-tuned-protT5)': prot2mol_akt1,
    'De novo DRD4 (Fine-tuned-protT5)': prot2mol_drd4
}

properties = ['MW', 'HeavyAtoms', 'ALogP', 'PSA', 'HBA', 'HBD', 'RotatableBonds', 'AromaticRings']
titles = ['Molecular Weight', 'Number of Heavy Atoms', 'ALogP', 'Polar Surface Area (PSA)', 
          'Hydrogen Bond Acceptors (HBA)', 'Hydrogen Bond Donors (HBD)', 'Rotatable Bonds', 'Aromatic Rings']

fig, axes = plt.subplots(2, 4, figsize=(30, 20))

# carry suptitle to a higher position
#fig.subplots_adjust(top=0.9)
#fig.suptitle('Distribution of Physicochemical Properties', fontsize=16)

for idx, (prop, title) in enumerate(zip(properties, titles)):
    ax = axes[idx // 4, idx % 4] 
    for name, df in datasets.items():
        data = df['smiles'].apply(calculate_properties).dropna()
        sns.kdeplot(data=data[prop], ax=ax, label=name)
    ax.set_xlim(xtick_intervals[prop])
    ax.set_title(title)
    ax.set_xlabel(prop)
    ax.set_ylabel('Density')
    ax.legend()

plt.tight_layout()
plt.savefig('physicochemical_property/physicochemical_properties_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("Physicochemical properties distribution plot saved as 'physicochemical_properties_distribution.png'")
