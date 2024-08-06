from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_from_disk, IterableDataset
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw
from utils import *
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem import QED
import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger    
RDLogger.DisableLog('rdApp.*')  
from transformers import set_seed
from multiprocessing import Pool
import torch
import numpy as np
import wandb

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]


def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def fraction_unique(gen, k=None, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn(
                "Can't compute unique@{}.".format(k) +
                "gen contains only {} molecules".format(len(gen))
            )
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        canonic = [i for i in canonic if i is not None]
        #raise ValueError("Invalid molecule passed to unique@k")
    return 0 if len(gen) == 0 else len(canonic) / len(gen)


def novelty(gen, train, n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return 0 if len(gen_smiles_set) == 0 else len(gen_smiles_set - train_set) / len(gen_smiles_set)


def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1, no_list=True):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto) if no_list else agg_tanimoto


def metrics_calculation(predictions, references, train_data, train_vec,training=True):
    
    # clear tokenizer white spaces
    
    predictions = [x.replace(" ", "") for x in predictions]
    references = [x.replace(" ", "") for x in references]

    prediction_smiles = [sf.decoder(x) for x in predictions]
    
    prediction_validity_ratio = fraction_valid(prediction_smiles)

    
    if prediction_validity_ratio != 0:
        
        prediction_mols = [Chem.MolFromSmiles(x) for x in prediction_smiles if x != '']
    
        training_data_smiles = [sf.decoder(x) for x in train_data.Compound_SELFIES]
        reference_smiles = [sf.decoder(x) for x in references] 
        
        prediction_uniqueness_ratio = fraction_unique(prediction_smiles)
        
        prediction_smiles_novelty_against_training_samples = novelty(prediction_smiles, training_data_smiles)

        prediction_smiles_novelty_against_reference_samples = novelty(prediction_smiles, reference_smiles)
        
        prediction_vecs = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in prediction_mols if x is not None])
        reference_vec = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024) for x in reference_smiles if Chem.MolFromSmiles(x) is not None])
        
        #gen_smi_imgs_grid = Draw.MolsToGridImage(prediction_mols[:16], molsPerRow=4, subImgSize=(600, 600))
        
        #wandb.log({"examples":gen_smi_imgs_grid})
        
        predicted_vs_reference_sim = average_agg_tanimoto(reference_vec,prediction_vecs)
        

        predicted_vs_training_sim = average_agg_tanimoto(train_vec,prediction_vecs)
        
        IntDiv = 1 - average_agg_tanimoto(prediction_vecs, prediction_vecs, agg="mean")
        
        prediction_sa_score = np.mean([sascorer.calculateScore(x) for x in prediction_mols if x is not None])

        prediction_qed_score = np.mean([QED.qed(x) for x in prediction_mols if x is not None])
        
        prediction_logp_score = np.mean([Chem.Crippen.MolLogP(x) for x in prediction_mols if x is not None])
        
        
        metrics = {"validity": prediction_validity_ratio,
                   "uniqueness": prediction_uniqueness_ratio,
                   "novelty_against_training_samples": prediction_smiles_novelty_against_training_samples,
                   "novelty_against_reference_samples": prediction_smiles_novelty_against_reference_samples,
                   "intdiv": IntDiv,
                   "similarity_to_training_samples": predicted_vs_training_sim,
                   "similarity_to_reference_samples": predicted_vs_reference_sim,
                   "sa_score": prediction_sa_score,
                   "qed_score": prediction_qed_score,
                   "logp_score": prediction_logp_score}
    
    else:
        metrics = {"validity": 0,
                   "uniqueness": 0,
                   "novelty_against_training_samples": 0,
                   "novelty_against_reference_samples": 0,
                   "intdiv": 0,
                   "similarity_to_training_samples": 0,
                   "similarity_to_reference_samples": 0,
                   "sa_score": 0,
                   "qed_score": 0,
                   "logp_score": 0}
    if training: 
        wandb.log(metrics) 
    if training:
        return metrics
    else:
        return metrics, prediction_smiles
        


        
        
        