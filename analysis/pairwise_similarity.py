import argparse
import os
from rdkit import Chem
import numpy as np
import pandas as pd
import selfies as sf
from rdkit.Chem import AllChem
import torch
import matplotlib.pyplot as plt
import seaborn as sns

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
    return np.mean(agg_tanimoto) 

def generate_vecs(mols):
    zero_vec = np.zeros(1024)
    mols = to_mol(mols)
    return np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) if mol is not None else zero_vec for mol in mols])

def to_mol(smiles_list):
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

def to_smiles(selfies_list):
    return [sf.decoder(selfies) for selfies in selfies_list]

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    if "Compound_SELFIES" in df.columns:
        df["smiles"] = to_smiles(df["Compound_SELFIES"])
    elif "smiles" not in df.columns:
        raise ValueError(f"File {file_path} must contain either 'Compound_SELFIES' or 'smiles' column")
    return generate_vecs(df["smiles"])

def compute_similarity_matrix(dataset_list, dataset_names):
    n = len(dataset_list)
    correlation_df = pd.DataFrame(index=dataset_names, columns=dataset_names)
    
    for i in range(n):
        for j in range(n):
            if i < j:
                similarity = round(average_agg_tanimoto(dataset_list[i], dataset_list[j]), 3)
                correlation_df.loc[dataset_names[i], dataset_names[j]] = similarity
                correlation_df.loc[dataset_names[j], dataset_names[i]] = similarity
            elif i == j:
                correlation_df.loc[dataset_names[i], dataset_names[j]] = 1.000
    
    return correlation_df

def plot_heatmap(correlation_df, output_file):
    correlation_matrix = correlation_df.astype(float)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    cmap.set_over('white')
    vmax = correlation_matrix[correlation_matrix < 1].max().max()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt='.3f', ax=ax,
                vmin=correlation_matrix.min().min(), vmax=vmax, 
                center=(vmax+correlation_matrix.min().min())/2,
                mask=mask)

    ax.set_title('Pairwise Similarity Heatmap', fontsize=16)
    ax.set_xlabel('Datasets', fontsize=12)
    ax.set_ylabel('Datasets', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main(args):
    dataset_list = []
    dataset_names = []

    for file_path in args.input_files:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        dataset_names.append(dataset_name)
        dataset_list.append(load_dataset(file_path))

    correlation_df = compute_similarity_matrix(dataset_list, dataset_names)
    correlation_df.to_csv(args.output_csv)
    
    plot_heatmap(correlation_df, args.output_heatmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute pairwise similarity between molecular datasets")
    parser.add_argument("--input_files", nargs="+", help="Input CSV files containing molecular data")
    parser.add_argument("--input_names", nargs="+", help="Names of the input datasets")
    parser.add_argument("--output_csv", default="pairwise_similarity.csv", help="Output CSV file for similarity matrix")
    parser.add_argument("--output_heatmap", default="pairwise_similarity_heatmap.png", help="Output PNG file for heatmap")
    args = parser.parse_args()

    main(args)