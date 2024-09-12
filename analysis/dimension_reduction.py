import argparse
import os
import random
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from matplotlib.colors import to_rgba
import selfies as sf

RDLogger.DisableLog('rdApp.*')
random.seed(0)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate UMAP or t-SNE visualizations for molecular fingerprints.")

    parser.add_argument("--fp_type", choices=["morgan", "maccs"], default="morgan", help="Fingerprint type to use")
    parser.add_argument("--algorithm", choices=["umap", "tsne"], default="umap", help="Dimensionality reduction algorithm")

    parser.add_argument("--reference_datasets", nargs="+", required=True, help="List of reference dataset names to process")
    parser.add_argument("--generated_datasets", nargs="+", required=True, help="List of generated dataset names to process")
    parser.add_argument("--background_dataset", required=True, help="List of names for reference datasets")

    parser.add_argument("--reference_names", nargs="+", required=True, help="List of names for background datasets")
    parser.add_argument("--background_name",  required=True, help="Name for background dataset")
    parser.add_argument("--gen_names", nargs="+", required=True, help="List of names for generated datasets")

    parser.add_argument("--n_neighbors", type=int, nargs="+", default=[10, 50, 100], help="List of n_neighbors values for UMAP")
    parser.add_argument("--min_dist", type=float, nargs="+", default=[0.1, 0.4, 0.8], help="List of min_dist values for UMAP")
    parser.add_argument("--metric", nargs="+", default=["jaccard", "dice"], help="List of metric values for UMAP")

    parser.add_argument("--perplexity", type=int, nargs="+", default=[10, 100, 500, 1000], help="List of perplexity values for t-SNE")
    parser.add_argument("--n_iter", type=int, nargs="+", default=[1000, 2000, 3000, 4000], help="List of n_iter values for t-SNE")

    parser.add_argument("--subsample", type=int, default=None, help="Number of samples to randomly select from each dataset")
    parser.add_argument("--background_subsample", type=int, default=None, help="Number of samples to randomly select from background dataset")

    parser.add_argument("--output_dir", default="dim_reduction", help="Output directory for results")

    return parser.parse_args()

def subsample_dataset(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    if len(df) <= n_samples:
        return df
    return df.sample(n=n_samples, random_state=42)

def get_fingerprint(smiles: str, fp_type: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if fp_type == "morgan":
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()
    elif fp_type == "maccs":
        return MACCSkeys.GenMACCSKeys(mol).ToBitString()

def load_reference_dataset(file_path: str, dataset_name: str, fp_type: str, subset: int = None) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if subset:
        df = df.sample(subset)
    df["smiles"] = df["Compound_SELFIES"].apply(sf.decoder)
    fp_df = pd.DataFrame({"fp": df["smiles"].apply(lambda x: get_fingerprint(x, fp_type))})
    fp_df = fp_df.dropna()
    fp_df["model_name"] = dataset_name
    return fp_df

def load_generated_dataset(file_path: str, dataset_name: str, fp_type: str, subset: int = None) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if subset:
        df = df.sample(subset)
    fp_df = pd.DataFrame({"fp": df["smiles"].apply(lambda x: get_fingerprint(x, fp_type))})
    fp_df = fp_df.dropna()
    fp_df["model_name"] = dataset_name
    return fp_df

def run_dimensionality_reduction(fps: np.ndarray, algorithm: str, **kwargs) -> np.ndarray:
    if algorithm == "umap":
        reducer = umap.UMAP(**kwargs)
    elif algorithm == "tsne":
        reducer = TSNE(n_components=2, verbose=1, **kwargs)
    return reducer.fit_transform(StandardScaler().fit_transform(fps))

def plot_and_save_embedding(embedding: np.ndarray, model_names: List[str], color_dict: Dict[str, tuple], 
                            size_dict: Dict[str, int], output_path: str, title: str):
    df = pd.DataFrame({
        "model_name": model_names,
        "dim-1": embedding[:, 0],
        "dim-2": embedding[:, 1],
        "point_size": [size_dict[name] for name in model_names]
    })

    # Save embedding as CSV
    csv_path = output_path.rsplit('.', 1)[0] + '.csv'
    df.to_csv(csv_path, index=False)

    # Plot and save as PNG
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="dim-1", y="dim-2", 
                    hue=df.model_name.tolist(),
                    size="point_size",
                    sizes=(1, 4),
                    linewidth=0, 
                    palette=color_dict, 
                    data=df)
    # increase font size
    plt.rcParams.update({'font.size': 15})
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    args = parse_arguments()
    print("Reference dataset is loading...\n")
    # Load reference_datasets
    reference_datasets = {gen_names: load_reference_dataset(name, 
                                   gen_names, args.fp_type, args.subsample) for name, gen_names in zip(args.reference_datasets, args.reference_names)}
    print("Generated dataset is loading...\n")
    #Load generated datasets
    generated_datasets = {gen_names: load_generated_dataset(name, 
                                   gen_names, args.fp_type, args.subsample) for name, gen_names in zip(args.generated_datasets, args.gen_names)}
    print("Background dataset is loading...\n")
    background_dataset = {args.background_name: load_reference_dataset(args.background_dataset, args.background_name, 
                                                                        args.fp_type, args.background_subsample)}
    

    print("Combining all fingerprints...\n")
    # Combine all fingerprints
    fp_all = pd.concat(list(background_dataset.values()) + list(reference_datasets.values()) + list(generated_datasets.values()), axis=0).reset_index(drop=True)
    fps = np.array([list(map(int, fp)) for fp in fp_all["fp"]])
    
    # Set up color and size dictionaries
    color_dict = {name: to_rgba(color, 1) for name, color in zip([args.background_name] + args.reference_names + args.gen_names, 
                                                                 ['lightgrey', 'red', 'darkblue', 'orange', 'cyan'])}

    # size 1 for background dataset, size 10 for generated and reference datasets
    size_dict = {args.background_name: 1}
    size_dict.update({name: 2 for name in args.reference_names + args.gen_names})

    print("Generating embeddings and plots...\n")
    # Generate embeddings and plots
    output_dir = os.path.join(args.output_dir, f"{args.algorithm}_{args.fp_type}_{'vs_'.join([args.background_name] + args.reference_names + args.gen_names)}")
    os.makedirs(output_dir, exist_ok=True)

    if args.algorithm == "umap":
        for n in args.n_neighbors:
            for d in args.min_dist:
                for m in args.metric:
                    print(f"Generating UMAP embedding with n_neighbors={n}, min_dist={d}, metric={m}...\n")
                    embedding = run_dimensionality_reduction(fps, args.algorithm, n_neighbors=n, min_dist=d, metric=m)
                    output_path = os.path.join(output_dir, f"umap_{n}_{d}_{m}.png")
                    plot_and_save_embedding(embedding, list(fp_all["model_name"]), color_dict, size_dict, output_path, 
                                            f"UMAP: n_neighbors={n}, min_dist={d}, metric={m}")
    elif args.algorithm == "tsne":
        for p in args.perplexity:
            for i in args.n_iter:
                print(f"Generating t-SNE embedding with perplexity={p}, n_iter={i}...\n")
                embedding = run_dimensionality_reduction(fps, args.algorithm, perplexity=p, n_iter=i)
                output_path = os.path.join(output_dir, f"tsne_{p}_{i}.png")
                plot_and_save_embedding(embedding, list(fp_all["model_name"]), color_dict, size_dict, output_path, 
                                        f"t-SNE: perplexity={p}, n_iter={i}")

if __name__ == "__main__":
    main()