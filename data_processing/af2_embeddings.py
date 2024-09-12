import os
import numpy as np
from datasets import Dataset
from conversion import ChemblUniprotConverter
import argparse
import requests
import zipfile
import h5py
import pandas as pd
converter = ChemblUniprotConverter()

def download_af2_emb(url, save_path, target_dir):
    
    if not os.path.exists(save_path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            
            os.makedirs(target_dir, exist_ok=True)
            zip_ref.extractall(target_dir)
        os.remove(save_path)
    
def get_af2_embeddings(path, file_name, max_length, unique_target):
    folder_names = []
    all_structures = []
    for root, _, files in os.walk(path):
        current_directory_path = os.path.abspath(root)
        folder_name = os.path.basename(root)
        chembl_id = converter.convert_2_chembl_id(folder_name)
        if chembl_id in unique_target['Target_CHEMBL_ID'].values:
            for f in files:
                if f == file_name:
                    current_structure_path = os.path.join(current_directory_path, f)
                    current_structure = np.load(current_structure_path)
                    
                    # Pad the array if necessary
                    if current_structure.shape[0] < max_length:
                        padding = max_length - current_structure.shape[0]
                        current_structure = np.pad(current_structure, ((0, padding), (0, 0)), 'constant')
                        all_structures.append(current_structure)
                        folder_names.append(folder_name)
                    elif current_structure.shape[0] == max_length:
                        all_structures.append(current_structure)
                        folder_names.append(folder_name)

    # Concatenate all the structures into a single array
    if all_structures:
        all_structures = np.stack(all_structures, axis=0)

    return all_structures, folder_names

def create_combined_hf_dataset(path, max_length, save_path, unique_target):
    all_structures, folder_names = get_af2_embeddings(path, "structure.npy", max_length, unique_target)
    all_single, _ = get_af2_embeddings(path, "single.npy", max_length, unique_target)
    all_combined = np.concatenate((all_structures, all_single), axis=-1)
    prot_names = [converter.convert_2_chembl_id(folder_name) for folder_name in folder_names]

    print("combined data:", all_combined.shape)
    save_to_h5(save_path, prot_names, all_combined)

    return prot_names

def create_hf_dataset(path, file_name, max_length, save_path, unique_target):
    all_structures, folder_names = get_af2_embeddings(path, file_name, max_length, unique_target)
    prot_names = [converter.convert_2_chembl_id(folder_name) for folder_name in folder_names]
    print(file_name, ":", all_structures.shape)
    save_to_h5(save_path, prot_names, all_structures)

    

def save_to_h5(save_path, prot_names, embeddings):
    with h5py.File(save_path, 'w') as h5_file:
        dt = h5py.string_dtype(encoding='utf-8')
        h5_file.create_dataset("Target_CHEMBL_ID", data=np.array(prot_names, dtype=dt))
        h5_file.create_dataset("encoder_hidden_states", 
                               data=embeddings.astype(np.float16),
                               dtype='float16')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--max_len", required=False, default=1000, help="Maximum protein length")
    parser.add_argument("--dataset", required=False, default="../data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False.csv", help="dataset path")

    config = parser.parse_args()        
    
    
    
    save_path = "../data/FoldedPapyrus_4581_v01.zip"
    af2_url = "https://zenodo.org/records/10671261/files/FoldedPapyrus_4581_v01.zip"
    target_dir = "../data/af2/"

    dataset_name = config.dataset.split("/")[-1].split(".")[0]
    prot_path = "../data/prot_embed/af2/" + dataset_name + "/"
    if not os.path.exists(prot_path):
        os.makedirs(prot_path)
    
    csv_path = prot_path + "unique_target.csv"
    data = pd.read_csv(config.dataset)
    unique_target  = data.drop_duplicates(subset=['Target_CHEMBL_ID'])[["Target_CHEMBL_ID", "Target_FASTA"]]
    unique_target["len"] = unique_target["Target_FASTA"].apply(lambda x: len(x))
    unique_target = unique_target[unique_target["len"] < config.max_len].reset_index(drop=True)
    unique_target.to_csv(csv_path, index=False)

    if not os.path.exists(target_dir):
        print("\nDownloading AlphaFold2 embeddings...\n")
        download_af2_emb(af2_url, save_path, target_dir)
    print("Alphafold2 embeddings have been found skipping download.\n")
    
    max_length = config.max_len  # Change this to the length you want to pad to
    
    data_name = save_path.split("/")[-1].split(".")[0]
    path = f"../data/af2/proteins"
    
    print("Processing AlphaFold2 embeddings...\n")
    
    if not os.path.exists(f"../data/prot_embed/af2_struct/{data_name}"):
        os.makedirs(f"../data/prot_embed/af2_struct/{data_name}") 
    create_hf_dataset(path, "structure.npy", config.max_len, f"../data/prot_embed/af2_struct/{data_name}/embeddings.h5", unique_target)
    
    if not os.path.exists(f"../data/prot_embed/af2_single/{data_name}"):
        os.makedirs(f"../data/prot_embed/af2_single/{data_name}")
    create_hf_dataset(path, "single.npy", config.max_len, f"../data/prot_embed/af2_single/{data_name}/embeddings.h5", unique_target)
    
    if not os.path.exists(f"../data/prot_embed/af2_combined/{data_name}"):
        os.makedirs(f"../data/prot_embed/af2_combined/{data_name}")
    prot_names = create_combined_hf_dataset(path, config.max_len, f"../data/prot_embed/af2_combined/{data_name}/embeddings.h5", unique_target)

    #filter out the the targets that are not in prot_names
    new_dataset = data[data["Target_CHEMBL_ID"].isin(prot_names)]
    new_dataset.to_csv(f"../data/papyrus/{dataset_name}/af2_filtered.csv", index=False)
    
    print("Residual files are being removed...\n")
    
    os.system(f"rm -r {target_dir}")
    
    print("\nAlphaFold2 embeddings have been successfully downloaded and processed.\n")
