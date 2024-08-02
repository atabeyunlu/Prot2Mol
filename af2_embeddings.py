import os
import numpy as np
from datasets import Dataset
from conversion import ChemblUniprotConverter
import argparse

converter = ChemblUniprotConverter()

def get_af2_embeddings(path, file_name):
    folder_names = []
    all_structures = []
    for root, _, files in os.walk(path):
        current_directory_path = os.path.abspath(root)
        folder_name = os.path.basename(root)
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

def create_combined_hf_dataset(path):
    all_structures, folder_names = get_af2_embeddings(path, "structure.npy")
    prot_names = [converter.convert_2_chembl_id(folder_name) for folder_name in folder_names]
    dataset = Dataset.from_dict({"Target_CHEMBL_ID": prot_names, "encoder_hidden_states": all_structures})
    
    all_single, folder_names = get_af2_embeddings(path, "single.npy")
    prot_names = [converter.convert_2_chembl_id(folder_name) for folder_name in folder_names]
    single_dataset = Dataset.from_dict({"Target_CHEMBL_ID": prot_names, "encoder_hidden_states": all_single})
    
    dataset = dataset.concatenate(single_dataset, axis=-1)
    return dataset

def create_hf_dataset(path, file_name):
    all_structures, folder_names = get_af2_embeddings(path, file_name)
    prot_names = [converter.convert_2_chembl_id(folder_name) for folder_name in folder_names]
    dataset = Dataset.from_dict({"Target_CHEMBL_ID": prot_names, "encoder_hidden_states": all_structures})
    return dataset

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--af2_data", required=True, metavar="./data/proteins/", help="Path of the SELFIES dataset.")
    parser.add_argument("--max_len", required=True, metavar=500, help="Maximum protein length")
    config = parser.parse_args()        
    
    path = config.af2_data
    max_length = config.max_len  # Change this to the length you want to pad to
    
    dataset = create_hf_dataset(path, "structure.npy")
    dataset.save_to_disk("./data/prot_embed/af2_struct/embeddings")
    
    dataset = create_hf_dataset(path, "single.npy")
    dataset.save_to_disk("./data/prot_embed/af2_single/embeddings")
    
    dataset = create_combined_hf_dataset(path)
    dataset.save_to_disk("./data/prot_embed/af2_combined/embeddings")
    
    
