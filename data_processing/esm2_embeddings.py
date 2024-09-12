import os
import h5py
import torch
import argparse
import pandas as pd
from transformers import AutoTokenizer, EsmModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load ESM-2 model
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", clean_up_tokenization_spaces=True)
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", torch_dtype=torch.float16).to(device)
model.eval()

def produce_esm2_embedding(data, prot_len):

    inputs = tokenizer(data, return_tensors="pt", padding="max_length", max_length=prot_len, truncation=True).to(device)
    outputs = model(**inputs)

    sequence_out = outputs.last_hidden_state
    return sequence_out

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--dataset", required=False, default="../data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False/test_CHEMBL219.csv", help="Path of the SELFIES dataset.")
    parser.add_argument("--prot_len", default=1000, help="Max length of the protein sequence.")
    config = parser.parse_args()
    
    dataset_name = config.dataset.split("/")[-1].split(".")[0]
    data = pd.read_csv(config.dataset)
    unique_target  = data.drop_duplicates(subset=['Target_CHEMBL_ID'])[["Target_CHEMBL_ID", "Target_FASTA"]]
    unique_target["len"] = unique_target["Target_FASTA"].apply(lambda x: len(x))
    unique_target = unique_target[unique_target["len"] < config.prot_len].reset_index(drop=True)
    path  = "../data/prot_embed/esm2/" + dataset_name + "/unique_target.csv"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    unique_target.to_csv(path, index=False)

    #token_rep = np.array([produce_esm2_embedding(seq, config.prot_len).detach().cpu().numpy() for seq in unique_target["Target_FASTA"]]).squeeze()
    
    #print(token_rep.shape)
    
    prot_path= f"../data/prot_embed/esm2/{dataset_name}/"
    
    #np.savez(prot_path, Target_CHEMBL_ID=unique_target["Target_CHEMBL_ID"], encoder_hidden_states=token_rep)  # Save the embeddings

    h5_path = prot_path + "embeddings_fp16.h5"

    # Create the h5 file and write the data
    with h5py.File(h5_path, 'w') as h5_file:
        dt = h5py.string_dtype(encoding='utf-8')
        
        # Create a single dataset for both Target_CHEMBL_ID and encoder_hidden_states
        h5_file.create_dataset("Target_CHEMBL_ID", data=unique_target["Target_CHEMBL_ID"].values.astype(dt))
        encoder_hidden_states_dataset = h5_file.create_dataset("encoder_hidden_states", 
                                                                shape=(len(unique_target), config.prot_len, 1280),  # 1280-dimensional embeddings
                                                                dtype='float16')
        
        # Write the embeddings into the h5 file
        for i, seq in enumerate(unique_target["Target_FASTA"]):
            token_rep = produce_esm2_embedding(seq, config.prot_len).detach().cpu()
            
            encoder_hidden_states_dataset[i] = token_rep
            
    print(f"H5 file with Target_CHEMBL_ID and encoder_hidden_states saved at {h5_path}")

