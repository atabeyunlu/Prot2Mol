import re
import os
import torch
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import T5Tokenizer, T5EncoderModel
import h5py
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False, legacy=True, clean_up_tokenization_spaces=True)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", torch_dtype=torch.float16).to(device)


def produce_prot_t5_embedding(batch,encoder_max_length):

  sequence_examples = " ".join(list(re.sub(r"[UZOB]", "X", batch))) 
  ids = tokenizer.encode_plus(sequence_examples, add_special_tokens=True, padding="max_length", max_length=encoder_max_length)
  input_ids = torch.tensor(ids['input_ids']).to(device).view(1,-1)
  attention_mask = torch.tensor(ids['attention_mask']).to(device).view(1,-1)
  with torch.no_grad():

    embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)
    


  return embedding_repr.last_hidden_state

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  # Dataset parameters
  parser.add_argument("--dataset", default="../data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False.csv", help="Path of the SELFIES dataset.")
  parser.add_argument("--prot_len", default=1000, help="Max length of the protein sequence.")
  config = parser.parse_args()
  
  dataset_name = config.dataset.split("/")[-1].split(".")[0]
  prot_path = "../data/prot_embed/prot_t5/" + dataset_name + "/"
  if not os.path.exists(prot_path):
    os.makedirs(prot_path)
  
  csv_path = prot_path + "unique_target.csv"
  data = pd.read_csv(config.dataset)
  unique_target  = data.drop_duplicates(subset=['Target_CHEMBL_ID'])[["Target_CHEMBL_ID", "Target_FASTA"]]
  unique_target["len"] = unique_target["Target_FASTA"].apply(lambda x: len(x))
  unique_target = unique_target[unique_target["len"] < config.prot_len].reset_index(drop=True)
  unique_target.to_csv(csv_path, index=False)

  #token_rep = np.array([produce_prot_t5_embedding(seq, config.prot_len).detach().cpu().numpy() for seq in unique_target["Target_FASTA"]]).squeeze()

  #hf_path = prot_path + "embeddings.npz"

  #np.savez(hf_path, Target_CHEMBL_ID=unique_target["Target_CHEMBL_ID"], encoder_hidden_states=token_rep)  # Save the embeddings
  
  # Define the h5 file path
  h5_path = prot_path + "embeddings_fp16.h5"
  
  # Create the h5 file and write the data
  with h5py.File(h5_path, 'w') as h5_file:
      dt = h5py.string_dtype(encoding='utf-8')
      
      # Create a single dataset for both Target_CHEMBL_ID and encoder_hidden_states
      h5_file.create_dataset("Target_CHEMBL_ID", data=unique_target["Target_CHEMBL_ID"].values.astype(dt))
      encoder_hidden_states_dataset = h5_file.create_dataset("encoder_hidden_states", 
                                                              shape=(len(unique_target), config.prot_len, 1024),  # Assuming 1024-dimensional embeddings
                                                              dtype='float16'
                                                              )
      
      # Write the embeddings into the h5 file
      for i, seq in enumerate(unique_target["Target_FASTA"]):
          token_rep = produce_prot_t5_embedding(seq, config.prot_len).detach().cpu()
          
          encoder_hidden_states_dataset[i] = token_rep
          
  print(f"H5 file with Target_CHEMBL_ID and encoder_hidden_states saved at {h5_path}")