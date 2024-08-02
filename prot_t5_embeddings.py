import re
import torch
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import T5Tokenizer, T5EncoderModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
encoder_max_length=500

def produce_prot_t5_embedding(batch):

  sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch["Target_FASTA"]]
  ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="max_length", max_length=encoder_max_length)
  input_ids = torch.tensor(ids['input_ids']).to(device)
  attention_mask = torch.tensor(ids['attention_mask']).to(device)
  with torch.no_grad():
    
    embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)
    
  batch["encoder_hidden_states"] = embedding_repr.last_hidden_state

  return batch

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  # Dataset parameters
  parser.add_argument("--dataset", required=True, metavar="./data/train_128.csv", help="Path of the SELFIES dataset.")
  config = parser.parse_args()

  data = pd.read_csv(config.dataset)
  unique_target  = data.drop_duplicates(subset=['Target_CHEMBL_ID']).drop(axis=1, labels=["Compound_CHEMBL_ID", "Compound_SELFIES"]).to_csv('./data/unique_target.csv', index=False)
  data_files = {"target_data": "data/unique_target.csv"}
  ds = load_dataset('csv', data_files=data_files)

  target_data = ds["target_data"].map(
      produce_prot_t5_embedding, 
      batched=True,
      batch_size=64, 
      remove_columns=["Target_FASTA"])

  target_data.save_to_disk('./data/prot_embed/prot_t5/embeddings')