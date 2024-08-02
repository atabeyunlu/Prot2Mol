import esm
import torch
import argparse
import pandas as pd
from datasets import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

def produce_esm2_embedding(data):

    unique_target_tuple = list(data.itertuples(index=False, name=None))

    _, _, batch_tokens = batch_converter(unique_target_tuple)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    
    return token_representations

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--dataset", required=True, metavar="./data/train_128.csv", help="Path of the SELFIES dataset.")
    config = parser.parse_args()
    
    data = pd.read_csv(config.dataset)
    unique_target  = data.drop_duplicates(subset=['Target_CHEMBL_ID']).drop(axis=1, labels=["Compound_CHEMBL_ID", "Compound_SELFIES"])
    
    token_rep = produce_esm2_embedding(unique_target)

    ds = Dataset.from_dict({"Target_CHEMBL_ID": list(unique_target["Target_CHEMBL_ID"]), "encoder_hidden_states": token_rep.cpu().numpy()})
    ds.save_to_disk('./data/prot_embed/esm2/embeddings')
    

