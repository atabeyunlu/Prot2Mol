import attr
import torch
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
import torch.nn.functional as F
from esm.models.esm3 import ESM3
from huggingface_hub import login
from esm.tokenization import get_model_tokenizers
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.sdk.api import (ESMProtein,ESMProteinTensor,SamplingConfig,SamplingTrackConfig,)

login(token="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")

device = "cuda" if torch.cuda.is_available() else "cpu"

client = ESM3.from_pretrained(ESM3_OPEN_SMALL)

def add_padding(protein_tensor: ESMProteinTensor, max_length: int) -> ESMProteinTensor:
    tokenizers = get_model_tokenizers(ESM3_OPEN_SMALL)

    current_length = len(protein_tensor)

    if current_length >= max_length:
        raise ValueError(
            f"Protein length is {current_length} which is greater than the maximum length of {max_length}"
        )

    left_pad = 0
    right_pad = max_length - current_length

    empty_protein_tensor = ESMProteinTensor.empty(
        current_length - 2,  # Account for BOS/EOS that our input already has
        tokenizers=tokenizers,
        device=protein_tensor.device,
    )

    for track in attr.fields(ESMProteinTensor):
        track_tensor = getattr(protein_tensor, track.name)

        if track_tensor is None:
            if track.name == "coordinates":
                continue
            else:
                # Initialize from empty tensor
                track_tensor = getattr(empty_protein_tensor, track.name)

        if track.name == "coordinates":
            pad_token = torch.inf
            new_tensor = F.pad(
                track_tensor,
                (0, 0, 0, 0, left_pad, right_pad),
                value=pad_token,
            )
        elif track.name in ["function", "residue_annotations"]:
            pad_token = getattr(tokenizers, track.name).pad_token_id
            new_tensor = F.pad(
                track_tensor,
                (0, 0, left_pad, right_pad),
                value=pad_token,
            )
        else:
            pad_token = getattr(tokenizers, track.name).pad_token_id
            new_tensor = F.pad(
                track_tensor,
                (
                    left_pad,
                    right_pad,
                ),
                value=pad_token,
            )
        protein_tensor = attr.evolve(protein_tensor, **{track.name: new_tensor})

    return protein_tensor


def produce_esm3_embeddings(seq, max_len):

    protein = ESMProtein(
        sequence=(
            seq
            )
        )

    protein_tensor = client.encode(protein)
    protein_tensor_padded = add_padding(protein_tensor, max_len)
    output = client.forward_and_sample(
        protein_tensor_padded,
        SamplingConfig(sequence=SamplingTrackConfig(), return_per_residue_embeddings=True),
    )
    
    return output.per_residue_embedding
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--dataset", required=True, metavar="./data/train_128.csv", help="Path of the SELFIES dataset.")
    parser.add_argument("--max_len", required=True, metavar=1024, help="Maximum protein length")
    parser.add_argument("--huggingface_token", required=True, metavar="<token>", help="User huggingface token (read only)")    
    config = parser.parse_args()    

    data = pd.read_csv(config.dataset)
    unique_target  = data.drop_duplicates(subset=['Target_CHEMBL_ID']).drop(axis=1, labels=["Compound_CHEMBL_ID", "Compound_SELFIES"])
    
    token_rep = np.array([produce_esm3_embeddings(seq, config.max_len).cpu().numpy() for seq in unique_target["Target_FASTA"]])
    
    
    print(token_rep.shape)
    
    ds = Dataset.from_dict({"Target_CHEMBL_ID": list(unique_target["Target_CHEMBL_ID"]), "encoder_hidden_states": token_rep})
    ds.save_to_disk('./data/prot_embed/esm3/embeddings')
        


