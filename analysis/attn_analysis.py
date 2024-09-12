import os
import torch
import argparse
import warnings
import selfies as sf
import matplotlib.pyplot as plt
import numpy as np
from transformers import  AutoTokenizer, GPT2LMHeadModel, T5Tokenizer, T5EncoderModel
import re
import torch
import argparse
from transformers import AutoTokenizer, EsmModel
import ast
warnings.filterwarnings("ignore")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def produce_prot_t5_embedding(batch, encoder_max_length):
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False, legacy=True, clean_up_tokenization_spaces=True)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", torch_dtype=torch.float16).to("cuda:1")
    sequence_examples = " ".join(list(re.sub(r"[UZOB]", "X", batch)))
    ids = tokenizer.encode_plus(sequence_examples, add_special_tokens=True, padding="max_length", max_length=encoder_max_length)
    input_ids = torch.tensor(ids['input_ids']).to("cuda:1").view(1,-1)
    attention_mask = torch.tensor(ids['attention_mask']).to("cuda:1").view(1,-1)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    return embedding_repr.last_hidden_state, input_ids

def produce_esm2_embedding(data, prot_len):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", clean_up_tokenization_spaces=True)
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", torch_dtype=torch.float16).to(device)
    model.eval()
    inputs = tokenizer(data, return_tensors="pt", padding="max_length", max_length=prot_len, truncation=True).to(device)
    outputs = model(**inputs)

    sequence_out = outputs.last_hidden_state
    return sequence_out, inputs

def visualize_cross_attention_scores(cross_attentions, sf_sample_token_list, esm2, prot_sample, index_ranges, output_folder="attention_plots"):
    num_layers = len(cross_attentions)
    
    sf_sample_token_list = ["<bos>"] + sf_sample_token_list + ["<eos>"]

    aa_dict = {'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys', 'E': 'Glu', 'Q': 'Gln',
               'G': 'Gly', 'H': 'His', 'I': 'Ile', 'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe',
               'P': 'Pro', 'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val', 'X': 'PAD'}

    if esm2 is True:
        prot_sample = "X" + prot_sample

    x_labels = []
    for start, end in index_ranges:
        if esm2 is True:
            x_labels.extend([f"{aa_dict.get(aa, aa)}{i}" for i, aa in enumerate(prot_sample[start:end], start=start)])
        elif args.esm2 is False:   
            x_labels.extend([f"{aa_dict.get(aa, aa)}{i+1}" for i, aa in enumerate(prot_sample[start:end], start=start)])
        x_labels.append("")  # Add a space between ranges

    os.makedirs(output_folder, exist_ok=True)
    
    for layer in range(num_layers):
        layer_attention = cross_attentions[layer].squeeze(0)
        num_heads = layer_attention.shape[0]
        
        for head in range(num_heads):
            print(f"Layer: {layer}, Head: {head}")
            scores = []
            for start, end in index_ranges:
                range_scores = layer_attention[head, :, start:end].detach().cpu().numpy()
                scores.append(range_scores)
                # Add a column of zeros between ranges, matching the shape of range_scores
                scores.append(np.zeros((range_scores.shape[0], 1)))
            scores.pop()
            scores = np.hstack(scores)
            fig, ax = plt.subplots(figsize=(20, 15))
            im = ax.imshow(scores, cmap='Greens', aspect='auto', interpolation='nearest')
            
            ax.set_title(f'Layer {layer+1}, Head {head+1} - Cross Attention Heatmap', fontsize=24)
            
            ax.set_yticks(np.arange(len(sf_sample_token_list)))
            ax.set_yticklabels(sf_sample_token_list, fontsize=16)
            ax.set_ylabel('SELFIES Tokens', fontsize=20)
            
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=90, ha='center', fontsize=16)
            ax.set_xlabel('Protein Sequence', fontsize=20)
            
            ax.set_xticks(np.arange(scores.shape[1]+1)-.5, minor=True)
            ax.set_yticks(np.arange(scores.shape[0]+1)-.5, minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
            
            ax.set_frame_on(False)
            
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.outline.set_visible(False)
            cbar.set_label('Attention', fontsize=20)
            cbar.ax.tick_params(labelsize=16)
            
            plt.tight_layout()
            
            filename = f"layer_{layer+1}_head_{head+1}.png"
            plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
            plt.close(fig)

def main(args):
    model_file = args.model_file
    ligand_smiles = args.ligand_smiles
    ligand_name = args.ligand_name
    protein_sequence = args.protein_sequence
    start_idx = args.start_idx
    end_idx = args.end_idx
    prot_name = args.prot_name
    index_ranges = ast.literal_eval(args.index_ranges)
    print(args.esm2)
    if args.esm2 is True:
        prot_emb_model = "esm2"
    elif args.esm2 is False:   
        prot_emb_model = "protT5"
    # Load tokenizer and model
    print("Loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left")
    model = GPT2LMHeadModel.from_pretrained(model_file, torch_dtype=torch.float16).to("cuda:1")
    #generation_config = GenerationConfig.from_pretrained(model_file)
    model.config.bos_token_id = 1
    
    # Generate protein embedding
    if args.esm2 is True:
        prot_emb, input_ids = produce_esm2_embedding(protein_sequence, 1000)
    elif args.esm2 is False:   
        prot_emb, input_ids = produce_prot_t5_embedding(protein_sequence, 1000)

    # Tokenize ligand SMILES
    ligand_sf = sf.encoder(ligand_smiles)
    ligand_sf_token_list = list(sf.split_selfies(ligand_sf))
    ligand_tokens = tokenizer.encode(ligand_sf)

    # Generate model outputs
    inputs = {"input_ids": torch.tensor([ligand_tokens]).to("cuda:1"),
              "encoder_hidden_states": prot_emb}
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    # Visualize attention scores
    visualize_cross_attention_scores(outputs.cross_attentions, ligand_sf_token_list, args.esm2, protein_sequence, 
                                     index_ranges=index_ranges, 
                                     output_folder=f"attention_maps/{ligand_name}_{prot_emb_model}_{prot_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate attention maps for protein-ligand interactions")
    parser.add_argument("--model_file", type=str, default="", help="Path to the model file")
    parser.add_argument("--ligand_smiles", type=str, default=True, help="SMILES string of the ligand")
    parser.add_argument("--ligand_name", type=str, default=True, help="Name of the ligand")
    parser.add_argument("--protein_sequence", type=str, default=True, help="Protein sequence")
    parser.add_argument("--prot_name", type=str, default="", help="Name of the protein")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for protein sequence visualization")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for protein sequence visualization")
    parser.add_argument("--index_ranges", type=str, default="[[0, None]]", help="List of index ranges for protein sequence visualization, e.g., '[[10, 90], [100, 120]]'")
    parser.add_argument("--esm2", default=False, type=bool, help="Use ESM-2 embedding")
    args = parser.parse_args()
    main(args)
