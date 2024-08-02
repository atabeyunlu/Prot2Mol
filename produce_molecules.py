import os
import json
import tqdm
import torch
import argparse
import warnings
import pandas as pd
from utils import *
import selfies as sf
from rdkit import RDLogger    
RDLogger.DisableLog('rdApp.*')  
warnings.filterwarnings("ignore")
from datasets import load_from_disk
from transformers import GenerationConfig
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.utils import logging
logging.set_verbosity_error() 
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--model_file", default="/home/atabey/DrugGEN2.0/finetuned_models/set_100_finetuned_model/checkpoint-3100", help="Path of the pretrained model file.")
parser.add_argument("--generated_mol_file", default="/home/atabey/DrugGEN2.0/saved_mols/_kt_finetune_mols.csv", help="Path of the output embeddings file.")
parser.add_argument("--selfies_dataset", default='/home/atabey/DrugGEN2.0/data/test.csv', help="Path of the input SEFLIES dataset.")
parser.add_argument("--attn_output", default=False, help="Path of the output embeddings file.")
parser.add_argument("--prot_id", default="CHEMBL4282", help="Target Protein ID.")
parser.add_argument("--num_samples", default=10000, help="Sample number.")
parser.add_argument("--bs", default=100, help="Batch size.")
args = parser.parse_args()

# Load tokenizer and the model
print("Loading model and tokenizer")
tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left") # we can convert this to our own tokenizer later.
model_name = args.model_file
model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda:0")
generation_config = GenerationConfig.from_pretrained(model_name)
model.config.bos_token_id = 1

# Load dataset to be used
print("Loading required data")
train_data = pd.read_csv("/home/atabey/DrugGEN2.0/data/train.csv")
alphabet =  list(sf.get_alphabet_from_selfies(list(train_data.Compound_SELFIES)))
tokenizer.add_tokens(alphabet)
train_vec = np.load("/home/atabey/DrugGEN2.0/data/train_vecs.npy")

eval_data = pd.read_csv('/home/atabey/DrugGEN2.0/data/eval.csv')
alphabet =  list(sf.get_alphabet_from_selfies(list(eval_data.Compound_SELFIES)))
tokenizer.add_tokens(alphabet)  
  
df = pd.read_csv(args.selfies_dataset)
target_data = load_from_disk('data/targets_data')
alphabet =  list(sf.get_alphabet_from_selfies(list(df.Compound_SELFIES)))
tokenizer.add_tokens(alphabet)
selected_target = df[df["Target_CHEMBL_ID"].isin([args.prot_id])].reset_index(drop=True)


def get_target(target, target_id):

    enc_state = target[target["Target_CHEMBL_ID"].index(target_id)]["encoder_hidden_states"]
    sample = {"encoder_hidden_states": enc_state, "target_chembl_id": target_id}
    return sample 

def generate_molecules(data):
    generated_tokens = model.generate(encoder_hidden_states=data,
                            num_return_sequences=args.bs,
                            do_sample=True,
                            max_length=200,
                            pad_token_id=1,
                            output_attentions = True if args.attn_output else False)
    
    generated_selfies = [tokenizer.decode(x, skip_special_tokens=True) for x in generated_tokens]
    return generated_selfies

print("Starting to generate molecules.")

gen_mols = []
sample = get_target(target_data, args.prot_id)["encoder_hidden_states"].view(1,-1,1024).to("cuda:0")

for _ in tqdm.tqdm(range(int(args.num_samples/args.bs))):
    gen_mols.extend(generate_molecules(sample))
    
gen_mols_df = pd.DataFrame(gen_mols, columns=["Generated_SELFIES"])

print("Metrics are being calculated.")

metrics, generated_smiles = metrics_calculation(predictions=gen_mols_df["Generated_SELFIES"], 
                              references=selected_target["Compound_SELFIES"], 
                              train_data = train_data, 
                              train_vec = train_vec,
                              training=False)
print(metrics)

print("Molecules and metrics are saved.")
gen_mols_df["smiles"] = generated_smiles
gen_mols_df.to_csv(args.generated_mol_file, index=False)
with open(args.generated_mol_file.replace(".csv", "_metrics.json"), "w") as f:
    json.dump(metrics, f)