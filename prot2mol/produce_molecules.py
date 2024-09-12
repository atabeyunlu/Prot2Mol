import os
import json
import tqdm
import h5py
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

def load_dataset(path, prot_emb_model, prot_id):
    
    print("Loading required data")
    train_data = pd.read_csv(f"{path}/train.csv")
    alphabet =  list(sf.get_alphabet_from_selfies(list(train_data.Compound_SELFIES)))
    tokenizer.add_tokens(alphabet)
    train_vec = np.load("../data/train_vecs.npy")

    eval_data = pd.read_csv(f"{path}/val.csv")
    alphabet =  list(sf.get_alphabet_from_selfies(list(eval_data.Compound_SELFIES)))
    tokenizer.add_tokens(alphabet)  
    del eval_data
    
    test_data = pd.read_csv(f"{path}/test_{prot_id}.csv")
    alphabet =  list(sf.get_alphabet_from_selfies(list(test_data.Compound_SELFIES)))
    tokenizer.add_tokens(alphabet)
    
    with h5py.File(prot_emb_model, 'r') as h5_file:
        target_ids = h5_file['Target_CHEMBL_ID'][:]
        
        target_id_to_index = {target_id.decode('utf-8'): i for i, target_id in enumerate(target_ids)}
        index = target_id_to_index.get(prot_id)  
        target_encoding = h5_file['encoder_hidden_states'][index]
    #target_data = np.load(prot_emb_model, allow_pickle=True)
    selected_target = test_data[test_data["Target_CHEMBL_ID"].isin([args.prot_id])].reset_index(drop=True)
    #target_encoding = get_target(target_data, args.prot_id)
    #del target_data
    sample = {"encoder_hidden_states": torch.from_numpy(np.expand_dims(target_encoding, axis=0)), "target_chembl_id": args.prot_id}
    return train_data, train_vec, selected_target, sample

def generate_molecules(data):
    if args.attn_output:
        print(data["encoder_hidden_states"].shape)
        generated_tokens = model.generate(encoder_hidden_states=data["encoder_hidden_states"],
                            num_return_sequences=args.bs,
                            do_sample=True,
                            max_length=200,
                            pad_token_id=1,
                            bos_token_id=1,
                            output_attentions = args.attn_output)
        print(generated_tokens)
    else:
        generated_tokens = model.generate(encoder_hidden_states=data,
                            num_return_sequences=args.bs,
                            do_sample=True,
                            max_length=200,
                            pad_token_id=1,
                            bos_token_id=1)
    
    generated_selfies = [tokenizer.decode(x, skip_special_tokens=True) for x in generated_tokens]
    return generated_selfies

print("Starting to generate molecules.")

def generation_loop(target_data, num_samples, bs, sample):

    gen_mols = []
    

    for _ in tqdm.tqdm(range(int(num_samples/bs))):
        gen_mols.extend(generate_molecules(sample))
        
    gen_mols_df = pd.DataFrame(gen_mols, columns=["Generated_SELFIES"])
    
    return gen_mols_df

print("Metrics are being calculated.")

def calc_metrics(dataset, prot_emb_model, prot_id, num_samples, bs, generated_mol_file):

    train_data, train_vec, selected_target, target_encoding = load_dataset(dataset, prot_emb_model=prot_emb_model, prot_id=prot_id)
    
    gen_mols_df = generation_loop(selected_target, num_samples, bs, target_encoding)
    
    print("Calculating metrics")
    
    metrics, generated_smiles = metrics_calculation(predictions=gen_mols_df["Generated_SELFIES"], 
                                references=selected_target["Compound_SELFIES"], 
                                train_data = train_data, 
                                train_vec = train_vec,
                                training=False)
    print(metrics)

    
    generated_smiles.to_csv(generated_mol_file, index=False)
    with open(generated_mol_file.replace(".csv", "_metrics.json"), "w") as f:
        json.dump(metrics, f)
    print("Molecules and metrics are saved.")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default="../finetuned_models/lr_1e-05_bs_64_ep_50_wd_0.0005_nlayer_4_nhead_16_prot_esm2_dataset_prot_comp_set_pchembl_6_protlen_1000_human_False_fp16/TARGET_CHEMBL4282_lr_1e-05_bs_64_ep_50_wd_0.0005/checkpoint-1000", help="Path of the pretrained model file.")
    parser.add_argument("--prot_emb_model", default="../data/prot_embed/esm2/prot_comp_set_pchembl_6_protlen_1000_human_False/embeddings_fp16.h5", help="Path of the pretrained model file.")
    parser.add_argument("--selfies_path", default='../data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False', help="Path of the input SEFLIES dataset.")
    parser.add_argument("--attn_output", default=False, help="Path of the output embeddings file.")
    parser.add_argument("--prot_id", default="CHEMBL4282", help="Target Protein ID.")
    parser.add_argument("--num_samples", default=10000, type=int, help="Sample number.")
    parser.add_argument("--bs", default=100, type=int, help="Batch size.")
    args = parser.parse_args()
    
    genearted_mol_folder_path = f"""../saved_mols/{args.selfies_path.split("/")[2]}_{args.prot_id}_{args.model_file.split("/")[-2]}_{args.prot_emb_model.split("/")[3]}"""
    if not os.path.exists(genearted_mol_folder_path):
        os.makedirs(genearted_mol_folder_path)
    # Load tokenizer and the model
    genearted_mol_file_path = f"""../saved_mols/{args.selfies_path.split("/")[2]}_{args.prot_id}_{args.model_file.split("/")[-2]}_{args.prot_emb_model.split("/")[3]}/{args.num_samples}_mols.csv"""
    print("Loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left") # we can convert this to our own tokenizer later.
    model_name = args.model_file
    model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda:0")
    generation_config = GenerationConfig.from_pretrained(model_name)
    model.config.bos_token_id = 1
    
    calc_metrics(args.selfies_path, args.prot_emb_model, args.prot_id, args.num_samples, args.bs, genearted_mol_file_path)