from transformers import BartTokenizer, GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import os 
import math
from utils import metrics_calculation
import numpy as np
from transformers import DataCollatorForLanguageModeling
import selfies as sf
import pandas as pd
import argparse
import torch 
import torch.distributed
import sys
sys.path.insert(1, '../data_processing')
import train_val_test
from data_loader import CustomDataset, CustomEffDataset
from gpt2_trainer import GPT2_w_crs_attn_Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"

class FinetuneScript(object):
    
    """(FineTune)Trainer for training and testing Prot2Mol."""

    def __init__(self, config,
                       selfies_path, 
                       finetune_save_to,
                       model_name,
                       dataset_name,
                       run_name):
        
        self.selfies_path = selfies_path
        self.finetune_save_to = finetune_save_to
        self.model_name = model_name
        self.prot_emb_model = config.prot_emb_model
        self.run_name = run_name
        self.max_mol_len = config.max_mol_len

        self.TRAIN_BATCH_SIZE = config.train_batch_size
        self.VALID_BATCH_SIZE = config.valid_batch_size
        self.TRAIN_EPOCHS = config.epoch
        self.LEARNING_RATE = config.learning_rate
        self.WEIGHT_DECAY = config.weight_decay

        self.training_vec = np.load("../data/train_vecs.npy")
        
        self.train_data, self.eval_data, self.test_data = train_val_test.train_val_test_split(config.selfies_path, config.target_id)

        if "af2" in self.prot_emb_model:
            self.prot_emb_model_path = f"../data/prot_embed/{self.prot_emb_model}/FoldedPapyrus_4581_v01/embeddings.npz"
        else:
            self.prot_emb_model_path = f"../data/prot_embed/{self.prot_emb_model}/{dataset_name}/embeddings_fp16.h5"
        print(f"Load protein embeddings {self.prot_emb_model_path}...\n")
        
        #self.target_data = np.load(prot_emb_model_path, mmap_mode='r')
        #self.N_EMBED = np.array(self.target_data["encoder_hidden_states"][0]).shape[-1]
        
        self.tokenizer = BartTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left")    
        alphabet =  list(sf.get_alphabet_from_selfies(list(self.train_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(alphabet)
        alphabet =  list(sf.get_alphabet_from_selfies(list(self.eval_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(alphabet)
        del alphabet

        print("Loading model from:", self.model_name)
            
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        print("Model parameter count:", format(self.model.num_parameters(), "_d"))

    def compute_metrics(self, eval_pred):
        #if torch.distributed.get_rank() == 0:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        return metrics_calculation(predictions=decoded_preds, references=decoded_labels, train_data = self.train_data, train_vec = self.training_vec)

        
    def finetune_with_target(self):
        
        finetune_data = self.test_data
        self.alphabet =  list(sf.get_alphabet_from_selfies(list(finetune_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(self.alphabet)
        
        finetune_dataset = CustomEffDataset(ligand_data=finetune_data, target_data=self.prot_emb_model_path, tokenizer=self.tokenizer, max_length=self.max_mol_len)
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        
        self.model.train()
        
        finetune_args = TrainingArguments(
            run_name=self.run_name,
            output_dir=self.finetune_save_to,
            overwrite_output_dir=True,
            evaluation_strategy="steps", #evaluate every 10 epoch
            eval_steps=200,
            save_strategy="epoch",
            num_train_epochs=self.TRAIN_EPOCHS,
            learning_rate=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
            per_device_train_batch_size=self.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.VALID_BATCH_SIZE,
            save_total_limit=1,
            disable_tqdm=True,
            logging_steps=10,
            dataloader_num_workers=10,
            fp16=True,
            ddp_find_unused_parameters=False)

        finetune_trainer = GPT2_w_crs_attn_Trainer(
            model=self.model,
            args=finetune_args,
            train_dataset=finetune_dataset,
            eval_dataset=finetune_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
            # prediction_loss_only=True,
        )
        finetune_trainer.args._n_gpu = 1
        
        print("build finetuning with on device:", finetune_args.device, "with n gpus:", finetune_args.n_gpu)
        
        finetune_trainer.train()
        print("training finished.")

        eval_results = finetune_trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        finetune_trainer.save_model(self.finetune_save_to)      
 
def main(config):
    pretrain_model = config.pretrained_model_path.split("/")[-2]
    run_name = f"""TARGET_{str(config.target_id)}_lr_{str(config.learning_rate)}_bs_{str(config.train_batch_size)}_ep_{str(config.epoch)}_wd_{str(config.weight_decay)}"""
    if config.prot_dataset_name is None:
        dataset_name = config.selfies_path.split("/")[-1].split(".")[0]
    else:
        dataset_name = config.prot_dataset_name
    
    trainingscript = FinetuneScript(config=config, 
                                    selfies_path=config.selfies_path, 
                                    finetune_save_to=f"../finetuned_models/{pretrain_model}/{run_name}",
                                    model_name = config.pretrained_model_path,
                                    dataset_name=dataset_name,
                                    run_name=run_name)
    
    trainingscript.finetune_with_target()
    
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Dataset parameters
    
    parser.add_argument("--selfies_path", required=False, default="../data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False.csv", help="Path of the SELFIES dataset.")
    parser.add_argument("--target_id", default="CHEMBL4282", help="Target ID (ChEMBL_ID) to finetune on.")
    parser.add_argument("--prot_emb_model", required=False, default="esm2", help="Which protein embedding model to use", choices=["prot_t5", "esm2", "esm3", "af2_single", "af2_struct", "af2_combined"])
    # Model parameters
    parser.add_argument("--prot_dataset_name", default=None)
    parser.add_argument("--pretrained_model_path", default="../saved_models/lr_1e-05_bs_64_ep_50_wd_0.0005_nlayer_4_nhead_16_prot_esm2_dataset_prot_comp_set_pchembl_6_protlen_1000_human_False_fp16/checkpoint-294600")
    
    parser.add_argument("--learning_rate", default=1.0e-5)
    parser.add_argument("--max_mol_len", default=200)
    parser.add_argument("--train_batch_size", default=64)
    parser.add_argument("--valid_batch_size", default=64)
    parser.add_argument("--epoch", default=50)
    parser.add_argument("--weight_decay", default=0.0005)
    parser.add_argument("--max_positional_emb", default=202)
    parser.add_argument("--n_layer", default=4)
    parser.add_argument("--n_head", default=16)
    parser.add_argument("--n_emb", default=1280) # prot_t5=1024, esm2=1280

    
    config = parser.parse_args()
    main(config)

