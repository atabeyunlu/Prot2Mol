import os 
import sys
import math
import argparse
import numpy as np
import selfies as sf
from utils import metrics_calculation
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import BartTokenizer, GPT2Config, GPT2LMHeadModel
sys.path.insert(1, '../data_processing')
import train_val_test
import torch.distributed
from data_loader import CustomDataset, MemoryEfficientDataset
from gpt2_trainer import GPT2_w_crs_attn_Trainer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"

class TrainingScript(object):
    
    """(Pre)Trainer for training and testing Prot2Mol."""

    def __init__(self, config,
                       selfies_path, 
                       pretrain_save_to,
                       dataset_name,
                       run_name):
        
        self.selfies_path = selfies_path
        self.pretrain_save_to = pretrain_save_to
        self.prot_emb_model = config.prot_emb_model
        self.run_name = run_name
        
        self.TRAIN_BATCH_SIZE = config.train_batch_size
        self.VALID_BATCH_SIZE = config.valid_batch_size
        self.TRAIN_EPOCHS = config.epoch
        self.LEARNING_RATE = config.learning_rate
        self.WEIGHT_DECAY = config.weight_decay
        self.N_LAYER = config.n_layer
        self.N_HEAD = config.n_head
        self.max_mol_len = config.max_mol_len
        self.N_EMB = config.n_emb
        print("Load training vectors...\n")
        self.training_vec = np.load("../data/train_vecs.npy") # write a script for this
        print("Load train validation test data...\n")
        self.train_data, self.eval_data, self.test_data = train_val_test.train_val_test_split(config.selfies_path, config.prot_ID)
        
        if "af2" in self.prot_emb_model:
            self.prot_emb_model_path = f"../data/prot_embed/{self.prot_emb_model}/FoldedPapyrus_4581_v01/embeddings.npz"
        else:
            self.prot_emb_model_path = f"../data/prot_embed/{self.prot_emb_model}/{dataset_name}/embeddings.npz"
        print("Load protein embeddings...\n")
        self.target_data = np.load(self.prot_emb_model_path, mmap_mode='r', allow_pickle=True)
        

        self.tokenizer = BartTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left")    
        alphabet =  list(sf.get_alphabet_from_selfies(list(self.train_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(alphabet)
        alphabet =  list(sf.get_alphabet_from_selfies(list(self.eval_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(alphabet)
        del alphabet
        
        print("Load GPT2 model...\n")
        self.configuration = GPT2Config(add_cross_attention=True, is_decoder = True,
                                n_embd=self.N_EMB, n_head=self.N_HEAD, vocab_size=len(self.tokenizer.added_tokens_decoder), 
                                n_positions=256, n_layer=self.N_LAYER, bos_token_id=self.tokenizer.bos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id)
        self.model = GPT2LMHeadModel(self.configuration)
        
        print("Model parameter count:", self.model.num_parameters())

    def compute_metrics(self, eval_pred):
        #if torch.distributed.get_rank() == 0:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        return metrics_calculation(predictions=decoded_preds, references=decoded_labels, train_data = self.train_data, train_vec = self.training_vec)
        #else: 
            #return {}

    def model_training(self):
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        print("Build train datasets...\n")
        train_dataset = MemoryEfficientDataset(ligand_data=self.train_data, target_data=self.target_data, tokenizer=self.tokenizer, max_length=self.max_mol_len)
        print("Build eval datasets...\n")
        eval_dataset = MemoryEfficientDataset(ligand_data=self.eval_data, target_data=self.target_data, tokenizer=self.tokenizer, max_length=self.max_mol_len)
        print("Build test datasets...\n")
        #test_dataset = MemoryEfficientDataset(ligand_data=self.test_data, target_data=self.target_data, tokenizer=self.tokenizer, max_length=self.max_mol_len)
        
        self.model.train()
        
        training_args = TrainingArguments(
            run_name=self.run_name,
            output_dir=self.pretrain_save_to,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.TRAIN_EPOCHS,
            learning_rate=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
            per_device_train_batch_size=self.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.VALID_BATCH_SIZE,
            save_total_limit=1,
            disable_tqdm=True,
            logging_steps=10,
            #max_steps=(len(list(self.train_data.Compound_SELFIES))//self.TRAIN_BATCH_SIZE) * self.TRAIN_EPOCHS,
            dataloader_num_workers=15,
            fp16=True,
            ddp_find_unused_parameters=False)

        trainer = GPT2_w_crs_attn_Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics)
        
        trainer.args._n_gpu = 1
        
        print("build pretrain trainer with on device:", training_args.device, "with n gpus:", training_args.n_gpu)
        trainer.train()
        print("training finished.")

        eval_results = trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        trainer.save_model(self.pretrain_save_to) 

def main(config):
    dataset_name = config.selfies_path.split("/")[-1].split(".")[0]
    run_name =  f"""lr_{str(config.learning_rate)}_bs_{str(config.train_batch_size)}_ep_{str(config.epoch)}_wd_{str(config.weight_decay)}_nlayer_{str(config.n_layer)}_nhead_{str(config.n_head)}_prot_{config.prot_emb_model}_dataset_{dataset_name}_testID_{config.prot_ID}"""
    
    trainingscript = TrainingScript(config=config, 
                                    selfies_path=config.selfies_path, 
                                    pretrain_save_to=f"../saved_models/{run_name}",
                                    dataset_name = dataset_name,
                                    run_name=run_name)
    
    trainingscript.model_training()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Dataset parameters

    parser.add_argument("--selfies_path", required=False, default="../data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False.csv", help="Path of the SELFIES dataset.")
    parser.add_argument("--prot_emb_model", required=False, default="prot_t5", help="Which protein embedding model to use", choices=["prot_t5", "esm2", "esm3", "af2_single", "af2_struct", "af2_combined"])
    parser.add_argument("--prot_ID", required=False, default="CHEMBL4282")
    
    # Model parameters
    parser.add_argument("--learning_rate", default=1.0e-5)
    parser.add_argument("--max_mol_len", default=200)
    parser.add_argument("--train_batch_size", default=64)
    parser.add_argument("--valid_batch_size", default=64)
    parser.add_argument("--epoch", default=50)
    parser.add_argument("--weight_decay", default=0.0005)
    parser.add_argument("--max_positional_emb", default=202)
    parser.add_argument("--n_layer", default=4)
    parser.add_argument("--n_head", default=16)
    parser.add_argument("--n_emb", default=1024) # prot_t5=1024
    
    
    config = parser.parse_args()
    main(config)


              