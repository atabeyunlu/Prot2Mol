from transformers import BartTokenizer, GPT2Config, GPT2LMHeadModel
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
import os 
import math
from utils import metrics_calculation
import numpy as np
from transformers import DataCollatorForLanguageModeling
from datasets import IterableDataset
import selfies as sf
import pandas as pd
import yaml
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"

class GPT2_w_crs_attn_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_sequence = inputs["input_ids"]

        last_hidden_state = inputs["encoder_hidden_states"]
    
    
        outputs = model(input_ids=input_sequence, encoder_hidden_states=last_hidden_state, labels=input_sequence)
        
        return (outputs.loss, outputs) if return_outputs else outputs.loss


class FientuneScript(object):
    
    """(FineTune)Trainer for training and testing Prot2Mol."""

    def __init__(self, config,
                       selfies_path, 
                       finetune_save_to,
                       model_name):
        
        self.selfies_path = selfies_path
        self.finetune_save_to = finetune_save_to
        self.model_name = model_name
        self.prot_emb_model = config.prot_emb_model

        self.TRAIN_BATCH_SIZE = config.train_batch_size
        self.VALID_BATCH_SIZE = config.valid_batch_size
        self.TRAIN_EPOCHS = config.epoch
        self.LEARNING_RATE = config.learning_rate
        self.WEIGHT_DECAY = config.weight_decay
        self.N_LAYER = config.n_layer

        self.training_vec = np.load("./data/train_vecs.npy")

        prot_emb_model_path = "./data/prot_embed/" + self.prot_emb_model + "/embeddings"
        self.target_data = load_from_disk(prot_emb_model_path)
        
        self.tokenizer = BartTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left")    
        self.alphabet =  list(sf.get_alphabet_from_selfies(list(self.train_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(self.alphabet)

        print("Loading model from:", self.model_name)
            
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

    def gen(self, ligand, target, tokenizer):
        for i in range(len(ligand)):
            
            x = tokenizer(ligand.loc[i]["Compound_SELFIES"], max_length=200, padding="max_length", truncation=True, return_tensors="pt")["input_ids"].squeeze()
            enc_state = target[target["Target_CHEMBL_ID"].index(ligand.loc[i]["Target_CHEMBL_ID"])]["encoder_hidden_states"]
            sample = {"input_ids": x, "encoder_hidden_states": enc_state}
            yield sample 

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return metrics_calculation(predictions=decoded_preds, references=decoded_labels, train_data = self.train_data, train_vec = self.training_vec)
    
    def finetune_with_target(self, target_id):
        
        finetune_data = pd.read_csv("./data/test.csv")
        finetune_data_selected = finetune_data[finetune_data["Target_CHEMBL_ID"].isin([target_id])].reset_index(drop=True)
        
        self.alphabet =  list(sf.get_alphabet_from_selfies(list(finetune_data_selected.Compound_SELFIES)))
        self.tokenizer.add_tokens(self.alphabet)
        
        finetune_dataset = IterableDataset.from_generator(self.gen, gen_kwargs={"ligand": finetune_data_selected, "target": self.target_data, "tokenizer": self.tokenizer})
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        
        self.model.train()
        
        finetune_args = TrainingArguments(
            output_dir=self.finetune_save_to,
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
            max_steps=(len(list(finetune_data.Compound_SELFIES))//self.TRAIN_BATCH_SIZE) * self.TRAIN_EPOCHS,
            fp16=True)

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
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Dataset parameters
    
    parser.add_argument("--selfies_path", required=False, default="./data/fasta_to_selfies_500.csv", help="Path of the SELFIES dataset.")
    parser.add_argument("--target_id", default="CHEMBL4282", help="Target ID (ChEMBL_ID) to finetune on.")
    parser.add_argument("--prot_emb_model", required=False, default="prot_t5", help="Which protein embedding model to use", choices=["prot_t5", "esm2", "esm3", "af2_single", "af2_struct", "af2_combined"])
    # Model parameters
    
    parser.add_argument("--pretrained_model_path", default="./saved_models/set_100_saved_model/checkpoint-31628")
    
    parser.add_argument("--learning_rate", default=1.0e-5)
    parser.add_argument("--max_mol_len", default=200)
    parser.add_argument("--train_batch_size", default=64)
    parser.add_argument("--valid_batch_size", default=64)
    parser.add_argument("--epoch", default=50)
    parser.add_argument("--weight_decay", default=0.0005)
    parser.add_argument("--max_positional_emb", default=202)
    parser.add_argument("--n_layer", default=4)

    
    config = parser.parse_args()

    run_name = f"""TARGET_{str(config.target_id)}_
                    lr_{str(config.learning_rate)}_
                    bs_{str(config.train_batch_size)}_
                    ep_{str(config.epoch)}_
                    wd_{str(config.weight_decay)}_
                    nlayer_{str(config.n_layer)}"""

    trainingscript = FientuneScript(hyperparameters_dict=config, 
                                    selfies_path=config.selfies_path, 
                                    finetune_save_to="./finetuned_models/" + run_name)
    
    trainingscript.finetune_with_target(config.target_id)
