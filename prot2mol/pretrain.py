import os 
import math
import argparse
import numpy as np
import pandas as pd
import selfies as sf
from datasets import load_from_disk
from datasets import IterableDataset
from utils import metrics_calculation
from train_val_test import train_val_test_split
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import BartTokenizer, GPT2Config, GPT2LMHeadModel

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

class TrainingScript(object):
    
    """(Pre)Trainer for training and testing Prot2Mol."""

    def __init__(self, config,
                       selfies_path, 
                       pretrain_save_to,
                       dataset_name):
        
        self.selfies_path = selfies_path
        self.pretrain_save_to = pretrain_save_to
        self.prot_emb_model = config.prot_emb_model
        
        self.TRAIN_BATCH_SIZE = config.train_batch_size
        self.VALID_BATCH_SIZE = config.valid_batch_size
        self.TRAIN_EPOCHS = config.epoch
        self.LEARNING_RATE = config.learning_rate
        self.WEIGHT_DECAY = config.weight_decay
        self.N_LAYER = config.n_layer
        self.N_HEAD = config.n_head
        
        self.train_data = pd.read_csv("./data/train.csv")
        self.eval_data = pd.read_csv('./data/eval.csv')
        self.training_vec = np.load("./data/train_vecs.npy")
        
        self.train_data, self.eval_data, self.test_data = train_val_test_split(self.data, "CHEMBL4282")
        
        if "af2" in self.prot_emb_model:
            self.prot_emb_model_path = f"./data/prot_embed/{self.prot_emb_model}/FoldedPapyrus_4581_v01/embeddings"
        else:
            prot_emb_model_path = f"./data/prot_embed/{self.prot_emb_model}/{dataset_name}/embeddings"
        
        self.target_data = load_from_disk(prot_emb_model_path)
        self.N_EMBED = np.array(self.target_data[0]["encoder_hidden_states"]).shape[-1]
        self.tokenizer = BartTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left")    
        self.alphabet =  list(sf.get_alphabet_from_selfies(list(self.train_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(self.alphabet)

        self.configuration = GPT2Config(add_cross_attention=True, is_decoder = True,
                                n_embd=self.N_EMBED, n_head=self.N_HEAD, vocab_size=len(self.tokenizer.added_tokens_decoder), 
                                n_positions=256, n_layer=self.N_LAYER, bos_token_id=self.tokenizer.bos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id)
        self.model = GPT2LMHeadModel(self.configuration)
        
        print("Model parameter count:", self.model.num_parameters())

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

    def model_training(self):
        
        train_dataset = IterableDataset.from_generator(self.gen, gen_kwargs={"ligand": self.train_data, "target": self.target_data, "tokenizer": self.tokenizer})
        eval_dataset = IterableDataset.from_generator(self.gen, gen_kwargs={"ligand": self.eval_data, "target": self.target_data, "tokenizer": self.tokenizer})
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        
        self.model.train()
        
        print("training_len", len(list(self.train_data.Compound_SELFIES)), "\n", "bs", self.TRAIN_BATCH_SIZE, "\n", "epochs", self.TRAIN_EPOCHS, "\n", "steps", round((len(list(self.train_data.Compound_SELFIES))/self.TRAIN_BATCH_SIZE)*self.TRAIN_EPOCHS))
        
        training_args = TrainingArguments(
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
            max_steps=(len(list(self.train_data.Compound_SELFIES))//self.TRAIN_BATCH_SIZE) * self.TRAIN_EPOCHS,
            fp16=True)

        trainer = GPT2_w_crs_attn_Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics)
        
        trainer.args._n_gpu = 1
        
        print("build finetune trainer with on device:", training_args.device, "with n gpus:", training_args.n_gpu)
        trainer.train()
        print("training finished.")

        eval_results = trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        trainer.save_model(self.pretrain_save_to) 
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Dataset parameters

    parser.add_argument("--selfies_path", required=False, default="./data/fasta_to_selfies_500.csv", help="Path of the SELFIES dataset.")
    parser.add_argument("--prot_emb_model", required=False, default="prot_t5", help="Which protein embedding model to use", choices=["prot_t5", "esm2", "esm3", "af2_single", "af2_struct", "af2_combined"])
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
    
    config = parser.parse_args()
    
    run_name = "lr_" + str(config.learning_rate) + "_bs_" + str(config.train_batch_size) + "_ep_" + str(config.epoch) + "_wd_" + str(config.weight_decay) + "_nlayer_" + str(config.n_layer)
    dataset_name = config.selfies_path.split("/")[-1].split(".")[0]
    trainingscript = TrainingScript(hyperparameters_dict=config, 
                                    selfies_path=config.selfies_path, 
                                    pretrain_save_to=f"./saved_models/{run_name}",
                                    dataset_name = dataset_name)
    
    trainingscript.model_training()


              