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
from data_loader import CustomDataset, CustomEffDataset
from gpt2_trainer import GPT2_w_crs_attn_Trainer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"


class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        if "protein_sequences" in features[0]:
            batch["protein_sequences"] = [feature.get("protein_sequences", "") for feature in features]
        return batch

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

        self.TRAIN_BATCH_SIZE = int(config.train_batch_size)
        self.VALID_BATCH_SIZE = int(config.valid_batch_size)
        self.TRAIN_EPOCHS = int(config.epoch)
        self.LEARNING_RATE = float(config.learning_rate)
        self.WEIGHT_DECAY = float(config.weight_decay)
        self.N_LAYER = int(config.n_layer)
        self.N_HEAD = int(config.n_head)
        self.max_mol_len = int(config.max_mol_len)
        self.N_EMB = int(config.n_emb)
        print("Load training vectors...\n")
        #self.training_vec = np.load("../data/train_vecs.npy") # write a script for this
        self.training_vec = None
        print("Load train validation test data...\n")
        if config.full_set:
            print("Loading full dataset...\n")
            self.train_data, self.eval_data = train_val_test.train_val_test_split(config.selfies_path, config.prot_ID, full_set=config.full_set)
            #here selfies path actually includes both protein seq and compound selfies (also chembl ids)
        else:
            self.train_data, self.eval_data, self.test_data = train_val_test.train_val_test_split(config.selfies_path, config.prot_ID)

        if "af2" in self.prot_emb_model:
            self.prot_emb_model_path = f"../data/prot_embed/{self.prot_emb_model}/FoldedPapyrus_4581_v01/embeddings.h5"
        else:
            self.prot_emb_model_path = f"../data/prot_embed/{self.prot_emb_model}/{dataset_name}/embeddings_fp16.h5"
        print("Load protein embeddings...\n")


        self.tokenizer = BartTokenizer.from_pretrained("zjunlp/MolGen-large", padding_side="left")    
        alphabet =  list(sf.get_alphabet_from_selfies(list(self.train_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(alphabet)
        alphabet =  list(sf.get_alphabet_from_selfies(list(self.eval_data.Compound_SELFIES)))
        self.tokenizer.add_tokens(alphabet)
        del alphabet

        print("Load GPT2 model...\n")
        print("Configuration:")
        print(f"N_EMB: {self.N_EMB}, N_HEAD: {self.N_HEAD}, N_LAYER: {self.N_LAYER}")
        print(f"Vocab Size: {len(self.tokenizer.added_tokens_decoder)}, BOS Token ID: {self.tokenizer.bos_token_id}, EOS Token ID: {self.tokenizer.eos_token_id}")

        self.configuration = GPT2Config(add_cross_attention=True, is_decoder=True,
                                        n_embd=self.N_EMB, n_head=self.N_HEAD, vocab_size=len(self.tokenizer.added_tokens_decoder), 
                                        n_positions=256, n_layer=self.N_LAYER, bos_token_id=self.tokenizer.bos_token_id,
                                        eos_token_id=self.tokenizer.eos_token_id)
        self.model = GPT2LMHeadModel(self.configuration)

        print("Model parameter count:", self.model.num_parameters())

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return metrics_calculation(predictions=decoded_preds, references=decoded_labels, train_data=self.train_data, train_vec=self.training_vec)

    def model_training(self):
        #data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        data_collator = CustomDataCollator(tokenizer=self.tokenizer, mlm=False) 
        print("Build train datasets...\n")
        train_dataset = CustomDataset(ligand_data=self.train_data, target_data=self.train_data, tokenizer=self.tokenizer, max_length=self.max_mol_len)
        print("Build eval datasets...\n")
        eval_dataset = CustomDataset(ligand_data=self.eval_data, target_data=self.eval_data, tokenizer=self.tokenizer, max_length=self.max_mol_len)
        print("Build test datasets...\n")

        print("step_1")
        self.model.train()
        print("step_2")
        
        # Check if a CUDA GPU is available for fp16 precision
        use_fp16 = torch.cuda.is_available()
        print("step_3")
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
            dataloader_num_workers=10,
            fp16=use_fp16,  # Enable fp16 if CUDA is available
            ddp_find_unused_parameters=False)
        print("step_4")
        trainer = GPT2_w_crs_attn_Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
            #encoder_max_length=config.encoder_max_length
            )
        print("step_5")
        #trainer.print_embedding_dimensions(self.train_data.iloc[0]["Target_FASTA"])
        #you need to add the max size as argument as well 
        print("step_6")
        trainer.args._n_gpu = 1
        print("build pretrain trainer with on device:", training_args.device, "with n gpus:", training_args.n_gpu)
        print("train_Data:")
        print(self.train_data)

        print("train_dataset")
        print(train_dataset._get_single(0))
        trainer.train()
        print("training finished.")

        eval_results = trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        trainer.save_model(self.pretrain_save_to) 

def main(config):
    dataset_name = config.selfies_path.split("/")[-1].split(".")[0]
    if config.full_set:
        run_name =  f"""lr_{str(config.learning_rate)}_bs_{str(config.train_batch_size)}_ep_{str(config.epoch)}_wd_{str(config.weight_decay)}_nlayer_{str(config.n_layer)}_nhead_{str(config.n_head)}_prot_{config.prot_emb_model}_dataset_{dataset_name}_fp16"""
    else:
        run_name =  f"""lr_{str(config.learning_rate)}_bs_{str(config.train_batch_size)}_ep_{str(config.epoch)}_wd_{str(config.weight_decay)}_nlayer_{str(config.n_layer)}_nhead_{str(config.n_head)}_prot_{config.prot_emb_model}_dataset_{dataset_name}_testID_{config.prot_ID}"""

    trainingscript = TrainingScript(config=config, 
                                    selfies_path=config.selfies_path, 
                                    pretrain_save_to=f"../saved_models/{run_name}",
                                    dataset_name=dataset_name,
                                    run_name=run_name)

    trainingscript.model_training()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--selfies_path", required=False, default="../data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False/af2_filtered.csv", help="Path of the SELFIES dataset.")
    parser.add_argument("--prot_emb_model", required=False, default="af2_combined", help="Which protein embedding model to use", choices=["prot_t5", "esm2", "esm3", "af2_single", "af2_struct", "af2_combined"])
    parser.add_argument("--prot_ID", required=False, default="CHEMBL4282")
    parser.add_argument("--full_set", required=False, default=False, help="Use full dataset.")
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
    parser.add_argument("--n_emb", default=768) # prot_t5=1024, esm2=1280, esm3=1024, af2_single=384, af2_struct=384, af2_combined=768
    #parser.add_argument("--encoder_max_length", default=2000, type=int, help="Max length of the encoder embeddings")

    config = parser.parse_args()
    main(config)


              
