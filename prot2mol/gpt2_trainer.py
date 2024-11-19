import re
import os
import torch
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
import h5py
from transformers import Trainer, T5Tokenizer, T5EncoderModel,DataCollatorForLanguageModeling
import sys
sys.path.insert(1, '../data_processing')


class GPT2_w_crs_attn_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("substep_1")
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False, legacy=True, clean_up_tokenization_spaces=True)
        print("substep_2")
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", torch_dtype=torch.float16).to(self.device)
        print("substep_3")
        #self.encoder_max_length = encoder_max_length
        print("sejeseje")

    def produce_prot_t5_embedding(self,batch,encoder_max_length):

        sequence_examples = " ".join(list(re.sub(r"[UZOB]", "X", batch))) 
        ids = self.tokenizer.encode_plus(sequence_examples, add_special_tokens=True, padding="max_length", max_length=encoder_max_length)
        input_ids = torch.tensor(ids['input_ids']).to(self.device).view(1,-1)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device).view(1,-1)
        with torch.no_grad():

            embedding_repr = self.model(input_ids=input_ids,attention_mask=attention_mask)
            
        return embedding_repr.last_hidden_state 

    def print_embedding_dimensions(self, prot_seq_raw, encoder_max_length=2000):
        last_hidden_state = self.produce_prot_t5_embedding(prot_seq_raw, encoder_max_length)
        print(f"Generated embedding dimensions: {last_hidden_state.shape}")
        
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        input_sequence = inputs["input_ids"]
        #last_hidden_state = inputs["encoder_hidden_states"]
        print(inputs)
        prot_seq_raw = inputs["protein_sequences"]
        ##last_hidden_state = self.produce_prot_t5_embedding(prot_seq_raw, self.encoder_max_length)
        last_hidden_state = self.produce_prot_t5_embedding(prot_seq_raw, 2000)

        print(f"Generated embedding dimensions: {last_hidden_state.shape}")

        outputs = model(input_ids=input_sequence, encoder_hidden_states=last_hidden_state, labels=input_sequence)
       
        return (outputs.loss, outputs) if return_outputs else outputs.loss

