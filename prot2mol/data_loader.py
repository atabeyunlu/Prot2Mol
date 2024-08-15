from torch.utils.data import Dataset
import torch
from collections import defaultdict
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, ligand_data, target_data, tokenizer, max_length=200):
        self.ligand_data = ligand_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_data_dict = self.preprocess_target_data(target_data)
        self.length = len(self.ligand_data)
        
    def preprocess_target_data(self, target_data):
        return {target_id: enc_state for target_id, enc_state in zip(target_data["Target_CHEMBL_ID"], target_data["encoder_hidden_states"])}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return self._get_batch(indices)
        elif isinstance(idx, list):
            return self._get_batch(idx)
        else:
            return self._get_single(idx)

    def _get_single(self, idx):
        ligand_row = self.ligand_data.iloc[idx]
        target_id = ligand_row["Target_CHEMBL_ID"]
        enc_state = self.target_data_dict[target_id]
        compound_selfies = ligand_row["Compound_SELFIES"]
        input_ids = self.get_input_ids(compound_selfies)
        return {"input_ids": input_ids, "encoder_hidden_states": enc_state}

    def _get_batch(self, indices):
        batch_size = len(indices)
        sample = self._get_single(indices[0])
        input_ids_shape = (batch_size,) + sample["input_ids"].shape
        encoder_hidden_states_shape = (batch_size,) + sample["encoder_hidden_states"].shape

        input_ids_batch = torch.empty(input_ids_shape, dtype=torch.long)
        encoder_hidden_states_batch = torch.empty(encoder_hidden_states_shape, dtype=torch.float)

        for i, idx in enumerate(indices):
            sample = self._get_single(idx)
            input_ids_batch[i] = sample["input_ids"]
            encoder_hidden_states_batch[i] = sample["encoder_hidden_states"]

        return {"input_ids": input_ids_batch, "encoder_hidden_states": encoder_hidden_states_batch}

    def get_input_ids(self, compound_selfies):
        return self.tokenizer(compound_selfies, 
                              max_length=self.max_length, 
                              padding="max_length", 
                              truncation=True, 
                              return_tensors="pt")["input_ids"].squeeze()


class MemoryEfficientDataset(Dataset):
    def __init__(self, ligand_data, target_data, tokenizer, max_length=200):
        self.ligand_data = ligand_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_data_dict = self.preprocess_target_data(target_data)
        self.length = len(self.ligand_data)
        
        # Create an index mapping for faster access
        self.target_id_to_index = defaultdict(list)
        for idx, target_id in enumerate(self.ligand_data["Target_CHEMBL_ID"]):
            self.target_id_to_index[target_id].append(idx)

    def preprocess_target_data(self, target_data):
        # Use numpy arrays instead of lists for encoder_hidden_states
        return {target_id: np.array(enc_state, dtype=np.float32) 
                for target_id, enc_state in zip(target_data["Target_CHEMBL_ID"], target_data["encoder_hidden_states"])}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return self._get_batch(indices)
        elif isinstance(idx, list):
            return self._get_batch(idx)
        else:
            return self._get_single(idx)

    def _get_single(self, idx):
        ligand_row = self.ligand_data.iloc[idx]
        target_id = ligand_row["Target_CHEMBL_ID"]
        enc_state = torch.from_numpy(self.target_data_dict[target_id])
        compound_selfies = ligand_row["Compound_SELFIES"]
        input_ids = self.get_input_ids(compound_selfies)
        return {"input_ids": input_ids, "encoder_hidden_states": enc_state}

    def _get_batch(self, indices):
        batch = [self._get_single(idx) for idx in indices]
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "encoder_hidden_states": torch.stack([item["encoder_hidden_states"] for item in batch])
        }

    def get_input_ids(self, compound_selfies):
        return self.tokenizer(
            compound_selfies,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"].squeeze()