from torch.utils.data import Dataset
import torch
import h5py

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

class CustomEffDataset(Dataset):
    def __init__(self, ligand_data, target_data, tokenizer, max_length=200):
        self.ligand_data = ligand_data
        self.h5_file_path = target_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.length = len(self.ligand_data)

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            target_ids = h5_file['Target_CHEMBL_ID'][:]
            
        self.target_id_to_index = {target_id.decode('utf-8'): i for i, target_id in enumerate(target_ids)}
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
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            enc_state = self.get_encoding(h5_file, target_id)
        
        compound_selfies = ligand_row["Compound_SELFIES"]
        input_ids = self.get_input_ids(compound_selfies)
        return {"input_ids": input_ids, "encoder_hidden_states": enc_state}
    
    def get_encoding(self, h5_file, target_id):
        index = self.target_id_to_index.get(target_id)
        if index is not None:
            return h5_file['encoder_hidden_states'][index]
        else:
            raise KeyError(f"Target_CHEMBL_ID '{target_id}' not found in the dataset.")
        
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