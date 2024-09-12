import pandas as pd
import argparse
import os

def train_val_test_split(dataset, test_ids, train_val_ratio=0.9, full_set=False):
    
    data = pd.read_csv(dataset)
    path = dataset.split(".csv")[0]
    if not os.path.exists(path):
        os.makedirs(path)    
    if full_set:
        train_val_data = data.sample(frac=1, random_state=42)
        train_data = train_val_data[:int(train_val_data.shape[0] * train_val_ratio)]
        val_data = train_val_data[int(train_val_data.shape[0] * train_val_ratio):]
        return train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    else:
        train_val_data = data[~data["Target_CHEMBL_ID"].isin([test_ids])]
        test_data = data[data["Target_CHEMBL_ID"].isin([test_ids])]
        train_val_data = train_val_data.sample(frac=1, random_state=42)
        train_data = train_val_data[:int(train_val_data.shape[0] * train_val_ratio)]
        val_data = train_val_data[int(train_val_data.shape[0] * train_val_ratio):]
        
        return train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--dataset", required=False, default="../data/papyrus/prot_comp_set_pchembl_6_protlen_1000_human_False.csv", help="Path of the SELFIES dataset.")
    parser.add_argument("--test_ids", required=False, default="CHEMBL4282", help="Protein ids for test set.")
    parser.add_argument("--full_set", required=False, default=False, help="Use full dataset.")
    config = parser.parse_args()
    
    path = config.dataset.split(".csv")[0]
    if not os.path.exists(path):
        os.makedirs(path)   
    
    if config.full_set:
        train_data, val_data = train_val_test_split(config.dataset, config.test_ids, full_set=True)
        train_data.to_csv(f"{path}_full_set/train.csv", index=False)
        val_data.to_csv(f"{path}_full_set/val.csv", index=False)
    else:
        train_data, val_data, test_data = train_val_test_split(config.dataset, config.test_ids)
        train_data.to_csv(f"{path}/train_wo_{config.test_ids}.csv", index=False)
        val_data.to_csv(f"{path}/val_wo_{config.test_ids}.csv", index=False)
        test_data.to_csv(f"{path}/test_{config.test_ids}.csv", index=False)