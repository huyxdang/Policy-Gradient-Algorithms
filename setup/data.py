"""
This module does three things regarding the GSM8K dataset: 
1. Load the GSM8K dataset
2. Sample the dataset according to a given ratio
3. Save the sampled dataset to a JSON file
"""

from datasets import load_dataset
import random

def load_data_gsm8k():
    dataset_train = load_dataset("openai/gsm8k", "main", split="train")
    dataset_test = load_dataset("openai/gsm8k", "main", split="test")
    return dataset_train, dataset_test

def sample_data_gsm8k(ratio: float):
    dataset_train, dataset_test = load_data_gsm8k()
    
    # Get random indices for sampling
    train_size = int(len(dataset_train) * ratio)
    test_size = int(len(dataset_test) * ratio)

    train_indices = random.sample(range(len(dataset_train)), train_size)
    test_indices = random.sample(range(len(dataset_test)), test_size)
    
    # Select samples using indices
    dataset_train = dataset_train.select(train_indices)
    dataset_test = dataset_test.select(test_indices)
    
    return dataset_train, dataset_test

def save_gsm8k_data(dataset_train, dataset_test, path: str):
    dataset_train.to_json(path + "train.json")
    dataset_test.to_json(path + "test.json")

if __name__ == "__main__":
    dataset_train, dataset_test = sample_data_gsm8k(0.3)
    save_gsm8k_data(dataset_train, dataset_test, "data/gsm8k/")
    print("Training Size:", len(dataset_train))
    print("Test Size:", len(dataset_test))