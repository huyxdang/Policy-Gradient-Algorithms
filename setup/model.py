"""
This module contains functions to load, save, and fine-tune the model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
    return model    

def load_tokenzier(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def load_model_and_tokenizer(model_name: str):
    " Convenient function to both load and tokenize model"
    model = load_model(model_name)
    tokenizer = load_tokenzier(model_name)
    return model, tokenizer

def save_model(model, tokenizer, save_path: str):
    " Save fine-tuned model and tokenizer"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def load_fine_tuned_model(model_path:str):
    "Load a fine-tuned model from a given path"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
