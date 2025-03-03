"""
Author: MD Touhidul Islam
Description: Fine-tuning BERT model for Twitter data (2021 Q1)
"""
# Standard library imports
import math
import os
import collections

# Deep learning and ML imports
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer, 
    AutoModel,
    DataCollatorForLanguageModeling,
    default_data_collator
)
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import get_full_repo_name
import numpy as np

# Local imports
from traning_f import training_function, training_function2

# Disable wandb sync and set cache directory
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "online"
os.environ['TRANSFORMERS_CACHE'] = '/pfs/work7/workspace/scratch/ma_mislam-mislam_twitterlm_thesis/cache'

# Model Configuration
model_checkpoint = "Twitter/twhin-bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Training Parameters
chunk_size = 128  # Size of text chunks for processing
percent_trian = 0.01  # Percentage of data to use for training
start_point = 1  # Starting week
end_point = 2  # Ending week
name = "twhin-bert-base_retrain_Cu"  # Model name prefix
year = "2019"  # Year of data
half = "_2"  # Half of the year
batch_size = 64  # Training batch size
wwm_probability = 0.2  # Whole word masking probability

# Load test dataset for evaluation
test_dateset = load_dataset('text', data_files = ["/pfs/work7/workspace/scratch/ma_mislam-mislam_twitterlm_thesis/Code/files/2023_11/2023_11_tweet_text"])
test_dateset = test_dateset['train'].shuffle(seed=42)
test_dateset = test_dateset.select(range(int(len(test_dateset)*percent_trian)))
test_dateset = test_dateset.train_test_split(test_size=0.2, seed=42)

# Setup accelerator and device
accelerator = Accelerator()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

print(f"'>>> BERT number of parameters: 110M'")

def process_data(filename):
    """
    Process the input text data for BERT fine-tuning.
    
    Args:
        filename: Path to the input text file
        
    Returns:
        lm_datasets: Processed dataset ready for language modeling
    """
    # Load and print dataset statistics
    dataset = load_dataset('text', data_files = [filename])
    print(f"'Dataset lenght: {len(dataset)}'")

    # Split dataset and sample examples
    dataset = dataset['train'].shuffle(seed=42)
    datatoPrint = dataset.select(range(10))
    for row in datatoPrint:
        print(f"\n'>>> Review: {row['text']}'")
    
    # Calculate sizes for train/test split
    size = len(dataset)*percent_trian
    print(f"'>>> Dataset size: {size}'")
    train_size = int(0.8 * size)
    test_size = int(0.2 * size)
    
    # Create train/test splits
    datasets = dataset.train_test_split(test_size=test_size,train_size=train_size, seed=42)
    datasets["eval"] = test_dateset["test"]
    
    def tokenize_function(examples):
        """Tokenize text and generate word IDs"""
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    # Tokenize datasets using parallel processing
    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Process sample data to verify chunk sizes
    tokenized_samples = tokenized_datasets["train"][:3]
    for idx, sample in enumerate(tokenized_samples["input_ids"]):
        print(f"'>>> Review {idx} length: {len(sample)}'")
    
    # Create chunks of specified size
    concatenated_examples = {
        k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
    }
    total_length = len(concatenated_examples["input_ids"])
    
    chunks = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }

    # Map texts into fixed-size chunks
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    print(f"'LM Dataset lenght: {len(lm_datasets)}'")
    
    return lm_datasets

def group_texts(examples):
    """
    Group texts into chunks of fixed size for efficient processing.
    
    Args:
        examples: Dictionary containing tokenized texts
        
    Returns:
        result: Dictionary with texts grouped into fixed-size chunks
    """
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # Ensure chunks are of equal size by truncating
    total_length = (total_length // chunk_size) * chunk_size
    
    # Split into chunks
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    
    # Add labels for masked language modeling
    result["labels"] = result["input_ids"].copy()
    return result

def whole_word_masking_data_collator(features):
    """
    Implement whole word masking for more natural masked language modeling.
    
    Args:
        features: List of features to process
        
    Returns:
        Processed features with whole words masked
    """
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Map tokens to words
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Apply random masking to whole words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        
        # Mask tokens and set labels
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)

def insert_random_mask(batch):
    """
    Insert random masks into a batch of data for masked language modeling.
    """
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

def get_eval_data(insert_random_mask, downsampled_dataset):
    """
    Prepare evaluation dataset with masked inputs.
    """
    downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
    
    eval_dataset = downsampled_dataset["test"].map(
        insert_random_mask,
        batched=True,
        remove_columns=downsampled_dataset["test"].column_names,
    )
    
    # Rename columns to standard format
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )
    
    return eval_dataset

# Main training loop
results = {}
size_list = []

# Process each week of data
for i in range(start_point, end_point):
    # Handle week numbering and previous week reference
    if i < 10:
        if i == 1:
            previous_Week = str(int(year)-1)+"_53"
        else:
            previous_Week = year+"_0"+ str(i-1)
    else:
        previous_Week = year+"_"+ str(i-1)

    # Load appropriate model checkpoint
    if previous_Week == "2019_22":
        model_checkpoint = "touhidulislam/twhin-bert-base_retrain"+previous_Week
    else:
        model_checkpoint = "touhidulislam/twhin-bert-base_retrain_Cu"+previous_Week
        
    print(Week)
    print(previous_Week)
    print(model_name)
    
    # Initialize model and optimizer
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    distilbert_num_parameters = model.num_parameters() / 1_000_000
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Format week number
    if i < 10:
        Week = year+"_0"+ str(i)
    else:
        Week = year+"_"+ str(i)
    model_name = name + Week
    
    # Process data for current week
    filename = "/pfs/work7/workspace/scratch/ma_mislam-mislam_twitterlm_thesis/Code/files/"+Week+"/"+Week+"_tweet_text"
    print("Filename: "+filename)
    lm_datasets = process_data(filename)
    size = len(lm_datasets["train"])
    size_list.append(size)
    
    # Prepare datasets for training
    downsampled_dataset = lm_datasets
    downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
    
    # Prepare test dataset
    test_dataset = downsampled_dataset["test"].map(
        insert_random_mask,
        batched=True,
        remove_columns=downsampled_dataset["test"].column_names,
    )
    test_dataset = test_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )
    
    # Prepare evaluation dataset
    eval_dataset = downsampled_dataset["eval"].map(
        insert_random_mask,
        batched=True,
        remove_columns=downsampled_dataset["eval"].column_names,
    )
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )
    
    # Train model and get results
    result = training_function(model, optimizer, tokenizer, chunk_size, batch_size, 
                             wwm_probability, data_collator, model_name, 
                             downsampled_dataset, test_dataset, eval_dataset, i)
    
    # Print training results
    print("Perplexity_difference: ", result["Perplexity_difference"])
    print("Perplexity_start: ", result["Perplexity_start"])
    print("Perplexity_end: ", result["Perplexity_end"])
    print("Perplexity_eval: ", result["Perplexity_eval"])
    
    results[Week] = result

# Visualization and logging with wandb
import wandb

# Prepare data for visualization
week = []
difference = []
proplexcity_start = []
proplexcity_end = []
proplexcity_eval = []
step = 1

# Collect results for each week
for key, value in results.items():
    week.append(value["week"])
    difference.append(value["Perplexity_difference"])
    proplexcity_start.append(value["Perplexity_start"])
    proplexcity_end.append(value["Perplexity_end"])
    proplexcity_eval.append(value["Perplexity_eval"])
    
    print("Week: ", key)
    print("Perplexity_difference: ", value["Perplexity_difference"])
    print("Perplexity_start: ", value["Perplexity_start"])
    print("Perplexity_end: ", value["Perplexity_end"])
    
    step += 1

# Initialize wandb run
run = wandb.init(project="Final_Result_"+year+half)

# Create data tables for visualization
data = [[x, y] for (x, y) in zip(week, difference)]
table = wandb.Table(data=data, columns=["Week", "Difference"])
Pdata = [[x, y] for (x, y) in zip(week, proplexcity_start)]
Ptable = wandb.Table(data=Pdata, columns=["Week", "Before"])
edata = [[x, y] for (x, y) in zip(week, proplexcity_end)]
etable = wandb.Table(data=edata, columns=["Week", "After"])
sdata = [[x, y] for (x, y) in zip(size_list, difference)]
stable = wandb.Table(data=sdata, columns=["Size", "Difference"])
evaldata = [[x, y] for (x, y) in zip(size_list, proplexcity_eval)]
evaltable = wandb.Table(data=evaldata, columns=["Size", "Evalucation"])

# Create final combined data table
Final_data = [[x, y,z,a,b,c] for (x, y,z,a,b,c) in zip(week, difference,size_list,proplexcity_start,proplexcity_end,proplexcity_eval)]
firal_table = wandb.Table(data=Final_data, columns=["Week", "Difference", "Size", "Before", "After","Evalucation"])

# Log visualizations to wandb
wandb.log(
    {
        "Difference_In_Proplexcity": wandb.plot.line(
            table, "Week", "Difference", title="Difference_In_Proplexcity"
        ),
        "Proplexcity_before": wandb.plot.line(
            Ptable, "Week", "Before", title="Proplexcity_Before"
        ),
        "Proplexcity_After": wandb.plot.line(
            etable, "Week", "After", title="Proplexcity_After"
        ),
        "Size_and_Differenct": wandb.plot.line(
            stable, "Size", "Difference", title="Size_and_Differenct"
        ),
        "Size_and_Differenct": wandb.plot.line(
            evaltable, "Week", "Evalucation", title="Week_and_Evalucation"
        ),
        "Size_and_Differenct": wandb.plot.scatter(
            firal_table, "Size", "Difference", title="Scatter_Size_and_Differenct"
        ),
    })