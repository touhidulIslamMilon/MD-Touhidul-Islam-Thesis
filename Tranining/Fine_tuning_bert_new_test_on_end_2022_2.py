"""
Author: MD Touhidul Islam
Description: Fine-tuning BERT model for Twitter data (2022 Q2)
"""

import math
from transformers import AutoModelForMaskedLM
import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import os
from transformers import AutoTokenizer
import collections
import numpy as np
from transformers import default_data_collator
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import DataCollatorForLanguageModeling
from huggingface_hub import get_full_repo_name
from traning_f import training_function, training_function2
from torch.utils.data import DataLoader
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "online"
os.environ['TRANSFORMERS_CACHE'] = '/pfs/work7/workspace/scratch/ma_mislam-mislam_twitterlm_thesis/cache'

model_checkpoint = "Twitter/twhin-bert-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# tokenizer = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')
# model = AutoModel.from_pretrained('Twitter/twhin-bert-base')

chunk_size = 128
percent_trian = 1
year = "2022"
half= "_2"
name= "twhin-bert-base_retrain"
batch_size = 64
wwm_probability = 0.2
test_dateset=load_dataset('text', data_files = ["/pfs/work7/workspace/scratch/ma_mislam-mislam_twitterlm_thesis/Code/files/2023_11/2023_11_tweet_text"])
test_dateset= test_dateset['train'].shuffle(seed=42)
test_dateset = test_dateset.select(range(int(len(test_dateset)*percent_trian)))
test_dateset = test_dateset.train_test_split(test_size=0.2, seed=42)
accelerator = Accelerator()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")

text = "This is a great [MASK]."

# inputs = tokenizer(text, return_tensors="pt")
# token_logits = model(**inputs).logits
# # Find the location of [MASK] and extract its logits
# mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
# mask_token_logits = token_logits[0, mask_token_index, :]
# # Pick the [MASK] candidates with the highest logits
# top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

# for token in top_5_tokens:
#     print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

#open files

def process_data(filename):
    dataset = load_dataset('text', data_files = [filename])

    print(f"'Dataset lenght: {len(dataset)}'")

    #split dataset
    dataset= dataset['train'].shuffle(seed=42)
    datatoPrint = dataset.select(range(10))
    for row in datatoPrint:
        print(f"\n'>>> Review: {row['text']}'")
    size = len(dataset)*percent_trian
    print(f"'>>> Dataset size: {size}'")
    train_size= int(0.8 * size)
    test_size = int(0.2 * size)
    datasets = dataset.train_test_split(test_size=test_size,train_size=train_size, seed=42)
    print(f"'>>> Test size: {len(datasets['test'])}'")
    print(datasets)
    datasets["eval"] = test_dateset["test"]
    print(f"'>>> Test size after: {len(datasets['eval'])}'")
    print(datasets)

    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    # Use batched=True to activate fast multithreading!
    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    print("hape of tokanize Dataset: ")
    print(tokenized_datasets)
    print("Lenght of tokanizer: ")
    print(tokenizer.model_max_length)

   
    # Slicing produces a list of lists for each feature
    tokenized_samples = tokenized_datasets["train"][:3]

    for idx, sample in enumerate(tokenized_samples["input_ids"]):
        print(f"'>>> Review {idx} length: {len(sample)}'")
    concatenated_examples = {
        k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
    }
    total_length = len(concatenated_examples["input_ids"])

    print(f"'>>> Concatenated reviews length: {total_length}'")


    chunks = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }

    for chunk in chunks["input_ids"]:
        print(f"'>>> Chunk length: {len(chunk)}'")
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    print(f"'LM Dataset lenght: {len(lm_datasets)}'")
    
    # tokenizer.decode(lm_datasets["train"][1]["input_ids"])

    # samples = [lm_datasets["train"][i] for i in range(2)]
    # for sample in samples:
    #     _ = sample.pop("word_ids")

    # for chunk in data_collator(samples)["input_ids"]:
    #     print(f"\n'>>> {tokenizer.decode(chunk)}'")

    # samples = [lm_datasets["train"][i] for i in range(2)]
    # batch = whole_word_masking_data_collator(samples)

    # for chunk in batch["input_ids"]:
    #     print(f"\n'>>> {tokenizer.decode(chunk)}'")

    return lm_datasets
    


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result
def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
def get_eval_data(insert_random_mask, downsampled_dataset):
    downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])

    eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
    )
    eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
    )
    
    return eval_dataset

results = {}
size_list=[]
for i in range(30, 53): # change tzhe number here
    if i < 10:
        if i == 1:
            previous_Week = str(int(year)-1)+"_53"
        else:
            previous_Week = year+"_0"+ str(i-1)
    else:
        previous_Week = year+"_"+ str(i-1)

    if previous_Week == "2019_22":
        model_checkpoint = "touhidulislam/twhin-bert-base_retrain"+previous_Week
    else:
        model_checkpoint = "touhidulislam/twhin-bert-base_retrain_Cu"+previous_Week
    print(Week)
    print(previous_Week)
    print(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    distilbert_num_parameters = model.num_parameters() / 1_000_000
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    if i < 10:
        Week = year+"_0"+ str(i)
    else:
        Week = year+"_"+ str(i)
    model_name = name+Week  
    filename = "/pfs/work7/workspace/scratch/ma_mislam-mislam_twitterlm_thesis/Code/files/"+Week+"/"+Week+"_tweet_text"
        
    print("Filename: "+filename)
    lm_datasets = process_data(filename)
    size = len(lm_datasets["train"])
    size_list.append(size)
    downsampled_dataset = lm_datasets
    print("Downsize dataset shape: ")
    print(downsampled_dataset)

    downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
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
    
    result = training_function(model,optimizer,tokenizer, chunk_size, batch_size, wwm_probability, data_collator, model_name, downsampled_dataset, test_dataset,eval_dataset,i)
    print("Perplexity_difference: " )
    print(result["Perplexity_difference"])
    print("Perplexity_start: " )
    print(result["Perplexity_start"])
    print("Perplexity_end: " )
    print(result["Perplexity_end"])
    print("Perplexity_eval: " )
    print(result["Perplexity_eval"])

   
    results[Week]=result

    # training_function2(model, tokenizer, chunk_size, batch_size, wwm_probability, data_collator, model_name, downsampled_dataset)
import wandb
week=[]
difference=[]
proplexcity_start=[]
proplexcity_end = []
proplexcity_eval = []
step = 1
for key, value in results.items():

    week.append(value["week"])
    difference.append(value["Perplexity_difference"])
    proplexcity_start.append(value["Perplexity_start"])
    proplexcity_end.append(value["Perplexity_end"])
    proplexcity_eval.append(value["Perplexity_eval"])
    print("Week: ")
    print(key)
    print("Perplexity_difference: " )
    print(value["Perplexity_difference"])
    print("Perplexity_start: " )
    print(value["Perplexity_start"])
    print("Perplexity_end: " )
    print(value["Perplexity_end"])

    step += 1

run = wandb.init(project="Final_Result_"+year+half)  # change tzhe name here
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

Final_data = [[x, y,z,a,b,c] for (x, y,z,a,b,c) in zip(week, difference,size_list,proplexcity_start,proplexcity_end,proplexcity_eval)]
firal_table = wandb.Table(data=Final_data, columns=["Week", "Difference", "Size", "Before", "After","Evalucation"])
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
        # "Proplcity_base_on_week" : wandb.plot.scatter(
        #                xs=week, 
        #                ys=[difference, proplexcity_start, proplexcity_end],
        #                keys=["Difference_In_Proplexcity", "Proplexcity_Before", "Proplexcity_After"],
        #                title="Proplcity_base_on_week",
        #                xname="Week",
        #                yname="Proplexcity"),
        # "Proplcity_base_on_size" : wandb.plot.scatter(
        #                xs=size_list, 
        #                ys=[difference, proplexcity_start, proplexcity_end],
        #                keys=["Difference_In_Proplexcity", "Proplexcity_Before", "Proplexcity_After"],
        #                title="Proplcity_base_on_size",
        #                xname="Size",
        #                yname="Proplexcity"),
        # "Final_table": wandb.Table(
        #     Ptable, "Week", "Difference", "Size", "Before", "After", title="Final_table"
        # ),
    }
)


run.finish()