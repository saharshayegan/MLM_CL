import numpy as np
import pandas as pd
import random
import math
import os
import collections
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset, concatenate_datasets
from transformers import DataCollatorForLanguageModeling, default_data_collator
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import get_scheduler
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from datasets import load_from_disk
import wandb
import pickle

# os.environ["WANDB_DISABLED"] = "true"
run = wandb.init()
# sweep_id = wandb.run.sweep_id
# run_id = wandb.run.id
# config = wandb.config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
project_path = '/home/mila/s/sahar.omidishayegan/scratch/iran/nlp_course/project/'

run_name = 'Final_NoCL_fullData' # TODO: Change this for your model
runs_folder = project_path+'runs/'
run_path = runs_folder+run_name+'/'
model_save_path = run_path+'model/' 

data_path = project_path+'data/'
labeled_data_path = data_path+'df_monarchy_and_gov_top_25_rsrc_tweets.csv'

# make sure the path exists
if not os.path.exists(run_path):
    os.makedirs(run_path)

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

NUM_TRAIN_EPOCHS = 1
WITH_CL = False # TODO: Change this for your model
CL_LOSS_COEFF = 0.1044
CL_M = 1
BATCH_SIZE = 16
# WITH_CL = config.with_cl
# CL_LOSS_COEFF = config.cl_loss_coeff
# CL_M = config.cl_m
# BATCH_SIZE = config.batch_size
 
# These numbers should be the same as the ones used in the preprocessing script
NUM_FILES_TO_PICK = 500
MAX_NUM_TRAIN_SAMPLES = 842952 
dataset_name = f"lm_dataset_{NUM_FILES_TO_PICK}_{MAX_NUM_TRAIN_SAMPLES}"
preprocessed_data_path = data_path + dataset_name 

# create a config file at run_path and save the configurations
config = {
    "WITH_CL": WITH_CL, 
    "CL_LOSS_COEFF": CL_LOSS_COEFF, 
    "CL_M": CL_M, 
    "BATCH_SIZE": BATCH_SIZE, 
    "NUM_FILES_TO_PICK": NUM_FILES_TO_PICK, 
    "MAX_NUM_TRAIN_SAMPLES": MAX_NUM_TRAIN_SAMPLES, 
    "NUM_TRAIN_EPOCHS": NUM_TRAIN_EPOCHS,  
    "model_save_path": model_save_path, 
    "preprocessed_data_path": preprocessed_data_path
}
with open(run_path+'config.json', 'w') as f:
    f.write(str(config))

# print all the configurations
print(f"WITH_CL: {WITH_CL}")
print(f"CL_LOSS_COEFF: {CL_LOSS_COEFF}")
print(f"CL_M: {CL_M}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"NUM_FILES_TO_PICK: {NUM_FILES_TO_PICK}")
print(f"MAX_NUM_TRAIN_SAMPLES: {MAX_NUM_TRAIN_SAMPLES}")
print(f"NUM_TRAIN_EPOCHS: {NUM_TRAIN_EPOCHS}")
print(f"model_save_path: {model_save_path}")
print(f"preprocessed_data_path: {preprocessed_data_path}")


# read the dataset
downsampled_dataset = load_from_disk(preprocessed_data_path+"/dataset")
tokenizer = AutoTokenizer.from_pretrained(preprocessed_data_path+"/tokenizer")

# mask the dataset
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

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

train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=BATCH_SIZE, collate_fn=default_data_collator
)

# read the model and prepare the training
model = AutoModelForMaskedLM.from_pretrained('HooshvareLab/bert-fa-zwnj-base')
optimizer = AdamW(model.parameters(), lr=5e-5)
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# read the labeled data and split it
df = pd.read_csv(labeled_data_path)
df = df[df['weak_label'] !=3]
tweets = df['json'].tolist()
labels = df['weak_label'].tolist()
labels = [x - 1 for x in labels]

X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

# function to encode the text with MLM model
def encode_tweet(tweet):
    encoded_input = tokenizer(tweet, return_tensors='pt',padding=True,truncation=True).to(device)
    last_hidden_states = model(**encoded_input).hidden_states[-1]
    sentence_embedding = last_hidden_states.mean(dim=1)
    return sentence_embedding

# make sure the model returns hidden states
model.config.output_hidden_states = True 

train_losses = []
progress_bar = tqdm(range(num_training_steps))

for epoch in range(NUM_TRAIN_EPOCHS):
    print(f"Epoch {epoch}")
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        # the contrastive loss
        if WITH_CL:
            batch_loss = 0
            batch = [random.sample(range(len(X_train)), 2) for _ in range(BATCH_SIZE)]
            for i, j in batch:
                y = not y_train[i] == y_train[j]
                y = int(y)
                e_i = encode_tweet(X_train[i])
                e_j = encode_tweet(X_train[j])
                d = torch.dist(e_i, e_j)
                l = (1-y)*0.5*(d**2) + y*0.5*(max(0, CL_M-d)**2)
                batch_loss += l
            batch_loss /= len(batch)
            loss = (batch_loss * CL_LOSS_COEFF) + loss * (1 - CL_LOSS_COEFF)
        # end of contrastive loss
        wandb.log({'train_loss':loss.item()}, step=step)
        train_losses.append(loss.item())
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    plt.plot(train_losses)
    plt.savefig(run_path+'losses.png')
    
    # Evaluation
    model.eval()
    val_losses_list = []
    losses = []
    print('Evaluating')
    for step, batch in enumerate(eval_dataloader):
        batch.pop('masked_token_type_ids', None)
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        # print('val loss:', loss.item())
        losses.append(loss.item())
        val_losses_list.append(accelerator.gather(loss.repeat(BATCH_SIZE)))
    val_losses = torch.cat(val_losses_list)
    val_losses = val_losses[: len(eval_dataset)]

    try:
        perplexity = math.exp(torch.mean(val_losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
    model.val_perplexity = perplexity
    wandb.log({'v_perp':perplexity})
    with open(run_path+'val_perplexity.txt', 'w') as f:
        f.write(str(perplexity))
    with open(run_path+'val_losses.txt', 'w') as f:
        f.write(str(losses))

    # Save 
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(model_save_path, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(model_save_path)

    # pickle train_losses  at model_save_path
    with open(run_path+'train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
