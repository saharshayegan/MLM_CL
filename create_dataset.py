import torch
import os
import random
from datasets import load_dataset, concatenate_datasets
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

project_path = '/home/mila/s/sahar.omidishayegan/scratch/iran/nlp_course/project/'
# this is the path to the data folder. where all are data needed for the project should be stored
data_path = project_path + 'data/'
# this is the path to the tweets folder. where all the raw tweets are stored
raw_tweets_directory = data_path + 'tweets/'

NUM_FILES_TO_PICK = 500
MAX_NUM_TRAIN_SAMPLES = 842952
dataset_name = f"lm_dataset_{NUM_FILES_TO_PICK}_{MAX_NUM_TRAIN_SAMPLES}"
save_preprocessed_data_path = data_path+dataset_name+"/"
if not os.path.exists(save_preprocessed_data_path):
    os.makedirs(save_preprocessed_data_path)

with open(data_path+'file_names.txt') as f:
    file_names = f.readlines()
file_names = [str(x.strip()) for x in file_names]
file_paths = [os.path.join(raw_tweets_directory, file_name) for file_name in file_names]

random.seed(42)
random.shuffle(file_paths)
file_paths = file_paths[:NUM_FILES_TO_PICK]

datasets= []
counter = -1
for file in file_paths:
    counter+=1
    print(counter)
    # first make sure the file is not empty
    if os.path.getsize(file) == 0:
        print('file is empty')
        continue
    dataset = load_dataset("json", data_files=file, split="train")
    # drop fields that are not needed
    drop_fields = ['_id', 'uid', 'extText','tid', 'isTrunc', 'retUid', 'retText', 'retID', 'retretCount', 'retExtText', 'isRetTrunc']
    for field in drop_fields:
        try:
            dataset = dataset.remove_columns(field)
        except:
            continue
    datasets.append(dataset)


all_data = concatenate_datasets(datasets)
print('Number of tweets:', len(all_data))
del datasets
print('Deleted datasets')


print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
print('Tokenizer loaded')

# Making sure the tokenizer has the correct PAD token
if tokenizer.pad_token is None:
    tokenizer.pad_token = '[PAD]'

pad_token = tokenizer.pad_token
print(f"New padding token: {pad_token}")

if '[PAD]' not in tokenizer.get_vocab():
    tokenizer.add_tokens(['[PAD]'])
    tokenizer.pad_token = '[PAD]'

vocab = tokenizer.get_vocab()
pad_token = tokenizer.pad_token
print(f"New padding token: {pad_token}")
print(f"Vocabulary contains '[PAD]': {'[PAD]' in vocab}")

def preprocess_function(examples):
    result = tokenizer(["".join(x) for x in examples["text"]],max_length=128, padding='max_length',truncation=True)#,padding='max_length', truncation=True, max_length=512)#,padding=False, truncation=False)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

print('Tokenizing...')
tokenized_data = all_data.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=all_data.column_names
)
print('Tokenizing done')
del all_data

block_size = 128
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy() #added
    return result

lm_dataset = tokenized_data.map(group_texts, batched=True, num_proc=1)
print(lm_dataset)
del tokenized_data

train_size = MAX_NUM_TRAIN_SAMPLES
test_size = int(0.1 * train_size)

downsampled_dataset = lm_dataset.train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
print('Downsampled dataset:',downsampled_dataset)

# Save the downsampled dataset
downsampled_dataset.save_to_disk(save_preprocessed_data_path+"dataset")
# save the tokenizer
tokenizer.save_pretrained(save_preprocessed_data_path+"tokenizer")

print('Done')
print('The dataset is preprocessed and saved in the following path:', save_preprocessed_data_path)