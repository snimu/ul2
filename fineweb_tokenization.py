import os
import tiktoken
from datasets import load_dataset, Dataset, Features, Sequence, Value
import argparse
from multiprocessing import Pool, cpu_count
import itertools


def tokenize_text(args):
    text, tokenizer_name = args
    encoder = tiktoken.get_encoding(tokenizer_name)
    tokens = encoder.encode_ordinary(text)
    return tokens, len(tokens)

def process_batch(batch, tokenizer_name, num_processes):
    with Pool(num_processes) as pool:
        results = list(pool.imap(tokenize_text, zip([item['text'] for item in batch], itertools.repeat(tokenizer_name))))
    
    return [
        {**{k: item[k] for k in item.keys() if k != "text"}, "tokens": tokens, "token_count": count}
        for item, (tokens, count) in zip(batch, results)
    ]

def data_generator(dataset, tokenizer_name, batch_size=1000, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()

    batch = []
    for item in dataset:
        batch.append(item)
        if len(batch) >= batch_size:
            yield from process_batch(batch, tokenizer_name, num_processes)
            batch = []
    if batch:
        yield from process_batch(batch, tokenizer_name, num_processes)

def create_and_push_dataset(tokenizer_name="gpt2", num_processes=None, validation_size=100, seed=42):
    if num_processes is None:
        num_processes = cpu_count()

    print(f"Using {num_processes} CPU cores for parallel processing.")

    # Load the Hugging Face token from environment variable
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set. Please set it with your Hugging Face token.")

    # Load the original dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    # Shuffle the dataset
    shuffled_fw = fw.shuffle(seed=seed, buffer_size=10000)

    # Manually create the new features
    new_features = Features({
        'tokens': Sequence(Value('int64')),
        'token_count': Value('int64'),
        **{k: v for k, v in fw.features.items() if k not in ['text', 'token_count']}
    })

    # Create validation dataset first
    validation_dataset = Dataset.from_generator(
        lambda: itertools.islice(data_generator(shuffled_fw, tokenizer_name, num_processes=num_processes), validation_size),
        features=new_features,
    )

    # Push the validation dataset to the Hugging Face Hub
    validation_dataset.push_to_hub(
        "snimu/fineweb-edu-sample-10BT-tiktokenized",
        config_name=tokenizer_name,
        split="val",  # Changed from "validation" to "val"
        token=hf_token
    )

    print(f"Validation dataset uploaded successfully. Size: {len(validation_dataset)}")

    # Create train dataset
    train_dataset = Dataset.from_generator(
        lambda: data_generator(shuffled_fw, tokenizer_name, num_processes=num_processes),
        features=new_features,
    )

    # Push the train dataset to the Hugging Face Hub
    train_dataset.push_to_hub(
        "snimu/fineweb-edu-sample-10BT-tiktokenized",
        config_name=tokenizer_name,
        split="train",
        token=hf_token
    )

    print(f"Train dataset uploaded successfully. Size: {len(train_dataset)}")
    print(f"Datasets uploaded successfully with tokenizer: {tokenizer_name}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and push tokenized dataset to Hugging Face")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Name of the tokenizer to use")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes to use (default: number of CPU cores)")
    parser.add_argument("--validation-size", type=int, default=100, help="Size of the validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    create_and_push_dataset(args.tokenizer, args.processes, args.validation_size, args.seed)
