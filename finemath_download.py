# /// script
# requires-python = "==3.12"
# dependencies = [
#   "numpy",
#   "tiktoken",
#   "tqdm",
#   "datasets",
#   "transformers",
# ]
# ///

"""
Taken & modiefied from https://github.com/karpathy/llm.c/blob/master/dev/data/fineweb.py
and from https://github.com/karpathy/llm.c/blob/master/dev/data/data_common.py

FineMath dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceTB/finemath
"""
import os
import argparse
import requests
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

"""
Common utilities for the datasets
"""

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


HEADERS_INFO = {
    "gpt-2": {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint16,
    },
    "llama-3": {
        "magic": 20240801,
        "version": 7,
        "token_dtype": np.uint32,
    },
}

def write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 (gpt-2) or uint32 (llama)
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    assert model_desc in ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = HEADERS_INFO[model_desc]
    # construct the header
    header = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    header[0] = info["magic"]
    header[1] = info["version"]
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    num_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def write_evalfile(filename, datas):
    """
    Saves eval data as a .bin file, for reading in C.
    Used for multiple-choice style evals, e.g. HellaSwag and MMLU
    - First comes a header with 256 int32s
    - The examples follow, each example is a stream of uint16_t:
        - <START_EXAMPLE> delimiter of 2**16-1, i.e. 65,535
        - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
        - <EXAMPLE_INDEX>, the index of the example in the dataset
        - <LABEL>, the index of the correct completion
        - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
        - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
        - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
    """
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240522 # magic
    header[1] = 1 # version
    header[2] = len(datas) # number of examples
    header[3] = 0 # reserved for longest_example_bytes, fill in later
    # now write the individual examples
    longest_example_bytes = 0 # in units of uint16s
    full_stream = [] # the stream of uint16s, we'll write a single time at the end
    assert len(datas) < 2**16, "too many examples?"
    for idx, data in enumerate(datas):
        stream = []
        # header of the example
        stream.append(2**16-1) # <START_EXAMPLE>
        stream.append(0) # <EXAMPLE_BYTES> (fill in later)
        stream.append(idx) # <EXAMPLE_INDEX>
        stream.append(data["label"]) # <LABEL>
        ending_tokens = data["ending_tokens"]
        assert len(ending_tokens) == 4, "expected 4 completions for now? can relax later"
        stream.append(len(ending_tokens)) # <NUM_COMPLETIONS>
        # the (shared) context tokens
        ctx_tokens = data["ctx_tokens"]
        assert all(0 <= t < 2**16-1 for t in ctx_tokens), "bad context token"
        stream.append(len(ctx_tokens))
        stream.extend(ctx_tokens)
        # the completion tokens
        for end_tokens in ending_tokens:
            assert all(0 <= t < 2**16-1 for t in end_tokens), "bad completion token"
            stream.append(len(end_tokens))
            stream.extend(end_tokens)
        # write to full stream
        nbytes = len(stream)*2 # 2 bytes per uint16
        assert nbytes < 2**16, "example too large?"
        stream[1] = nbytes # fill in the <EXAMPLE_BYTES> field
        longest_example_bytes = max(longest_example_bytes, nbytes)
        full_stream.extend(stream)
    # construct the numpy array
    stream_np = np.array(full_stream, dtype=np.uint16)
    # fill in the longest_example field
    assert 0 < longest_example_bytes < 2**16, "bad longest_example"
    header[3] = longest_example_bytes
    # write to file (for HellaSwag val this is 10,042 examples, 3.6MB file)
    print(f"writing {len(datas):,} examples to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(stream_np.tobytes())


# ------------------------------------------


parser = argparse.ArgumentParser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each data shard in the output .bin files, in tokens")
args = parser.parse_args()

local_dir, remote_name = "finemath-4plus", "finemath-4plus"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceTB/finemath", name=remote_name, split="train")
name = "finemath"


def tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint = tokens_np.astype(np.uint16)
    return tokens_np_uint


# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize_gpt2, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np.tolist(), args.model_desc)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_tokens_np[:token_count]).tolist(), args.model_desc)
