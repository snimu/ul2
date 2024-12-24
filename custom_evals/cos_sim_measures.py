# /// script
# requires-python = "==3.12"
# dependencies = [
#   "numpy",
#   "torch",
#   "polars",
#   "tqdm",
#   "tiktoken",
#   "safetensors",
#   "huggingface_hub",
#   "transformers@git+https://github.com/huggingface/transformers.git",
#   "sentence-transformers",
#   "flash-attn",
#   "rich"
# ]
# ///

import argparse
import itertools

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import tiktoken
import polars as pl
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_and_tokenizer() -> tuple[AutoModel, AutoTokenizer]:
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    return model, tokenizer


def get_cos_sim_stats_query(
        query: str,
        completion: str,
        tokenizer: AutoTokenizer,
        encoding: tiktoken.Encoding,
        model: AutoModel,
        chunksize: int = 100,
) -> tuple[list[float], list[float]]:
    """
    Computes the cosine similarity between each chunk of the completion and the query;
    and between each chunk and the neighboring chunk.

    Returns:
        tuple[list[float], list[float]]: cos-sim to query; cos-sim to neighbor.
    """
    completion_tokens = encoding.encode_ordinary(completion, return_tensors="pt")
    chunks = []
    for chunk in itertools.batched(completion_tokens, chunksize):
        chunk_text = encoding.decode(chunk)
        chunk_tokens = tokenizer(chunk_text, return_tensors="pt")
        chunk_tokens = {k: v.to(DEVICE) for k, v in chunk_tokens.items()}
        with torch.no_grad():
            chunk_logits = model(**chunk_tokens, output_hidden_states=True).last_hidden_state
        chunks.append(chunk_logits)
    chunks = torch.cat(chunks, dim=0)

    query_tokens = encoding.encode_ordinary(query, return_tensors="pt")
    query_tokens = {k: v.to(DEVICE) for k, v in tokenizer(query_tokens).items()}
    with torch.no_grad():
        query_logits = model(**query_tokens, output_hidden_states=True).last_hidden_state
    query_logits = query_logits.mean(dim=0)

    cos_sim_to_query = F.cosine_similarity(query_logits, chunks, dim=0)
    cos_sim_neighbor = []
    for chunk in chunks:
        cos_sim_neighbor.append(F.cosine_similarity(query_logits, chunk, dim=0).squeeze().item())
    return cos_sim_to_query.tolist(), cos_sim_neighbor


def get_cos_sim_stats(
        dataset: pl.DataFrame,
        tokenizer: AutoTokenizer,
        encoding: tiktoken.Encoding,
        model: AutoModel,
        chunksize: int,
        savefile: str,
) -> pl.DataFrame:
    loop = tqdm(dataset.iter_rows(), total=dataset.height)
    df = pl.DataFrame(columns=["query", "completion", "cos_sim_to_query", "cos_sim_neighbor"])
    for row in loop:
        query = row["query"]
        completion = row["completion"]
        cos_sim_to_query, cos_sim_neighbor = get_cos_sim_stats_query(
            query, completion, tokenizer, encoding, model, chunksize
        )
        df = df.append(
            {
                "query": query,
                "completion": completion,
                "cos_sim_to_query": str(cos_sim_to_query),
                "cos_sim_neighbor": str(cos_sim_neighbor),
            }
        )
    df.write_csv(savefile)
    return df


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--chunksize", type=int, default=100)
    parser.add_argument("--savefile", type=str)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    dataset = pl.read_csv(args.dataset)
    model, tokenizer = get_model_and_tokenizer()
    encoding = tiktoken.get_encoding("gpt2")
    get_cos_sim_stats(dataset, tokenizer, encoding, model, args.chunksize, args.savefile)


if __name__ == "__main__":
    main()
