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
#   "git+https://github.com/huggingface/transformers.git",
#   "flash-attn",
# ]
# ///

import argparse
import json
from typing import Literal
from pathlib import Path
from dataclasses import dataclass

import torch
from tqdm import tqdm
import safetensors.torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import polars as pl
from huggingface_hub import hf_hub_download


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        logits = logits.float() # use tf32/fp32 for logits
        return logits


# HELPERS

def get_modelname_from_size(size: int, model: Literal["c", "r"]) -> str:
    if size == 1549:
        model_name_c = "snimu/p1549M_t100B_w1536_d52_h12_b480_s1024_i203450_clip0-15_seed0"
        model_name_r = "snimu/p1549M_t100B_w1536_d52_h12_b480_s1024_i203450_clip0-15_withMask_seed0"
    elif size == 2556:
        model_name_c = "snimu/p2556M_t100B_w1792_d64_h14_b480_s1024_i203450_clip0-15_seed1234"
        model_name_r = "snimu/p2556M_t100B_w1792_d64_h14_b480_s1024_i203450_clip0-15_withMask_seed1234"
    return model_name_c if model == "c" else model_name_r


def make_net_from_name(name: str) -> GPT:
    if "2556M" in name:
        return GPT(GPTConfig(
            vocab_size=50304,
            n_layer=64,
            n_head=14,
            n_embd=1792,
        )).to(DEVICE)
    if "1549M" in name:
        return GPT(GPTConfig(
            vocab_size=50304,
            n_layer=52,
            n_head=12,
            n_embd=1536,
        )).to(DEVICE)


def download_model(pretrained: str, cache_dir: str = ".") -> str:
    hf_hub_download(repo_id=pretrained, filename="model.safetensors", cache_dir=cache_dir)
    # Find model.safetensors in cache_dir (is in some subfolder)
    model_path = Path(cache_dir) / f"models--snimu--{pretrained.split('/')[1]}"
    model_path = list(model_path.glob("**/model.safetensors"))
    assert len(model_path) == 1, f"Expected exactly one model.safetensors file in cache_dir, got {model_path}"
    model_path = model_path[0]
    return str(model_path)


def fix_param_names(model_name: str):
    """Safetensors for some reason added an '_orig_mod' prefix to all param names 
    in the gpt2.muon runs
    -> remove it or model won't load"""
    if not ("1549M" in model_name or "2556M" in model_name):
        return

    root_path = Path(f"models--snimu--{model_name.split('/')[1]}")
    filepath = next(root_path.rglob('model.safetensors'))
    loaded = safetensors.torch.load_file(filepath)
    corrected = {k.replace("_orig_mod.", ""): v for k, v in loaded.items()}
    safetensors.torch.save_file(corrected, filepath)


def load_gpt(size: int) -> GPT:
    model_name = get_modelname_from_size(size, model="c")
    net = make_net_from_name(model_name)
    path  = download_model(model_name)
    fix_param_names(model_name)
    safetensors.torch.load_model(net, path, device=DEVICE)
    return net


@torch.no_grad()
def generate(
        net: GPT, 
        encoder: tiktoken.Encoding, 
        query: str, 
        min_gen_tokens: int = 128, 
        choose_nth_best: int = 1,
        temperature: float = 0.0,
        masking_rate: float = 0.0,
        mask: int = 50308,
        stepsize: int = 1,
) -> tuple[str, list[int], torch.Tensor, torch.Tensor]:
    # Encode the input tokens
    input_ids = encoder.encode_ordinary(query)
    input_ids = torch.tensor(input_ids, device=DEVICE, dtype=torch.int)
    if masking_rate > 0:
        mask_positions = torch.rand(len(input_ids)) < masking_rate
        input_ids[mask_positions] = mask
    input_ids = input_ids.unsqueeze(0)
    input_len = input_ids.shape[1]
    
    # Generate the output tokens
    output_str = []
    all_ids = input_ids
    gen_logits = []
    # Compare to encoded tokens of text because some of them get merged & I want to be able to compare to some target text (which also consists of merged tokens)
    while len(encoder.encode_ordinary("".join(output_str))) < min_gen_tokens:
        if stepsize == 1:
            inf_ids = all_ids
        else:
            inf_ids = torch.cat(
                [
                    all_ids, 
                    torch.empty((1, stepsize-1), device=DEVICE, dtype=torch.int).fill_(mask),
                ],
                dim=1,
            )

        logits: torch.Tensor = net(inf_ids)
        gen_logits.append(logits[:, -stepsize:, :50304])
        if temperature == 0.0:
            output_ids = logits[:, -stepsize:, :50304].topk(choose_nth_best, dim=-1).indices[:, :, -1].squeeze().tolist()  # ignore last token position, only decode valid token indices ( up to50304)
        else:
            logits = logits / temperature
            logits = F.softmax(logits, dim=-1)
            output_ids = torch.multinomial(logits[:, -stepsize:, :50304], 1).squeeze().tolist()  # ignore last token position, only decode valid token indices ( up to50304)
        output_ids = output_ids if isinstance(output_ids, list) else [output_ids]
        chars = encoder.decode(output_ids)
        output_str.extend(chars)
        all_ids = torch.cat([all_ids, torch.tensor(output_ids, device=DEVICE, dtype=torch.int).unsqueeze(0)], dim=1)

    # Get the logprops
    output_logprobs = F.softmax(torch.cat(gen_logits, dim=1), dim=-1).log()
    input_logprobs = F.softmax(net(input_ids)[:, :, :50304], dim=-1).log()
    all_logprobs = torch.cat([input_logprobs, output_logprobs], dim=1)
    
    # Get the output text
    output_text = "".join(output_str)
    return output_text, all_ids[:, input_len:].squeeze().tolist(), all_logprobs, output_logprobs.squeeze()


def get_dataset() -> torch.Tensor:
    with open("wikitext-custom.json", "r") as f:
        sentences = json.load(f)
    encoding = tiktoken.get_encoding("gpt2")
    tokenized = [encoding.encode_ordinary(sentence) for sentence in sentences]
    minlen = min(len(toks) for toks in tokenized)
    tokenized = [toks[:minlen] for toks in tokenized]
    return torch.tensor(tokenized, dtype=torch.int)


def generate_completions(
        net: GPT,
        encoder: tiktoken.Encoding,
        batchsize: int,
        num_completions: int,
        temperature: float,
        max_gen_tokens: int,
        stepsize: int,
        dataset: torch.Tensor,
        savefile: str,
) -> pl.DataFrame:
    loop = tqdm(range(0, len(dataset), batchsize))
    for i in loop:
        start = i
        stop = i+batchsize if i+batchsize < len(dataset) else None
        batch = dataset[start:stop].to(DEVICE)
        for completion_num in range(num_completions):
            loop.set_description(f"Generating completion {completion_num+1}/{num_completions}")
            completions = generate(
                net=net,
                encoder=encoder,
                query=batch,
                min_gen_tokens=max_gen_tokens,
                temperature=temperature,
                stepsize=stepsize,
            )[1].to("cpu")
            df = pl.DataFrame(
                {
                    "query": [encoder.decode(q) for q in batch],
                    "completion": [encoder.decode(c) for c in completions],
                }
            )
            if Path(savefile).exists():
                with open(savefile, "ab") as f:
                    df.write_csv(f, include_header=False)
            else:
                df.write_csv(savefile)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument(
        "--num-completions",
        type=int, default=1,
        help="The number of completions per query to generate. TYPE: int; DEFAULT: 1"
    )
    parser.add_argument(
        "--model_size",
        type=int, choices=[2556, 1549], default=2556,
        help="The model size to use. TYPE: int; DEFAULT: 2556"
    )
    parser.add_argument(
        "--max-gen-tokens",
        type=int, default=2048,
        help="The maximum number of tokens to generate. TYPE: int; DEFAULT: 2048"
    )
    parser.add_argument(
        "--stepsize",
        type=int, default=1,
        help="The number of steps to use for generation. TYPE: int; DEFAULT: 1"
    )
    parser.add_argument(
        "--temperature",
        type=float, default=0.0,
        help="The temperature to use for generation. TYPE: float; DEFAULT: 0.0"
    )
    parser.add_argument(
        "--masking-rate",
        type=float, default=0.0,
        help="The masking rate to use. TYPE: float; DEFAULT: 0.0"
    )
    parser.add_argument(
        "--savefile",
        type=str, default="results.csv",
        help="The CSV file to save the results to. TYPE: str; DEFAULT: results.csv"
    )

    return parser.parse_args()


def main():
    args = get_args()
    net = load_gpt(args.model_size)
    dataset = get_dataset()

    generate_completions(
        net=net,
        encoder=tiktoken.get_encoding("gpt2"),
        batchsize=args.batchsize,
        num_completions=args.num_completions,
        temperature=args.temperature,
        max_gen_tokens=args.max_gen_tokens,
        stepsize=args.stepsize,
        dataset=dataset,
        savefile=args.savefile,
    )
