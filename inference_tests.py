
import os
from dotenv import load_dotenv
import argparse
from typing import Any, Literal
from pathlib import Path
import gzip
import json
from tqdm import tqdm

import torch
import safetensors.torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import tiktoken
from huggingface_hub import hf_hub_download
from rich import print
import Levenshtein
from pydantic import BaseModel
import openai
import instructor


load_dotenv()


class Preference(BaseModel):
    """A preference for one of two completions."""
    reflection: str
    better_completion: Literal["completion1", "completion2"]


client = instructor.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
def choose_preference(
        base_text: str, 
        completion1: str, 
        completion2: str, 
        openai_model: str = "gpt-4o-mini",
) -> int:
    preference = client.chat.completions.create(
        model=openai_model,
        response_model=Preference,
        temperature=1.0,
        messages=[
            {
                "role": "system", 
                "content": (
                    "Given a base text, and two alternative completions to it, choose the better completion. "
                    "Better means less repetition, a more grammatically correct and more coherent text, "
                    "and more similarity to the base text. "
                    "Reflect on your answer before making a decision."
                )
            },
            {"role": "user", "content": f"{base_text=}\n\{completion1=}\n\{completion2=}"},
        ],
        
    )
    return 0 if preference.better_completion == "completion1" else 1


max_seq_len = 4096


with torch.no_grad():
    # Create the base arrays for the learnable linear positional bias. This helps save some memory consumption & processing time
    bias_range                    = torch.arange(-max_seq_len+1, 1).to("cuda", dtype=torch.bfloat16)
    position_bias_base            = bias_range.unsqueeze(0) - bias_range.unsqueeze(1)
    negative_infinity_matrix_base = torch.empty_like(position_bias_base).fill_(-float("inf"))
    causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), device="cuda", dtype=torch.bool))


class LatentAttentionBlock(nn.Module):
    """ Efficient fused latent-space attention block. Linear keys and queries, nonlinear values."""
    def __init__(self, width: int,  depth: int, linear_value: bool, num_heads: int):
        super().__init__()
        # Layer dim parameters. Play around with these, there's likely some undiscovered stuff still!
        self.dim        = width
        self.qk_dim     = self.dim//8
        self.v_dim      = width
        self.expand_dim = width * 2
        self.linear_value = linear_value 
        self.num_heads = num_heads

        # Main layer weights
        self.norm    = nn.LayerNorm(self.dim, bias=False)
        self.expand  = nn.Parameter(.5 * 1./width**.5 * 1./2                              * torch.randn(2*self.qk_dim+2*self.expand_dim, self.dim))
        self.project = nn.Parameter(1. * 1./width**.5 * 1./2 * 1./depth * torch.randn((self.dim, self.expand_dim)))

        # Learnable linear positional encodings. Similar to but different than https://arxiv.org/abs/2108.12409
        # Has a high lr mult applied to it so that each layer can learn its own attention scale.
        self.position_bias_mult = nn.Parameter(torch.tensor(1., device='cuda'))

    def forward(self, x, attn_mask: torch.Tensor):
        residual = x

        # Make additive attention mask, scaled by a learned mult for the position bias (lets us learn dynamic attention ranges per layer as needed)
        attn_mask_with_positional_bias = torch.where(attn_mask, F.softplus(self.position_bias_mult) * position_bias_base[:x.shape[1], :x.shape[1]], negative_infinity_matrix_base[:x.shape[1], :x.shape[1]])
        
        # Shared LayerNorm for linear layers and attention
        x = self.norm(x)

        # Fused into one kernel for memory+speed/etc
        query, key, linear, pre_gelu = F.linear(x, self.expand).split((self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), dim=-1)

        # Compute GeGLU (one portion of the channels this will stay locally, another will become the nonlinear value for attention)
        geglu = linear * F.gelu(pre_gelu)

        # Partition between the input values and the v dim values
        if self.linear_value:
            geglu_local, _ = geglu.split((self.expand_dim-self.v_dim, self.v_dim), -1)
            _, geglu_attention_value = pre_gelu.split((self.expand_dim-self.v_dim, self.v_dim), -1)
        else:
            geglu_local, geglu_attention_value = geglu.split((self.expand_dim-self.v_dim, self.v_dim), -1)

        if self.num_heads > 1:
            if len(attn_mask_with_positional_bias.shape) == 3:
                attn_mask_with_positional_bias = einops.repeat(attn_mask_with_positional_bias, 'b s1 s2 -> b h s1 s2', h=self.num_heads)
            query, key, geglu_local, geglu_attention_value = map(lambda x: einops.rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), (query, key, geglu_local, geglu_attention_value))

        # Compute attention. Something to note is that there are no attention heads here. This seemed to work a bit better, maybe due to not needing memory `.contiguous()` calls or similar
        attention = F.scaled_dot_product_attention(query, key, geglu_attention_value, attn_mask=attn_mask_with_positional_bias)

        if self.num_heads > 1:
            attention = einops.rearrange(attention, 'b h n d -> b n (h d)')
            geglu_local = einops.rearrange(geglu_local, 'b h n d -> b n (h d)')

        # Output linear layer
        out = F.linear(torch.cat([geglu_local, attention], dim=-1), self.project)

        # Add to residual
        x = residual + out

        return x


#############################################
#            Network Definition             #
#############################################

# This may seem like an odd way to define a network, but it's a bit easier to hack into/make quick changes than other methods
class SpeedyLangNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict

    def forward(self, x, mode: Literal['causal', 'noncausal', 'mixed'] = "causal"):
        # Look up the input embeddings from the input tokens
        attn_mask = causal_mask[:x.shape[1], :x.shape[1]]
        x = self.net_dict['embedding'](x)
        for attn_block in self.net_dict['attn_layers']:
            x = attn_block(x, attn_mask=attn_mask) # note: residuals are included in the block definitions for these layers
        x = self.net_dict['norm'](x)
        x = self.net_dict['outputs'](x)
        return x
    

def make_attn(settings: dict[str, Any]):
    # You can parametrically change anything you want about the attn blocks here
    return LatentAttentionBlock(
        settings['width'], settings['depth'], settings['linear_value'], settings['num_heads']
    )


def make_net(settings: dict[str, Any]):
    total_num_tokens = 50310
    network_dict = nn.ModuleDict({
        'embedding': nn.Embedding(total_num_tokens, settings['width'], scale_grad_by_freq=True),
        'attn_layers': nn.ModuleList([make_attn(settings) for _ in range(settings['depth'])]),
        'norm': nn.LayerNorm(settings['width'], bias=False),
        'outputs': nn.Linear(settings['width'], total_num_tokens, bias=False),
    })
    net = SpeedyLangNet(network_dict)
    net = net.to("cuda", dtype=torch.bfloat16)
    net.eval()

    # Initialize the embedding and output matrixes, with weights scaled based upon the dimensionality of the network.
    torch.nn.init.normal_(net.net_dict['embedding'].weight.data, std=.25*1./settings['width']**.5)
    torch.nn.init.normal_(net.net_dict['outputs']  .weight.data, std=.5 *1./settings['width']**.5)

    return net


def make_net_from_name(name: str) -> SpeedyLangNet:
    if "46M" in name:
        depth, width =  8, 384
    elif "240M" in name:
        depth, width = 21, 1024
    elif "773M" in name:
        depth, width = 35, 1664
    elif "1300M" in name:
        depth, width = 43, 2048
    else:
        raise ValueError(f"Unknown pretrained model {name}")
    
    num_heads = int(name.split("-")[-2].split("head")[0])
    
    return make_net({
        'depth': depth,
        'width': width,
        'linear_value': False,
        'num_heads': num_heads,
    })


def download_model(pretrained: str, cache_dir: str = ".") -> str:
    hf_hub_download(repo_id=pretrained, filename="model.safetensors", cache_dir=cache_dir)
    # Find model.safetensors in cache_dir (is in some subfolder)
    model_path = Path(cache_dir) / f"models--snimu--{pretrained.split('/')[1]}"
    model_path = list(model_path.glob("**/model.safetensors"))
    assert len(model_path) == 1, f"Expected exactly one model.safetensors file in cache_dir, got {model_path}"
    model_path = model_path[0]
    return str(model_path)


@torch.no_grad()
def generate(
        net: SpeedyLangNet, 
        encoder: tiktoken.Encoding, 
        query: str, 
        max_gen_tokens: int = 128, 
        until: list[str] | None = None,
        choose_nth_best: int = 1,
        masking_rate: float = 0.0,
        mask: int = 50308,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    # Encode the input tokens
    input_ids = encoder.encode_ordinary(query)
    input_ids = torch.tensor(input_ids, device="cuda", dtype=torch.int)
    if masking_rate > 0:
        mask_positions = torch.rand(len(input_ids)) < masking_rate
        input_ids[mask_positions] = mask
    input_ids = input_ids.unsqueeze(0)
    input_len = input_ids.shape[1]
    
    # Generate the output tokens
    output_str = []
    all_ids = input_ids
    while len(encoder.encode_ordinary("".join(output_str))) < max_gen_tokens:
        logits: torch.Tensor = net(all_ids)
        output_id = logits[:, -1, :50304].topk(choose_nth_best, dim=-1).indices[:, -1].item()  # ignore last token position, only decode valid token indices ( up to50304)
        char = encoder.decode([output_id])
        output_str.append(char)
        all_ids = torch.cat([all_ids, torch.tensor([output_id], device="cuda", dtype=torch.int).unsqueeze(0)], dim=1)
        if until and char in until:
            break

    # Get the logprops
    logprobs = F.softmax(net(all_ids), dim=-1).log()
    
    # Get the output text
    output_text = "".join(output_str)
    return output_text, logprobs, logprobs.squeeze()[input_len:]


@torch.no_grad()
def generate_with_mask(
    net: SpeedyLangNet, 
    encoder: tiktoken.Encoding, 
    query: str, 
    max_gen_tokens: int = 128, 
    mask: int = 50308,
    choose_nth_best: int = 1,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    # Encode the input tokens
    input_ids = encoder.encode_ordinary(query)
    input_ids = torch.tensor(input_ids, device="cuda", dtype=torch.int).unsqueeze(0)
    input_len = input_ids.shape[1]

    all_ids = torch.cat(
        [
            input_ids, 
            torch.empty(
                (1, max_gen_tokens), 
                device="cuda", 
                dtype=torch.int,
            ).fill_(mask),
        ], 
        dim=1,
    )
    logits: torch.Tensor = net(all_ids)
    logprobs = F.log_softmax(logits, dim=-1)
    outputs = logits[:, input_len:, :50304].topk(choose_nth_best, dim=-1).indices[:, :, -1]
    outputs = outputs.squeeze().tolist()
    output_text = encoder.decode(outputs)
    
    return output_text, logprobs, logprobs.squeeze()[input_len:]


def calc_ratio_compression(completion: str, full: str) -> tuple[float, float]:
    completion_size = len(completion.encode())
    full_size = len(full.encode())
    compressed_completion_size = len(gzip.compress(completion.encode()))
    compressed_full_size = len(gzip.compress(full.encode()))
    return compressed_completion_size / completion_size, compressed_full_size / full_size


def calc_acc(completion: str, target_ids: torch.Tensor, encoding: tiktoken.Encoding) -> float:
    completion_ids = encoding.encode_ordinary(completion)
    if len(completion_ids) > len(target_ids):
        completion_ids = completion_ids[:len(target_ids)]
    if len(target_ids) > len(completion_ids):
        target_ids = target_ids[:len(completion_ids)]
    return (torch.tensor(completion_ids, device="cuda") == target_ids).float().mean().item()


def calc_ce_loss(logprobs: torch.Tensor, target_ids: torch.Tensor) -> float:
    if len(logprobs) > len(target_ids):
        logprobs = logprobs[:len(target_ids)]
    if len(target_ids) > len(logprobs):
        target_ids = target_ids[:len(logprobs)]
    return F.cross_entropy(logprobs, target_ids).item()


def calc_l2_loss(logprobs_c1: torch.Tensor, logprobs_c2: torch.Tensor) -> float:
    if len(logprobs_c1) > len(logprobs_c2):
        logprobs_c1 = logprobs_c1[:len(logprobs_c2)]
    if len(logprobs_c2) > len(logprobs_c1):
        logprobs_c2 = logprobs_c2[:len(logprobs_c1)]
    return F.mse_loss(logprobs_c1, logprobs_c2).item()


def calc_preference_stats(
        sentence: str,
        completion_c: str, 
        completion_r: str, 
        openai_model: str = "gpt-4o-mini",
        num_samples: int = 10,
) -> dict[Literal["c", "r"], int]:
    if num_samples % 2:
        raise ValueError("num_samples must be even")
    if not num_samples:
        raise ValueError("num_samples must be positive")
    
    preferences = dict(c=0, r=0)
    for i in range(num_samples // 2):
        preference1 = choose_preference(
            base_text=sentence,
            completion1=completion_c,
            completion2=completion_r,
            openai_model=openai_model,
        )
        preference2 = choose_preference(
            base_text=sentence,
            completion1=completion_r,
            completion2=completion_c,
            openai_model=openai_model,
        )
        preferences["c"] += (1 - preference1) + (1 - preference2)
        preferences["r"] += preference1 + preference2
        
    return preferences


def test_free_completion(
        net_c: SpeedyLangNet, 
        net_r: SpeedyLangNet, 
        encoder: tiktoken.Encoding, 
        sentences: list[str], 
        verbosity: int = 1, 
        max_choose_nth_best: int = 3,
        masking_rate: float = 0.0,
        num_preference_samples: int = 0,
) -> dict[str, Any]:
    results = dict()
    loop = tqdm(sentences, disable=not verbosity)
    for sentence in loop:
        results[sentence] = dict()
        for choose_nth_best in range(1, max_choose_nth_best+1):
            loop.set_description(f"{choose_nth_best=}/{max_choose_nth_best}")
            completion_c, _, _ = generate(
                net=net_c,
                encoder=encoder,
                query=sentence,
                max_gen_tokens=50,
                choose_nth_best=choose_nth_best,
                masking_rate=masking_rate,
            )
            size_ratio_completion_c, size_ratio_full_c = calc_ratio_compression(completion_c, sentence+completion_c)
            num_unique_words_c = len(set(completion_c.split()))
            num_unique_tokens_c = len(set(encoder.encode_ordinary(completion_c)))

            completion_r, _, _ = generate(
                net=net_r,
                encoder=encoder,
                query=sentence,
                max_gen_tokens=50,
                choose_nth_best=choose_nth_best,
                masking_rate=masking_rate,
            )
            size_ratio_completion_r, size_ratio_full_r = calc_ratio_compression(completion_r, sentence+completion_r)
            num_unique_words_r = len(set(completion_r.split()))
            num_unique_tokens_r = len(set(encoder.encode_ordinary(completion_r)))

            if num_preference_samples:
                preferences = calc_preference_stats(
                    sentence=sentence,
                    completion_c=completion_c,
                    completion_r=completion_r,
                    num_samples=num_preference_samples,
                )
                results[sentence][f"preference_c{choose_nth_best}"] = preferences["c"] / num_preference_samples
                results[sentence][f"preference_r{choose_nth_best}"] = preferences["r"] / num_preference_samples

            results[sentence][f"completion_c{choose_nth_best}"] = completion_c
            results[sentence][f"size_ratio_completion_c{choose_nth_best}"] = size_ratio_completion_c
            results[sentence][f"size_ratio_full_c{choose_nth_best}"] = size_ratio_full_c
            results[sentence][f"num_unique_words_c{choose_nth_best}"] = num_unique_words_c
            results[sentence][f"num_unique_tokens_c{choose_nth_best}"] = num_unique_tokens_c
            results[sentence][f"completion_r{choose_nth_best}"] = completion_r
            results[sentence][f"size_ratio_completion_r{choose_nth_best}"] = size_ratio_completion_r
            results[sentence][f"size_ratio_full_r{choose_nth_best}"] = size_ratio_full_r
            results[sentence][f"num_unique_words_r{choose_nth_best}"] = num_unique_words_r
            results[sentence][f"num_unique_tokens_r{choose_nth_best}"] = num_unique_tokens_r

        if verbosity > 1:
            print(
                f"\n\n{sentence=}\n\n"
                f"{results[sentence]=}\n\n"
            )
    
    summary = dict()
    for choose_nth_best in range(1, max_choose_nth_best+1):
        nth_summary = {
            f"mean_size_ratio_completion_c{choose_nth_best}": torch.tensor([results[sentence][f"size_ratio_completion_c{choose_nth_best}"] for sentence in results]).float().mean().item(),
            f"mean_size_ratio_full_c{choose_nth_best}": torch.tensor([results[sentence][f"size_ratio_full_c{choose_nth_best}"] for sentence in results]).float().mean().item(),
            f"mean_num_unique_words_c{choose_nth_best}": torch.tensor([results[sentence][f"num_unique_words_c{choose_nth_best}"] for sentence in results]).float().mean().item(),
            f"mean_num_unique_tokens_c{choose_nth_best}": torch.tensor([results[sentence][f"num_unique_tokens_c{choose_nth_best}"] for sentence in results]).float().mean().item(),
            f"mean_size_ratio_completion_r{choose_nth_best}": torch.tensor([results[sentence][f"size_ratio_completion_r{choose_nth_best}"] for sentence in results]).float().mean().item(),
            f"mean_size_ratio_full_r{choose_nth_best}": torch.tensor([results[sentence][f"size_ratio_full_r{choose_nth_best}"] for sentence in results]).float().mean().item(),
            f"mean_num_unique_words_r{choose_nth_best}": torch.tensor([results[sentence][f"num_unique_words_r{choose_nth_best}"] for sentence in results]).float().mean().item(),
            f"mean_num_unique_tokens_r{choose_nth_best}": torch.tensor([results[sentence][f"num_unique_tokens_r{choose_nth_best}"] for sentence in results]).float().mean().item(),
        }
        if num_preference_samples:
            nth_summary.update(
                {
                    f"mean_preference_c{choose_nth_best}": torch.tensor([results[sentence][f"preference_c{choose_nth_best}"] for sentence in results]).float().mean().item(),
                    f"mean_preference_r{choose_nth_best}": torch.tensor([results[sentence][f"preference_r{choose_nth_best}"] for sentence in results]).float().mean().item(),
                }
            )
        summary.update(nth_summary)
    results["summary"] = summary
    return results


def test_split_sentences(
        net_c: SpeedyLangNet, 
        net_r: SpeedyLangNet, 
        encoder: tiktoken.Encoding, 
        sentences: list[str], 
        verbosity: int = 1, 
        min_completion_len: int = 10,
        step_between_completion_lengths: int = 10,
        max_choose_nth_best: int = 3,
        masking_rate: float = 0.0,
) -> dict[str, dict[str, Any | list[dict[str, Any]]]]: 
    results = dict()
    loop = tqdm(sentences, disable=not verbosity)
    for sentence in loop:
        results[sentence] = dict()
        input_ids = encoder.encode_ordinary(sentence)
        max_completion_len = len(input_ids) // 2
        max_completion_len = max(
            min_completion_len, 
            max_completion_len - (max_completion_len % step_between_completion_lengths)
        )
        completion_lengths = list(range(min_completion_len, max_completion_len+1, step_between_completion_lengths))
        for i, completion_length in enumerate(completion_lengths):
            partial_sentence = encoder.decode(input_ids[:-completion_length])
            target_ids = torch.tensor(input_ids[-completion_length:]).to("cuda")
            details = dict(partial_sentence=partial_sentence, target_ids=target_ids.tolist())
            for choose_nth_best in range(1, max_choose_nth_best+1):
                loop.set_description(
                    f"{completion_length=}/{max_completion_len}, "
                    f"{choose_nth_best=}/{max_choose_nth_best}"
                ) 
                completion_c, _, logprobs_c = generate(
                    net=net_c,
                    encoder=encoder,
                    query=partial_sentence,
                    max_gen_tokens=completion_length,
                    choose_nth_best=choose_nth_best,
                    masking_rate=masking_rate,
                )
                completion_r, _, logprobs_r = generate(
                    net=net_r,
                    encoder=encoder,
                    query=partial_sentence,
                    max_gen_tokens=completion_length,
                    choose_nth_best=choose_nth_best,
                    masking_rate=masking_rate,
                )

                details[f"completion_c{choose_nth_best}"] = completion_c
                details[f"edit_distance_c{choose_nth_best}"] = Levenshtein.distance(target_ids.tolist(), encoder.encode_ordinary(completion_c))
                details[f"acc_c{choose_nth_best}"] = calc_acc(completion_c, target_ids, encoder)
                details[f"ce_loss_c{choose_nth_best}"] = calc_ce_loss(logprobs_c, target_ids)
                details[f"completion_r{choose_nth_best}"] = completion_r
                details[f"edit_distance_r{choose_nth_best}"] = Levenshtein.distance(target_ids.tolist(), encoder.encode_ordinary(completion_r))
                details[f"acc_r{choose_nth_best}"] = calc_acc(completion_r, target_ids, encoder)
                details[f"ce_loss_r{choose_nth_best}"] = calc_ce_loss(logprobs_r, target_ids)
                details[f"l2_loss_{choose_nth_best}"] = calc_l2_loss(logprobs_c, logprobs_r)

            results[sentence][f"completion_length_{completion_length}"] = details
            if verbosity > 1:
                print(results[sentence]["details"][-1])

    summary = dict()
    for choose_nth_best in range(1, max_choose_nth_best+1):
        summary.update(
            {
                f"mean_acc_c{choose_nth_best}": sum([results[sentence][completion_len][f"acc_c{choose_nth_best}"] for completion_len in results[sentence]]) / len(results[sentence]),
                f"mean_ce_loss_c{choose_nth_best}": sum([results[sentence][completion_len][f"ce_loss_c{choose_nth_best}"] for completion_len in results[sentence]]) / len(results[sentence]),
                f"mean_edit_distance_c{choose_nth_best}": sum([results[sentence][completion_len][f"edit_distance_c{choose_nth_best}"] for completion_len in results[sentence]]) / len(results[sentence]),
                f"mean_acc_r{choose_nth_best}": sum([results[sentence][completion_len][f"acc_r{choose_nth_best}"] for completion_len in results[sentence]]) / len(results[sentence]),
                f"mean_ce_loss_r{choose_nth_best}": sum([results[sentence][completion_len][f"ce_loss_r{choose_nth_best}"] for completion_len in results[sentence]]) / len(results[sentence]),
                f"mean_edit_distance_r{choose_nth_best}": sum([results[sentence][completion_len][f"edit_distance_r{choose_nth_best}"] for completion_len in results[sentence]]) / len(results[sentence]),
                f"mean_l2_loss_{choose_nth_best}": sum([results[sentence][completion_len][f"l2_loss_{choose_nth_best}"] for completion_len in results[sentence]]) / len(results[sentence]),
            }
        )
    results["summary"] = summary
    return results


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_size",
        type=int, default=773, choices=[1300, 773, 240],
        help="The model size to use. TYPE: int; DEFAULT: 773"
    )
    parser.add_argument(
        "--verbosity",
        type=int, default=1, choices=[0, 1, 2],
        help="The verbosity level. TYPE: int; DEFAULT: 1"
    )
    parser.add_argument(
        "--savefile",
        type=str, default=None,
        help="The file to save the results to. TYPE: str; DEFAULT: None"
    )
    parser.add_argument(
        "--no_test_split_sentences",
        action="store_false",
        help="Do not test the split_sentences function. TYPE: bool; DEFAULT: True"
    )
    parser.add_argument(
        "--no_test_free_completion",
        action="store_false",
        help="Do not test the free_completion function. TYPE: bool; DEFAULT: True"
    )
    parser.add_argument(
        "--max_choose_nth_best",
        type=int, default=3,
        help="The maximum number of nth best completions to test. TYPE: int; DEFAULT: 3"
    )
    parser.add_argument(
        "--masking_rate",
        type=float, default=0.0,
        help="The masking rate to use. TYPE: float; DEFAULT: 0.0"
    )
    parser.add_argument(
        "--num_preference_samples",
        type=int, default=0,
        help="The number of samples to use for preference calculation. TYPE: int; DEFAULT: 0"
    )

    return parser.parse_args()


def save_json(data: dict, path: str, postfix: str = ""):
    name = path.split(".")[0]
    ending = path.split(".")[1] if len(path.split(".")) > 1 else "json"
    with open(f"{name}_{postfix}.{ending}", "w") as f:
        json.dump(data, f, indent=2)


def main():
    """Test if the model weights are correctly loaded"""
    args = get_args()
    if args.model_size == 773:
        model_name_c = "snimu/causal-ul2-C-fineweb10BT-773M-26heads-lr090"
        model_name_r = "snimu/causal-ul2-R-fineweb10BT-773M-26heads-lr090"
    elif args.model_size == 240:
        model_name_c = "snimu/causal-ul2-C-fineweb10BT-240M-16heads-lr090"
        model_name_r = "snimu/causal-ul2-R-fineweb10BT-240M-16heads-lr090"
    elif args.model_size == 1300:
        model_name_c = "snimu/causal-ul2-C-wikitext-1300M-1head-lr100"
        model_name_r = "snimu/causal-ul2-RX-wikitext-1300M-1head-lr100"

    net_rand = make_net_from_name(model_name_c).to("cpu")
    net_c = make_net_from_name(model_name_c).to("cpu")
    for p1, p2 in zip(net_rand.parameters(), net_c.parameters()):
        p2.data.copy_(p1.data)
    assert all([torch.all(p1 == p2) for p1, p2 in zip(net_rand.parameters(), net_c.parameters())])

    model_path = download_model(model_name_c)
    safetensors.torch.load_model(net_c, model_path, device="cpu")
    assert not all([torch.all(p1 == p2) for p1, p2 in zip(net_rand.parameters(), net_c.parameters())])
    del net_rand

    net_r = make_net_from_name(model_name_c).to("cpu")
    model_path = download_model(model_name_r)
    safetensors.torch.load_model(net_r, model_path, device="cpu")

    sentences = [
        "The cinnamon quail-thrush (Cinclosoma cinnamomeum) is a species of bird in the family Cinclosomatidae. Endemic to Australia, it is typically found in arid and semi-arid regions of the central part of the continent, spanning southwest Queensland, northwest New South Wales, northeastern South Australia, and the southeast of the Northern Territory. It is most commonly found among dry stony areas, especially",
        'Carcharodontosauridae (carcharodontosaurids; from the Greek carcharodontosauros: "shark-toothed lizards") is a group of carnivorous theropod dinosaurs. In 1931, Ernst Stromer named Carcharodontosauridae as a family, which, in modern paleontology, indicates a clade within Carnosauria. Carcharodontosaurids include some of the largest land predators ever known: Giganotosaurus, Mapusaurus, Carcharodontosaurus, and Tyrannotitan all rivaled Tyrannosaurus in size. Estimates give a maximum weight of',
        "The 2000–01 World Sevens Series was the second edition of the global circuit for men's national rugby sevens teams, organised by the International Rugby Board. The season ran from November 2000 to June 2001 and consisted of nine tournaments (originally 10 were scheduled, but one was cancelled).\n\nThe series was won by New Zealand, who won six of the nine tournaments. Australia won the other three tournaments, and",
        'Peregian Beach within the Sunshine Coast Region comprises continual residential development along the eastern coastal strip of sandy beaches. The David Low Way passes north to south through this area. Development to the west is constrained by',
        "Linton was the eldest son of Jabez Linton of Hardrigg Lodge, Dumfriesshire, by Jane, daughter of William Crocket of Grahamshill in the same county. He was born in 1801 at Kirkpatrick Fleming. He was educated at Edinburgh University, and graduated L.R.C.S. in 1826. But he had already utilised four summer vacations as surgeon on a whaler in the arctic regions. He entered the army medical department in 1826, graduated M.D. at Glasgow in 1834, and became staff surgeon of the first class in 1848. After serving in Canada, the Mediterranean, and the West Indies, he was appointed deputy inspector-general of hospitals of the first division of the army in the Crimea, was present in every action up to the battle of Balaclava, and had care of the barrack hospital in Scutari shortly after its establishment in 1854 until the British forces",
        "Two years later, Mozambique qualified for their third Africa Cup of Nations held in Burkina Faso. They were again placed in group D along with Morocco, Egypt and Zambia. Mozambique lost their first game against eventual tournament winners Egypt 2–0, both goals coming from Hossam Hassan. In their second game they again lost to Morocco 3–0, therefore eliminating them from",
        "Kirkman intended Freedom Ring to be an example of a superhero who demonstrated inexperience with his superpowers, as he felt that most superheroes quickly adjusting to their powers and having a successful superhero career did not reflect reality. When asked by a fan about the number of visibly gay comic book superheroes, Editor-in-Chief of Marvel Comics, Joe Quesada, also touted"
        'Portage-du-Fort is named after the portage trail which started here and would lead upstream around a set of falls on the Ottawa River.\n\nHowever, there are several hypotheses to explain the "Fort" portion. Among the most popular is the assumption that a fort was present here on the shore of the Ottawa River to keep provisions at the portage. It has been claimed that a fort called Dufort was flooded in the rapids at this location. However, some researchers argue that the fort in question has never existed and may be a reference to another fort at the mouth of the Coulonge River (after which modern Fort-Coulonge is named). Moreover, the word formerly did not always convey a military connotation and could be more or less synonymous with a village or hamlet, or even a post or warehouse which was fortified.[1]\n\nOne theory suggests that the name goes back to a custom of the Algonquins who would paint their bodies here and it was originally named Portage du Fard (French for "make-up"), which changed into "Fort".[1]\n\nAnother possibility is that Fort (French also for "strong") makes reference to the strength needed to haul the heavy canoes and supplies"',
        "The Country Club of Birmingham, previously known as Birmingham Country Club, located in Birmingham, Alabama, United States, was founded in 1898. It moved in 1900 from North Birmingham to Lakeview, then again in 1926 to a site in Shades Valley, now within the city of Mountain Brook. The Lakeview club hosted former president Theodore Roosevelt and several Women's Southern Golf Association tournaments.",
        "Saint Barthélemy was for many years a French commune forming part of Guadeloupe, which is an overseas region and department of France. In 2003 the island voted in favour of secession from Guadeloupe to form a separate overseas collectivity (collectivité d'outre-mer, abbreviated to COM) of France. The collectivity is one of four territories among the Leeward Islands in the northeastern Caribbean that make up the French West Indies, along with Saint Martin, Guadeloupe (200 kilometres (120 mi) southeast) and",
        "Teuku Umar University (Indonesian: Universitas Teuku Umar, abbreviated UTU) is an Indonesian public university in Meulaboh, West Aceh Regency. Starting from a foundation in 1984, it became a university in 2006 and was nationalized to form the current public university (Perguruan Tinggi Negeri) in 2014.",
        "Mpanga Central Forest Reserve is in the Central Region of Uganda, Mpigi District, Mpabire, to be exact. By road, it is 37 km Southwest of Kampala City. It can easily be accessed through a one hour drive along the main Kampala – Masaka Highway. This makes Mpanga Central Forest Reserve the closest natural equatorial rainforest to the Capital City.",
        "Two units of the Late Permian Gerringong volcanic facies are exposed on Bombo Headland. The Kiama Sandstone member forms a narrow wave-cut platform and adjacent vertical cliff face around the south-eastern extremity of the quarry. To the north, the sandstone dips below sea level and is overlain by about 20m of porphyritic basalt, termed Bombo Latite member. The contact between the two units is well-exposed in the cliff section at the eastern end of the two points comprising the headland section. The red-brown colour (due to oxidization of haematite) of the sandstone contrasts markedly with the grey-black latite, which displays spectacular columnar jointing elsewhere in the quarry. Isolated columns 5–5 meters in height stand adjacent to the coast between the north and south parts of the quarry; just to the north the sea wall exposes cross-sections",
        "The modern pentathlon at the 2024 Summer Olympics in Paris took place during 8 to 11 August 2024 at the Palace of Versailles and the Arena Paris Nord. The Palace of Versailles hosted all the modern pentathlon events, with the only exclusion being the fencing ranking rounds, which occurred at the North Paris Arena. Two events were contested, one for men and another for women.",
        "At night, Marge is annoyed by Homer shaving batteries in bed. The next day, Principal Skinner, who is engaged to Edna Krabappel,[a] announces that they are to be married that weekend. Edna has her bachelorette party at the Simpson house with Duffman as a stripper. When Chief Wiggum tries to stop the party after a complaint from Ned, they coerce him into stripping as well. Meanwhile, Principal Skinner has his party at Moe's Tavern with Homer. After getting drunk, Skinner admits to Homer that he has doubts about marrying Edna. Homer tells Marge, and they agree to make sure Skinner and Edna get married. On the day of the wedding, Edna overhears Skinner and Homer discussing his doubts. After",
        "The Cathedral of the Blessed Virgin Mary is the cathedral of the Roman Catholic Diocese of Hamilton, New Zealand. It was opened in 1975, replacing an earlier Neo-Classical building known as St Mary's Church which was built in 1911–1912.[1] The Cathedral of the Blessed Virgin Mary was dedicated and renamed on 27 April 1980 and rededicated, following refurbishment, on 7 November",
        "Prior to 1991, the Singaporean government did not have an official classification system. Instead, works were either unconditionally allowed, partially censored, or completely banned. Starting from June 1991, the Media Development Authority (MDA) instituted a ratings system with 3 ratings – General Audience (G), Parental Guidance (PG), and Restricted 18 (R18).",
        "Dwight Marlon Washington (born 5 March 1983) is a West Indian international cricketer.\n\nWashington made his first-class debut as a fast bowler for West Indies B in the Carib Beer Cup in 2003–04, taking 20 wickets at 22.00 and earning a place in a strong Carib Beer XI against the England XI at the end of the season. Against Guyana, batting in his usual position of number 11, he scored 58 off 58 balls, including six sixes.",
        "Marshall Junction is an unincorporated community in Saline County, Missouri, United States. Marshall Junction is located at the junction of Interstate 70 and U.S. Route 65, 12 miles (19 km) south of Marshall.",
        "Hellgate Roller Derby (HRD) formerly known as the Hellgate Rollergirls (HGRG) is a skater-run, non-profit organization designed exclusively for the purpose of promoting the sport of flat track roller derby in Missoula, Montana.[1] HRD always welcomes new skaters of all genders, body types and athletic abilities. The board of directors (BOD), all committee heads, members, and all skaters are volunteers.",
        "Natasha Bharadwaj is an Indian actress who primarily works in Hindi web shows. Bharadwaj started her career as a contestant on the reality show India's Next Superstars (2018), where she emerged as the winner. She then made her acting debut with the web series Pawan & Pooja (2020). Bharadwaj is best known for her role in the series Mumbai Diaries 26/11 (2021).[1]",
        "The book begins by introducing a New York City policeman named Bill Nichols. He is investigating a mysterious house fire from which a little girl by the name of Kali survived. Out of the havoc that was reaped, it seems highly suspicious that a little girl survived. He believes it is luck and dismisses any suspicion, leaving Kali's fate unknown. The setting shifts to Chile, where a boy by the birth name of Catequil and the nickname of Tigre sits in class. The clouds roll in as he sits there, bored. The clouds take control and the boy jumps from class to run outside and enjoy the storm. He loses consciousness and faints. Meanwhile, an Egyptian tomb",
        "The rock firefinch (Lagonosticta sanguinodorsalis) is a species of estrildid finch found in the Jos Plateau of central Nigeria and in Cameroon. It has an estimated global extent of occurrence of 29,000 km2. The rock firefinch was discovered recently, in 1998. Rock firefinches fall in the family Estrildidae, which contains small passerine birds of the Old World and Australasia. Rock firefinches seem to be most closely related to Mali firefinches and Chad firefinches.[2] The species name sanguinodorsalis means blood-red back, which was chosen because it describes the vibrant red back color of the male plumage.[2] The status of the species is evaluated as Least Concern.\n\nDescription\nRock firefinches are sexually dimorphic, where adult males have more brightly colored plumage than adult females.[3] Males are characterized",
        "Archduchess Magdalena and her younger sister Margaret had long expressed a desire to remain unmarried and create a community of pious women, which their father had a difficult time accepting. After his death in 1564, Magdalena took a vow of celibacy and founded the Ladies' Convent of Hall (Haller Damenstift) in Hall in Tirol, County of Tyrol, a place for like-minded women to lead a reclusive, pious and God-fearing lives under the supervision of the Society of Jesus.",
        "Reason, also known as ReasonML, is a general-purpose, high-level, multi-paradigm, functional and object-oriented programming language and syntax extension and toolchain for OCaml created by Jordan Walke, who also created the React framework, at Facebook.[3][4] Reason uses many syntax elements from JavaScript, compiles to native code using OCaml's compiler toolchain, and can compile to JavaScript using the ReScript compiler",
        "The Liberty Bell Ruby is a sculpture crafted from the world's largest mined ruby,[1] discovered in East Africa in the 1950s.[2] It weighs four pounds, is eight and a half thousand carats (8,500), and is sculpted into a miniature form of the Liberty Bell. It has 50 diamonds set in it and is valued at $2 million.\n\nThe ruby was created in 1976 for Beverly Hills-based Kazanjian Brothers jewelry company by sculptor Alfonso de Vivanco for the United States Bicentennial.[3] It was made in the same spirit as sapphire busts of presidents that the jeweler's charitable",
        "Houssam El Kord (also spelled El-Kord, born 24 February 1993)[1] is a Moroccan fencer.[2][3] His sister Camélia El Kord is also a fencer who has represented Morocco at international level. He currently lives in France and also serves as a freelance chiropodist in Paris.[2]",
        "Given a set of points in an n-dimensional data space, QC represents each point with a multidimensional Gaussian distribution, with width (standard deviation) sigma, centered at each point’s location in the space. These Gaussians are then added together to create a single distribution for the entire data set. (This step is a particular example of kernel density estimation, often referred to as a Parzen-Rosenblatt window estimator.) This distribution is considered to be the quantum-mechanical",
        "The State Cal-Fire Authority officially designated Brentwood, from Mulholland down to Sunset Boulevard, a Very High Fire Hazard Severity Zone, due to the long, uninterrupted border of urban-wildlife interface in the hillsides that has resulted in multiple fires over many years, destroying entire neighborhoods and requiring numerous evacuations. For this reason, the Brentwood community has been strongly in favor of halting all further development in the hillside and canyon areas.",
        'Infinite Spring is the debut studio album by American indie folk band Superviolet, released by Lame-O Records on April 21, 2023.\n\nReception\nOn June 19, Paste reviewed the best albums of the year so far, ranking this release fifteenth for "a relentless curiosity [that is] so refreshingly brilliant and poetic".[4]',
        "On 24 October 1941, after a four-day battle, Kharkiv and the district was occupied by German forces. In advance of the Germans, most of the industrial plant, including the KhTZ, had been dismantled and moved east or rendered inoperative. On 14 December, the German Stadtkommandant ordered the Jewish population to be concentrated in a hut settlement near the KhTZ. In two days, 20,000 Jews were gathered there. Those an SS Sonderkommando did not shoot were killed throughout January in a gas van.[3][4] The district and the city were liberated by Soviet forces in February 1943. The district was liberated again, following a German counteroffensive in March,[5] in August 1943.",
        "In the 13th century, the territory on which Kirillov now stands was a part of the Principality of Beloozero, which was taken over by the Grand Duchy of Moscow in the 14th century.[citation needed] In 1397, St. Cyril of White Lake, a monk and a disciple of St. Sergius of Radonezh, founded the Kirillo-Belozersky Monastery on the shore of Lake Siverskoye.[3] A monastic sloboda, from which the town later grew, developed around the monastery.[3] The monastery was subordinate to Archbishops of Rostov.[citation needed] In the 15th–17th centuries, the monastery developed into one of the most influential monasteries in Russia.[3] It also helped that the Sheksna River was one of the most heavily used waterways connecting central and northern Russia.[citation needed] At some point, the monastery was the second biggest landowner after the Trinity Lavra of St. Sergius.[citation needed] Vasili III of Russia, the Grand Prince of Moscow, and Ivan the Terrible, the Tsar, visited the monastery on several occasions.",
        "There were at least six monasteries and more than 40 churches within the Nochiya Region. The Nochiyaye were best known for their adherence to the Assyrian Church of the East faith; because of this, religious customs such as Lent and prayer were strictly observed. The Mar Ishu Monastery in the village of Mar Ishu was a theological school for priests and was run by the Metropolitans of Shamizdin, who would not tolerate any changes to the church's canon laws.[2]",
        "Marchitecture (or marketecture) is a portmanteau of the words marketing and architecture. The term is applied to any form of electronic architecture, especially software, perceived to have been produced purely for marketing reasons.[1][2] It may be used by a vendor to place itself in such a way as to promote all their strongest abilities whilst simultaneously masking their weaknesses.\n\nThe term marketecture is also used in the context of an abstract description of a complex system, such as a distributed software system, for the purpose of discussion and analysis. In his book Essential Software Architecture, Ian Gorton describes it as",
        "The Cambridge Songs (Carmina Cantabrigiensia) are a collection of Goliardic medieval Latin poems found on ten leaves (ff. 432–41) of the Codex Cantabrigiensis (C, MS Gg. 5.35), now in Cambridge University Library.\n\nHistory and content\nThe songs as they survive are copies made shortly before or after the Norman Conquest (1066). They may have been collected by an English scholar while travelling on the continent sometime after the last datable song (1039), and brought back with him to the church of Saint Augustine at Canterbury, where they were copied and where the Codex was long kept. The original manuscript was possibly lost in a fire that struck Saint Augustine's in 1168. The dialect of the few vernacular portions found in some of the songs is in the North Rheno-Franconian dialect of Old High German, suggesting that the Goliard or Goliards who composed them came from the north or middle Rhineland, probably the area between Trier, Cologne, and Xanten. It has been suggested that some of the songs originated in France or Italy. While most of the Cambridge Songs survive only in the Cambridge manuscript, a few are duplicated in a manuscript, W, from Wolfenbüttel.",
        'The Plot Against Common Sense is the third studio album by Future of the Left.\n\nReception\nCritical response to the album was positive, with a Metacritic score of 81/100 or "universal acclaim".\n\nTrack listing\n1. "Sheena Is A T-Shirt Salesman" - 2:08\n2. "Failed Olympic Bid" - 3:14\n3. "Beneath The Waves An Ocean" - 3:47\n4. "Cosmo\'s Ladder" - 2:34\n5. "City Of Exploded Children" - 4:10\n6. "Goals In Slow Motion" - 3:11\n7. "Camp Cappuccino" - 2:48\n8. "Polymers Are Forever" - 4:07\n9. "Robocop 4 - Fuck Off Robocop" - 2:53\n10. "Sorry Dad, I Was Late For The Riots" - 3:08\n11. "I Am The Least Of Your Problems" - 2:33\n12. "A Guide To Men" - 3:54\n13. "Anchor" - 3:12\n14. "Rubber Animals" - 1:54\n15. "Notes On Achieving Orbit" - 6:22 (including hidden track)\nRunning time: 49:55',
        "Corwin Carl Guell (December 22, 1909 – December 1976) was a member of the Wisconsin State Assembly.\n\nBiography\nGuell was born Corwin Carl Guell on December 22, 1909, in Fond du Lac, Wisconsin.[1] He was later a resident of Thorp, Wisconsin,[2][3] where he worked as an attorney.[4] In 1932, he married Anna L. Zimmerman. They had three children. He attended North Central College, Northwestern University and the University of Wisconsin Law School. During World War II, he served as an officer in the United States Navy. He was also active in his local Methodist church, serving as a lay speaker.\n\nPolitical career\nGuell was a member of the Assembly from 1957 to 1958. He also made an unsuccessful run for the Assembly in 1960.[2] Guell was a Republican.[2]",
        "Mavis Anne Freeman (November 7, 1918 – October 1988) was an American competition swimmer who represented the United States in the 1936 Summer Olympics in Berlin, Germany. Freeman received a bronze medal as a member of the third-place U.S. team in the women's 4×100-meter freestyle relay, together with her teammates Katherine Rawls, Bernice Lapp and Olive McKean. The Americans finished in a time of 4:40.2, behind the women's teams from the Netherlands and Germany.[1]",
        "In biology, a tropism is a phenomenon indicating the growth or turning movement of an organism, usually a plant, in response to an environmental stimulus.[1] In tropisms, this response is dependent on the direction of the stimulus (as opposed to nastic movements, which are non-directional responses). Tropisms are usually named for the stimulus involved; for example, a phototropism is a movement to the light source, and an anemotropism is",
        "Lorita Sanderson was a rising actress with little fear of interrup-tion to her brilliant career. Then she went to see a fortune-teller, Mrs. Bates... When Mrs. Bates is asked to say what Lorita will be at 4 p.m. on March 15, her birthday, Mrs. Bates foretells a future that Lorita feels she must at all costs avoid. So worried is she by the possibility that Mrs. Bates’ prediction may come true that she refuses to take the lucrative and enticing star-part which a producer offers her, and hides herself away in a country cottage...Lorita drives herself almost into a frenzy—and her friends’ stories of other successful predictions by Mrs. Bates do not help her sanity.",
        "Gumercinda Páez (1904-1991) was a teacher, women's rights activist and suffragette, and Constituent Assemblywoman of Panama. She was the first woman deputy to serve the National Assembly for the Panamá Province and was a vice president of the Constituent Assembly of Panama in 1946, being also the first woman to serve in that position. As a woman of mixed heritage, she was acutely aware of bias and strove for policies of inclusion.",
        "Locally there is much evidence of Saxon iron works and a stretch of Roman Road still exists today known locally as the \"Quarter Mile\".[3] St Margaret's Church was built in the 13th century. It contains both the grave of Mark Lemon (the first editor of Punch), and the Holles family vault. Adjacent to St Margaret's Church is the Ifield Barn Theatre. The old parish of Ifield contained most of the western part of modern-day Crawley, and the old village is on the very western edge of the new town. As well as containing two modern churches, St.Leonards in Langley Green and St.Albans in Gossops Green, Ifield Parish also contains a Friends' Meeting House. Founded in 1676, it was the first purpose-built meeting place for the Quakers anywhere in the world.[3]",
        "This is a list of digraphs used in various Latin alphabets. In the list, letters with diacritics are arranged in alphabetical order according to their base, e.g. ⟨å⟩ is alphabetised with ⟨a⟩, not at the end of the alphabet, as it would be in Danish, Norwegian and Swedish. Substantially-modified letters, such as ⟨ſ⟩ (a variant of ⟨s⟩) and ⟨ɔ⟩ (based on ⟨o⟩), are placed at the end.\n\nCapitalisation only involves the first letter (⟨ch⟩ becomes ⟨Ch⟩) unless otherwise stated (⟨ij⟩ becomes ⟨IJ⟩ in Dutch, and digraphs marking eclipsis in Irish, are capitalised on the second letter, i.e. ⟨mb⟩ becomes ⟨mB⟩).",
        "UDP-glucuronic acid dehydrogenase (UDP-4-keto-hexauronic acid decarboxylating) (EC 1.1.1.305, UDP-GlcUA decarboxylase, ArnADH) is an enzyme with systematic name UDP-glucuronate:NAD+ oxidoreductase (decarboxylating).[1][2][3][4][5] This enzyme catalyses the following chemical reaction",
        'Maxim (more accurately spelled Maksim assuming that "X" is not a consonant, but the conjunction of "K" and "S" sounds; “Maksym”, or "Maxym") is an epicene (or gender-neutral) first name of Roman origin mainly given to males. It is adopted in Slavic-speaking countries such as Russia, Belarus, Bulgaria, Ukraine, Moldova, Kazakhstan, Serbia, Macedonia and Montenegro, as well as in countries which have maintained ties to the Soviet era. The spelling variant Maxime is also common in the French-speaking world. The name is derived from the Latin family name Maximus, meaning "the greatest".[1] Maxim is also a less well-known surname.',
        "Beauty and the Barge is a 1937 British comedy film directed by Henry Edwards and starring Gordon Harker, Judy Gunn and Jack Hawkins.[1] It was produced by Julius Hagen's production company Twickenham Film Studios, but made at the Riverside Studios in Hammersmith rather than at Twickenham.[2] It was based on the 1905 play Beauty and the Barge by W. W. Jacobs.",
        "Joseph Cornell began to collect films on 16mm in the 1920s, mainly to entertain his mother and disabled brother at the family home in Queens, where Cornell lived for two thirds of his life. To vary the program and to surprise his family, Cornell began to alter his films slightly by adding shots, or changing the endings to films with which they were familiar. This led, in time, to his first and most elaborate film collage, Rose Hobart (1936), coincidentally concurrent with his first box, Soap Bubble Set (1936), later sold to the Museum of Modern Art in New York. Recently discovered correspondence between Cornell and Iris Barry of MoMA's film library reveals that he was already conversant with language relevant to the then emerging field of film preservation and was also occasionally collecting films in 35mm nitrate.[5]",
        "The East Coast Conference Men's Basketball Player of the Year was an award given to the East Coast Conference's most outstanding player. The award was first given following the 1974–75 season and was discontinued after the league folded following the 1993–94 season. In 1994 the East Coast Conference was absorbed into the Mid-Continent Conference, now known as the Summit League.",
        "Pietro Paolo Agabito or Agabiti (c1470-c1540) was an Italian Renaissance painter, sculptor, and architect from the Marche region. His style is rather provincial, and most surviving works are in the churches and museums of the region. He may have trained with Carlo Crivelli, and among the artists generally credited with having influenced his style are the Venetians Cima da Conegliano and Alvise Vivarini, the Bolognese artist Francesco Francia and Marco Palmezzano of Forlì. However, Agabiti did not keep up with the changes of style occurring in the early sixteenth century, remaining attached to the more formal style of the fifteenth century.",
        "Utopia Planitia (Greek and Latin: \"Utopia Land Plain\") is a large plain[2] within Utopia, the largest recognized impact basin on Mars[a] and in the Solar System with an estimated diameter of 3,300 km (2,100 mi).[1] It is the Martian region where the Viking 2 lander touched down and began exploring on September 3, 1976, and the Zhurong rover touched down on May 14, 2021, as a part of the Tianwen-1 mission"
        'Lake Shawnee Amusement Park, abandoned in 1966, occupies a desecrated native burial ground which was the site of the 1783 Mitchell Clay settler farm. Three of the Clay children (Bartley, Tabitha, Ezekial) were killed by a band of natives; Mitchell Clay led a group of settlers in bloody retaliation, killing several natives. In the 1920s, businessman Conley T. Snidow purchased the site of the Clay farm for development as an amusement park. At least six amusement patrons were killed while the park was in operation; a little girl on the circling swing set was hit after a truck backed into the path of the swing, and a boy drowned in the amusement park\'s swimming pond. The park\'s structures and rides are still standing, abandoned and in disrepair. Tours are offered in the days leading up to Halloween, in which the site is described as being "cursed" or "haunted".[2]',
        "A moraine-dammed lake, occurs when the terminal moraine has prevented some meltwater from leaving the valley. When a glacier retreats, there is a space left over between the retreating glacier and the piece that stayed intact which holds leftover debris (moraine). Meltwater from both glaciers seep into this space creating a ribbon-shaped lake due to the pattern of ice melt. This ice melt may cause a glacier lake outburst flood, leading to severe damage to the environment and communities nearby. Examples of moraine-dammed lakes include:",
    ]

    net_c = net_c.to("cuda")
    net_r = net_r.to("cuda")

    encoder = tiktoken.get_encoding("gpt2")
    if args.no_test_free_completion:
        print("Testing free completion")
        results_free_completion = test_free_completion(
            net_c=net_c, 
            net_r=net_r, 
            encoder=encoder, 
            sentences=sentences, 
            verbosity=args.verbosity,
            max_choose_nth_best=args.max_choose_nth_best,
            masking_rate=args.masking_rate,
            num_preference_samples=args.num_preference_samples,
        )
        if args.verbosity > 0:
            print(results_free_completion.get("summary"))
        if args.savefile:
            save_json(
                data=results_free_completion, 
                path=args.savefile,
                postfix=(
                    f"free_completion__masking_rate_"
                    f"{round(args.masking_rate * 100)}_percent"
                    f"__num_preference_samples_{args.num_preference_samples}"
                )
            )
    if args.no_test_split_sentences:
        print("Testing split sentences")
        results_split_sentences = test_split_sentences(
            net_c=net_c, 
            net_r=net_r, 
            encoder=encoder, 
            sentences=sentences, 
            verbosity=args.verbosity,
            max_choose_nth_best=args.max_choose_nth_best,
            masking_rate=args.masking_rate,
        )
        if args.verbosity > 0:
            print(results_split_sentences.get("summary"))
        if args.savefile:
            save_json(
                data=results_split_sentences, 
                path=args.savefile,
                postfix=f"split_sentences__masking_rate_{round(args.masking_rate * 100)}_percent"
            )


if __name__ == "__main__":
    main()
