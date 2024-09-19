
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
        net, encoder, query: str, max_gen_tokens: int = 128, until: list[str] | None = None,
        choose_nth_best: int = 1,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    # Encode the input tokens
    input_ids = encoder.encode_ordinary(query)
    input_ids = torch.tensor(input_ids, device="cuda", dtype=torch.int).unsqueeze(0)
    input_len = input_ids.shape[1]
    
    # Generate the output tokens
    output_str = []
    all_ids = input_ids
    for _ in range(max_gen_tokens):
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


def test_free_completion(
        net_c, net_r, encoder, sentences, verbosity: int = 1
) -> dict[str, Any]:
    results = dict()
    for sentence in tqdm(sentences, disable=not verbosity):
        completion_c1, _, _ = generate(net_c, encoder, sentence, max_gen_tokens=50)
        size_ratio_completion_c1, size_ratio_full_c1 = calc_ratio_compression(completion_c1, sentence+completion_c1)
        num_unique_words_c1 = len(set(completion_c1.split()))
        num_unique_tokens_c1 = len(set(encoder.encode_ordinary(completion_c1)))

        completion_r1, _, _ = generate(net_r, encoder, sentence, max_gen_tokens=50)
        size_ratio_completion_r1, size_ratio_full_r1 = calc_ratio_compression(completion_r1, sentence+completion_r1)
        num_unique_words_r1 = len(set(completion_r1.split()))
        num_unique_tokens_r1 = len(set(encoder.encode_ordinary(completion_r1)))

        completion_c2, _, _ = generate(net_c, encoder, sentence, max_gen_tokens=50, choose_nth_best=2)
        size_ratio_completion_c2, size_ratio_full_c2 = calc_ratio_compression(completion_c2, sentence+completion_c2)
        num_unique_words_c2 = len(set(completion_c2.split()))
        num_unique_tokens_c2 = len(set(encoder.encode_ordinary(completion_c2)))

        completion_r2, _, _ = generate(net_r, encoder, sentence, max_gen_tokens=50, choose_nth_best=2)
        size_ratio_completion_r2, size_ratio_full_r2 = calc_ratio_compression(completion_r2, sentence+completion_r2)
        num_unique_words_r2 = len(set(completion_r2.split()))
        num_unique_tokens_r2 = len(set(encoder.encode_ordinary(completion_r2)))

        completion_c3, _, _ = generate(net_c, encoder, sentence, max_gen_tokens=50, choose_nth_best=3)
        size_ratio_completion_c3, size_ratio_full_c3 = calc_ratio_compression(completion_c3, sentence+completion_c3)
        num_unique_words_c3 = len(set(completion_c3.split()))
        num_unique_tokens_c3 = len(set(encoder.encode_ordinary(completion_c3)))

        completion_r3, _, _ = generate(net_r, encoder, sentence, max_gen_tokens=50, choose_nth_best=3)
        size_ratio_completion_r3, size_ratio_full_r3 = calc_ratio_compression(completion_r3, sentence+completion_r3)
        num_unique_words_r3 = len(set(completion_r3.split()))
        num_unique_tokens_r3 = len(set(encoder.encode_ordinary(completion_r3)))

        results[sentence] = {
            "completion_c1": completion_c1,
            "size_ratio_completion_c1": size_ratio_completion_c1,
            "size_ratio_full_c1": size_ratio_full_c1,
            "num_unique_words_c1": num_unique_words_c1,
            "num_unique_tokens_c1": num_unique_tokens_c1,
            "completion_c2": completion_c2,
            "size_ratio_completion_c2": size_ratio_completion_c2,
            "size_ratio_full_c2": size_ratio_full_c2,
            "num_unique_words_c2": num_unique_words_c2,
            "num_unique_tokens_c2": num_unique_tokens_c2,
            "completion_c3": completion_c3,
            "size_ratio_completion_c3": size_ratio_completion_c3,
            "size_ratio_full_c3": size_ratio_full_c3,
            "num_unique_words_c3": num_unique_words_c3,
            "num_unique_tokens_c3": num_unique_tokens_c3,
            "completion_r1": completion_r1,
            "size_ratio_completion_r1": size_ratio_completion_r1,
            "size_ratio_full_r1": size_ratio_full_r1,
            "num_unique_words_r1": num_unique_words_r1,
            "num_unique_tokens_r1": num_unique_tokens_r1,
            "completion_r2": completion_r2,
            "size_ratio_completion_r2": size_ratio_completion_r2,
            "size_ratio_full_r2": size_ratio_full_r2,
            "num_unique_words_r2": num_unique_words_r2,
            "num_unique_tokens_r2": num_unique_tokens_r2,
            "completion_r3": completion_r3,
            "size_ratio_completion_r3": size_ratio_completion_r3,
            "size_ratio_full_r3": size_ratio_full_r3,
            "num_unique_words_r3": num_unique_words_r3,
            "num_unique_tokens_r3": num_unique_tokens_r3,
        }

        if verbosity > 1:
            print(
                f"\n\n{sentence=}\n\n"
                f"{results[sentence]=}\n\n"
            )
    
    summary = {
        "mean_size_ratio_completion_c1": torch.tensor([results[sentence]["size_ratio_completion_c1"] for sentence in results]).float().mean().item(),
        "mean_size_ratio_full_c1": torch.tensor([results[sentence]["size_ratio_full_c1"] for sentence in results]).float().mean().item(),
        "mean_num_unique_words_c1": torch.tensor([results[sentence]["num_unique_words_c1"] for sentence in results]).float().mean().item(),
        "mean_num_unique_tokens_c1": torch.tensor([results[sentence]["num_unique_tokens_c1"] for sentence in results]).float().mean().item(),
        "mean_size_ratio_completion_c2": torch.tensor([results[sentence]["size_ratio_completion_c2"] for sentence in results]).float().mean().item(),
        "mean_size_ratio_full_c2": torch.tensor([results[sentence]["size_ratio_full_c2"] for sentence in results]).float().mean().item(),
        "mean_num_unique_words_c2": torch.tensor([results[sentence]["num_unique_words_c2"] for sentence in results]).float().mean().item(),
        "mean_num_unique_tokens_c2": torch.tensor([results[sentence]["num_unique_tokens_c2"] for sentence in results]).float().mean().item(),
        "mean_size_ratio_completion_c3": torch.tensor([results[sentence]["size_ratio_completion_c3"] for sentence in results]).float().mean().item(),
        "mean_size_ratio_full_c3": torch.tensor([results[sentence]["size_ratio_full_c3"] for sentence in results]).float().mean().item(),
        "mean_num_unique_words_c3": torch.tensor([results[sentence]["num_unique_words_c3"] for sentence in results]).float().mean().item(),
        "mean_num_unique_tokens_c3": torch.tensor([results[sentence]["num_unique_tokens_c3"] for sentence in results]).float().mean().item(),
        "mean_size_ratio_completion_r1": torch.tensor([results[sentence]["size_ratio_completion_r1"] for sentence in results]).float().mean().item(),
        "mean_size_ratio_full_r1": torch.tensor([results[sentence]["size_ratio_full_r1"] for sentence in results]).float().mean().item(),
        "mean_num_unique_words_r1": torch.tensor([results[sentence]["num_unique_words_r1"] for sentence in results]).float().mean().item(),
        "mean_num_unique_tokens_r1": torch.tensor([results[sentence]["num_unique_tokens_r1"] for sentence in results]).float().mean().item(),
        "mean_size_ratio_completion_r2": torch.tensor([results[sentence]["size_ratio_completion_r2"] for sentence in results]).float().mean().item(),
        "mean_size_ratio_full_r2": torch.tensor([results[sentence]["size_ratio_full_r2"] for sentence in results]).float().mean().item(),
        "mean_num_unique_words_r2": torch.tensor([results[sentence]["num_unique_words_r2"] for sentence in results]).float().mean().item(),
        "mean_num_unique_tokens_r2": torch.tensor([results[sentence]["num_unique_tokens_r2"] for sentence in results]).float().mean().item(),
        "mean_size_ratio_completion_r3": torch.tensor([results[sentence]["size_ratio_completion_r3"] for sentence in results]).float().mean().item(),
        "mean_size_ratio_full_r3": torch.tensor([results[sentence]["size_ratio_full_r3"] for sentence in results]).float().mean().item(),
        "mean_num_unique_words_r3": torch.tensor([results[sentence]["num_unique_words_r3"] for sentence in results]).float().mean().item(),
        "mean_num_unique_tokens_r3": torch.tensor([results[sentence]["num_unique_tokens_r3"] for sentence in results]).float().mean().item(),
    }
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
) -> dict[str, dict[str, Any | list[dict[str, Any]]]]: 
    results = dict()
    ce_loss  = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    l2_loss  = nn.MSELoss(reduction='mean')
    loop = tqdm(sentences, disable=not verbosity)
    for sentence in loop:
        results[sentence] = dict(details=[])
        input_ids = encoder.encode_ordinary(sentence)
        max_completion_len = len(input_ids) // 2
        max_completion_len = max(
            min_completion_len, 
            max_completion_len - (max_completion_len % step_between_completion_lengths)
        )
        completion_lengths = list(range(min_completion_len, max_completion_len+1, step_between_completion_lengths))
        for i, completion_length in enumerate(completion_lengths):
            loop.set_description(f"Splitting sentence {i+1}/{len(completion_lengths)}")
            partial_input_ids = input_ids[:-completion_length]
            partial_sentence = encoder.decode(partial_input_ids)
            partial_input_ids = torch.tensor(partial_input_ids).to("cuda")
            completion_c1, _, logprobs_c1 = generate(net_c, encoder, partial_sentence, max_gen_tokens=completion_length)
            completion_c2, _, logprobs_c2 = generate(net_c, encoder, partial_sentence, max_gen_tokens=completion_length, choose_nth_best=2)
            completion_c3, _, logprobs_c3 = generate(net_c, encoder, partial_sentence, max_gen_tokens=completion_length, choose_nth_best=3)
            completion_r1, _, logprobs_r1 = generate(net_r, encoder, partial_sentence, max_gen_tokens=completion_length)
            completion_r2, _, logprobs_r2 = generate(net_r, encoder, partial_sentence, max_gen_tokens=completion_length, choose_nth_best=2)
            completion_r3, _, logprobs_r3 = generate(net_r, encoder, partial_sentence, max_gen_tokens=completion_length, choose_nth_best=3)

            results[sentence]["details"].append(
                dict(
                    partial_sentence=partial_sentence,
                    completion_c1=completion_c1,
                    edit_distance_c1=Levenshtein.distance(partial_input_ids.tolist(), encoder.encode_ordinary(completion_c1)),
                    acc_c1=(encoder.encode_ordinary(completion_c1) == partial_input_ids).float().mean().item(),
                    ce_loss_c1=ce_loss(logprobs_c1.flatten(0, 1), partial_input_ids.flatten(0, 1)).item(),
                    completion_c2=completion_c2,
                    edit_distance_c2=Levenshtein.distance(partial_input_ids.tolist(), encoder.encode_ordinary(completion_c2)),
                    acc_c2=(encoder.encode_ordinary(completion_c2) == partial_input_ids).float().mean().item(),
                    ce_loss_c2=ce_loss(logprobs_c2.flatten(0, 1), partial_input_ids.flatten(0, 1)).item(),
                    completion_c3=completion_c3,
                    edit_distance_c3=Levenshtein.distance(partial_input_ids.tolist(), encoder.encode_ordinary(completion_c3)),
                    acc_c3=(encoder.encode_ordinary(completion_c3) == partial_input_ids).float().mean().item(),
                    ce_loss_c3=ce_loss(logprobs_c3.flatten(0, 1), partial_input_ids.flatten(0, 1)).item(),
                    completion_r1=completion_r1,
                    edit_distance_r1=Levenshtein.distance(partial_input_ids.tolist(), encoder.encode_ordinary(completion_r1)),
                    acc_r1=(encoder.encode_ordinary(completion_r1) == partial_input_ids).float().mean().item(),
                    ce_loss_r1=ce_loss(logprobs_r1.flatten(0, 1), partial_input_ids.flatten(0, 1)).item(),
                    completion_r2=completion_r2,
                    edit_distance_r2=Levenshtein.distance(partial_input_ids.tolist(), encoder.encode_ordinary(completion_r2)),
                    acc_r2=(encoder.encode_ordinary(completion_r2) == partial_input_ids).float().mean().item(),
                    ce_loss_r2=ce_loss(logprobs_r2.flatten(0, 1), partial_input_ids.flatten(0, 1)).item(),
                    completion_r3=completion_r3,
                    edit_distance_r3=Levenshtein.distance(partial_input_ids.tolist(), encoder.encode_ordinary(completion_r3)),
                    acc_r3=(encoder.encode_ordinary(completion_r3) == partial_input_ids).float().mean().item(),
                    ce_loss_r3=ce_loss(logprobs_r3.flatten(0, 1), partial_input_ids.flatten(0, 1)).item(),
                    l2_loss_c1_c2=l2_loss(logprobs_c1, logprobs_c2).item(),
                    l2_loss_c1_c3=l2_loss(logprobs_c1, logprobs_c3).item(),
                    l2_loss_c2_c3=l2_loss(logprobs_c2, logprobs_c3).item(),
                    l2_loss_r1_r2=l2_loss(logprobs_r1, logprobs_r2).item(),
                    l2_loss_r1_r3=l2_loss(logprobs_r1, logprobs_r3).item(),
                    l2_loss_r2_r3=l2_loss(logprobs_r2, logprobs_r3).item(),
                    l2_loss_c1_r1=l2_loss(logprobs_c1, logprobs_r1).item(),
                    l2_loss_c2_r2=l2_loss(logprobs_c2, logprobs_r2).item(),
                    l2_loss_c3_r3=l2_loss(logprobs_c3, logprobs_r3).item(),
                )
            )

            if verbosity > 1:
                print(results[sentence]["details"][-1])
        
        results[sentence]["mean_acc_c1"] = sum([d["acc_c1"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_ce_loss_c1"] = sum([d["ce_loss_c1"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_edit_distance_c1"] = sum([d["edit_distance_c1"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_acc_c2"] = sum([d["acc_c2"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_ce_loss_c2"] = sum([d["ce_loss_c2"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_edit_distance_c2"] = sum([d["edit_distance_c2"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_acc_c3"] = sum([d["acc_c3"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_ce_loss_c3"] = sum([d["ce_loss_c3"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_edit_distance_c3"] = sum([d["edit_distance_c3"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_acc_r1"] = sum([d["acc_r1"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_ce_loss_r1"] = sum([d["ce_loss_r1"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_edit_distance_r1"] = sum([d["edit_distance_r1"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_acc_r2"] = sum([d["acc_r2"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_ce_loss_r2"] = sum([d["ce_loss_r2"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_edit_distance_r2"] = sum([d["edit_distance_r2"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_acc_r3"] = sum([d["acc_r3"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_ce_loss_r3"] = sum([d["ce_loss_r3"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_edit_distance_r3"] = sum([d["edit_distance_r3"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_l2_loss_c1_c2"] = sum([d["l2_loss_c1_c2"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_l2_loss_c1_c3"] = sum([d["l2_loss_c1_c3"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_l2_loss_c2_c3"] = sum([d["l2_loss_c2_c3"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_l2_loss_r1_r2"] = sum([d["l2_loss_r1_r2"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_l2_loss_r1_r3"] = sum([d["l2_loss_r1_r3"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_l2_loss_r2_r3"] = sum([d["l2_loss_r2_r3"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_l2_loss_c1_r1"] = sum([d["l2_loss_c1_r1"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_l2_loss_c2_r2"] = sum([d["l2_loss_c2_r2"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
        results[sentence]["mean_l2_loss_c3_r3"] = sum([d["l2_loss_c3_r3"] for d in results[sentence]["details"]]) / len(results[sentence]["details"])
    
    summary = {
        "mean_acc_c1": torch.tensor([results[sentence]["mean_acc_c1"] for sentence in results]).float().mean().item(),
        "mean_ce_loss_c1": torch.tensor([results[sentence]["mean_ce_loss_c1"] for sentence in results]).float().mean().item(),
        "mean_edit_distance_c1": torch.tensor([results[sentence]["mean_edit_distance_c1"] for sentence in results]).float().mean().item(),
        "mean_acc_c2": torch.tensor([results[sentence]["mean_acc_c2"] for sentence in results]).float().mean().item(),
        "mean_ce_loss_c2": torch.tensor([results[sentence]["mean_ce_loss_c2"] for sentence in results]).float().mean().item(),
        "mean_edit_distance_c2": torch.tensor([results[sentence]["mean_edit_distance_c2"] for sentence in results]).float().mean().item(),
        "mean_acc_c3": torch.tensor([results[sentence]["mean_acc_c3"] for sentence in results]).float().mean().item(),
        "mean_ce_loss_c3": torch.tensor([results[sentence]["mean_ce_loss_c3"] for sentence in results]).float().mean().item(),
        "mean_edit_distance_c3": torch.tensor([results[sentence]["mean_edit_distance_c3"] for sentence in results]).float().mean().item(),
        "mean_acc_r1": torch.tensor([results[sentence]["mean_acc_r1"] for sentence in results]).float().mean().item(),
        "mean_ce_loss_r1": torch.tensor([results[sentence]["mean_ce_loss_r1"] for sentence in results]).float().mean().item(),
        "mean_edit_distance_r1": torch.tensor([results[sentence]["mean_edit_distance_r1"] for sentence in results]).float().mean().item(),
        "mean_acc_r2": torch.tensor([results[sentence]["mean_acc_r2"] for sentence in results]).float().mean().item(),
        "mean_ce_loss_r2": torch.tensor([results[sentence]["mean_ce_loss_r2"] for sentence in results]).float().mean().item(),
        "mean_edit_distance_r2": torch.tensor([results[sentence]["mean_edit_distance_r2"] for sentence in results]).float().mean().item(),
        "mean_acc_r3": torch.tensor([results[sentence]["mean_acc_r3"] for sentence in results]).float().mean().item(),
        "mean_ce_loss_r3": torch.tensor([results[sentence]["mean_ce_loss_r3"] for sentence in results]).float().mean().item(),
        "mean_edit_distance_r3": torch.tensor([results[sentence]["mean_edit_distance_r3"] for sentence in results]).float().mean().item(),
        "mean_l2_loss_c1_c2": torch.tensor([results[sentence]["mean_l2_loss_c1_c2"] for sentence in results]).float().mean().item(),
        "mean_l2_loss_c1_c3": torch.tensor([results[sentence]["mean_l2_loss_c1_c3"] for sentence in results]).float().mean().item(),
        "mean_l2_loss_c2_c3": torch.tensor([results[sentence]["mean_l2_loss_c2_c3"] for sentence in results]).float().mean().item(),
        "mean_l2_loss_r1_r2": torch.tensor([results[sentence]["mean_l2_loss_r1_r2"] for sentence in results]).float().mean().item(),
        "mean_l2_loss_r1_r3": torch.tensor([results[sentence]["mean_l2_loss_r1_r3"] for sentence in results]).float().mean().item(),
        "mean_l2_loss_r2_r3": torch.tensor([results[sentence]["mean_l2_loss_r2_r3"] for sentence in results]).float().mean().item(),
        "mean_l2_loss_c1_r1": torch.tensor([results[sentence]["mean_l2_loss_c1_r1"] for sentence in results]).float().mean().item(),
        "mean_l2_loss_c2_r2": torch.tensor([results[sentence]["mean_l2_loss_c2_r2"] for sentence in results]).float().mean().item(),
        "mean_l2_loss_c3_r3": torch.tensor([results[sentence]["mean_l2_loss_c3_r3"] for sentence in results]).float().mean().item(),
    }
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

    return parser.parse_args()


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
    ]

    net_c = net_c.to("cuda")
    net_r = net_r.to("cuda")

    encoder = tiktoken.get_encoding("gpt2")
    results_free_completion = test_free_completion(net_c, net_r, encoder, sentences, verbosity=args.verbosity)
    results_split_sentences = test_split_sentences(net_c, net_r, encoder, sentences, verbosity=args.verbosity)
    
    if args.verbosity > 0:
        print(results_free_completion.get("summary"))
        print(results_split_sentences.get("summary"))

    if args.savefile:
        with open(args.savefile, "w") as f:
            json.dump(results_free_completion, f, indent=2)
            json.dump(results_split_sentences, f, indent=2)


if __name__ == "__main__":
    main()
