"""Check out if the model pre-caches."""

import argparse
from dataclasses import dataclass
from typing import Literal
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
import safetensors.torch
import polars as pl
from huggingface_hub import hf_hub_download

from main import SpeedyLangNet, LatentAttentionBlock, max_sequence_length, get_batch, hyp, data, make_net


hyp = hyp


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--millions_of_params",
        type=int, default=773, choices=[773, 240, 1300],
        help="The number of parameters of the model. TYPE: int; DEFAULT: 773"
    )
    parser.add_argument(
        "--mode",
        type=str, default="R", choices=["R", "C"],
        help="The mode of the model. TYPE: str; DEFAULT: R"
    )
    parser.add_argument(
        "--num_tokens_predicted",
        type=int, default=3,
        help="The number of tokens predicted by the probe. TYPE: int; DEFAULT: 3"
    )
    parser.add_argument(
        "--train_batchsize",
        type=int, default=5,
        help="The batch-size. TYPE: int; DEFAULT: 5"
    )
    parser.add_argument(
        "--val_batchsize",
        type=int, default=5,
        help="The batch-size. TYPE: int; DEFAULT: 5"
    )
    parser.add_argument(
        "--num_steps",
        type=int, default=1000,
        help="The number of steps to train the probes. TYPE: int; DEFAULT: 1000"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the results. FLAG."
    )

    args = parser.parse_args()
    return args


def get_model_name(millions_of_params: int, mode: Literal["R", "C"]) -> str:
    match (mode, millions_of_params):
        case ("R", 773):
            model_name = "snimu/causal-ul2-R-fineweb10BT-773M-26heads-lr090"
        case ("R", 240):
            model_name = "snimu/causal-ul2-R-fineweb10BT-240M-16heads-lr090"
        case ("R", 1300):
            model_name = "snimu/causal-ul2-R-fineweb10BT-1300M-32heads-lr090"
        case ("C", 773):
            model_name = "snimu/causal-ul2-C-fineweb10BT-773M-26heads-lr090"
        case ("C", 240):
            model_name = "snimu/causal-ul2-C-fineweb10BT-240M-16heads-lr090"
        case ("C", 1300):
            model_name = "snimu/causal-ul2-C-fineweb10BT-1300M-32heads-lr090"
    return model_name


def get_depth_width(name: str) -> tuple[int, int]:
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
    return depth, width


def make_net_from_name(name: str) -> SpeedyLangNet:
    depth, width = get_depth_width(name)
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


def load_model(millions_of_params: int, mode: Literal["R", "C"]) -> SpeedyLangNet:
    model_name = get_model_name(millions_of_params, mode)
    net = make_net_from_name(model_name)
    model_path = download_model(model_name)
    safetensors.torch.load_model(net, model_path, device="cuda")
    return net


@dataclass
class Probe:
    probe: nn.Linear
    optimizer: torch.optim.AdamW


def make_probes(
        model: SpeedyLangNet, 
        model_name: str,
        num_tokens_predicted: int,
) -> dict[int, list[Probe]]:
    global metadata
    _, width = get_depth_width(model_name)
    total_num_tokens = 50310
    probes = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, LatentAttentionBlock):
            continue
        
        probes[module_name] = []
        for _ in range(num_tokens_predicted):
            probe = nn.Linear(
                in_features=width,
                out_features=total_num_tokens,
                bias=False,
                device="cpu",
                dtype=torch.bfloat16,
            )
            optimizer = torch.optim.AdamW(probe.parameters())
            probes[module_name].append(Probe(probe=probe, optimizer=optimizer))

    return probes


def get_xy(batchsize: int, length: int, num_tokens_predicted: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = get_batch(data, key='train', batchsize=batchsize, length=length)
    inputs, targets = sequence[:, :-1], sequence[:, 1:] # reslice to get our input tokens and our shifted-by-1 targets
    return inputs, targets


def train_probes(
        model: SpeedyLangNet,
        num_steps: int, 
        batchsize: int, 
        num_tokens_predicted: int, 
        probes: dict[int, list[Probe]],
        verbose: bool = True,
) -> list[Probe]:
    global metadata
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    for step in tqdm(range(num_steps), disable=not verbose):
        inputs, targets = get_xy(batchsize, max_sequence_length, num_tokens_predicted)
        attn_mask = model.make_mask(inputs, "causal")

        with torch.no_grad():
            x = model.net_dict['embedding'](inputs)
        for module_name, module in model.named_modules():
            if not isinstance(module, LatentAttentionBlock):
                continue
            with torch.no_grad():
                x = module(x, attn_mask)
                preds = model.net_dict['norm'](x)
            for i in range(1, num_tokens_predicted+1):
                inp = preds[:, :-i]
                outp = targets[:, :-i]
                probe = probes[module_name][-i]
                probe.optimizer.zero_grad()
                loss = loss_fn(probe.probe.to("cuda")(inp).flatten(0, 1), outp.flatten(0, 1))
                loss.backward()
                probe.optimizer.step()
                probes[module_name][-i].probe = probe.probe.to("cpu")

    return probes


@torch.no_grad()
def eval_probes(
        model: SpeedyLangNet,
        batchsize: int,
        num_tokens_predicted: int,
        probes: dict[int, list[Probe]]
) -> pl.DataFrame:
    num_eval_sequences       = hyp['opt']['num_eval_tokens']//hyp['misc']['sequence_length']['max']
    num_eval_steps           = num_eval_sequences//batchsize

    losses = {
        module_name: {t: 0.0 for t in range(num_tokens_predicted)}
        for module_name, module in model.named_modules()
        if isinstance(module, LatentAttentionBlock) 
    }
    accs = {
        module_name: {t: 0.0 for t in range(num_tokens_predicted)}
        for module_name, module in model.named_modules()
        if isinstance(module, LatentAttentionBlock)
    }
    loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)

    for _ in range(num_eval_steps):
        # Do the forward pass step by step.
        inputs, targets = get_xy(batchsize=batchsize, length=hyp['misc']['sequence_length']['max'], num_tokens_predicted=num_tokens_predicted)
        attn_mask = model.make_mask(inputs, "causal")
        x = model.net_dict['embedding'](inputs)

        for module_name, module in model.named_modules():
            if not isinstance(module, LatentAttentionBlock):
                continue
            x = module(x, attn_mask)
            x = model.net_dict['norm'](x)
            for i in range(1, num_tokens_predicted+1):
                inp = x[:, :-i]
                target = targets[:, :-i]
                probe = probes[module_name][-i]
                preds = probe.probe.to("cuda")(inp)
                loss = loss_fn(preds.flatten(0, 1).float(), target.flatten(0, 1)).item()
                losses[module_name][i-1] += 1./num_eval_steps * loss
                acc = (preds.argmax(-1).squeeze() == target).float().mean()
                accs[module_name][i-1] += 1./num_eval_steps * acc
                targets = targets.roll(-1, dims=1)
                probes[module_name][-i].probe = probe.probe.to("cpu")

    # Flatten the results and turn them into a DataFrame
    results = {
        "module_name": [],
        "token_num": [],
        "loss": [],
        "accuracy": []
    }
    for module_name, module in model.named_modules():
        if not isinstance(module, LatentAttentionBlock):
            continue
        for token_num in range(num_tokens_predicted):
            results["module_name"].append(module_name)
            results["token_num"].append(token_num+1)
            results["loss"].append(losses[module_name][token_num])
            results["accuracy"].append(accs[module_name][token_num])

    return pl.DataFrame(results)


def main():
    args = get_args()
    if args.verbose:
        print(f"\nWorking on {get_model_name(args.millions_of_params, args.mode).split('/')[1]}\n")
    model = load_model(args.millions_of_params, args.mode)
    probes = make_probes(model, get_model_name(args.millions_of_params, args.mode), args.num_tokens_predicted)
    probes = train_probes(model, args.num_steps, args.train_batchsize, args.num_tokens_predicted, probes, args.verbose)
    results = eval_probes(model, args.val_batchsize, args.num_tokens_predicted, probes)
    if args.verbose:
        print(results)
    
    results.write_csv(f"probes_results{get_model_name(args.millions_of_params, args.mode).split('/')[1]}.csv")


if __name__ == "__main__":
    main()
