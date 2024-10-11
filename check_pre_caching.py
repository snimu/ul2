"""Check out if the model pre-caches."""

import argparse
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import safetensors.torch
import polars as pl

from main import SpeedyLangNet, LatentAttentionBlock, max_sequence_length, get_batch, hyp, data
from inference_tests import make_net_from_name, download_model


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
        "--batchsize",
        type=int, default=5,
        help="The batch-size. TYPE: int; DEFAULT: 5"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the results. FLAG."
    )

    args = parser.parse_args()
    return args


def load_model(millions_of_params: int, mode: Literal["R", "C"]) -> SpeedyLangNet:
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
    
    net = make_net_from_name(model_name)
    model_path = download_model(model_name)
    safetensors.torch.load_model(net, model_path, device="cuda")
    return net


@dataclass
class Probe:
    probe: nn.Linear
    optimizer: torch.optim.AdamW


def make_probes(model: SpeedyLangNet, num_tokens_predicted: int) -> dict[int, list[Probe]]:
    global metadata
    probes = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, LatentAttentionBlock):
            continue
        
        probes[module_name] = []
        for _ in range(num_tokens_predicted):
            probe = nn.Linear(
                in_features=metadata["width"],
                out_features=metadata["num_tokens"],
                bias=False,
                device="cuda",
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
) -> list[Probe]:
    global metadata
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    for step in range(num_steps):
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
                loss = loss_fn(probe.probe(inp).flatten(0, 1), outp.flatten(0, 1))
                loss.backward()
                probe.optimizer.step()

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
        a: {t: 0.0 for t in range(num_tokens_predicted)}
        for a in range(len(model.net_dict['attn_layers'])) 
    }
    accs = {
        a: {t: 0.0 for t in range(num_tokens_predicted)}
        for a in range(len(model.net_dict['attn_layers'])) 
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
                outp = targets[:, :-i]
                probe = probes[module_name][-i]
                preds = probe.probe(inp)
                loss = loss_fn(preds.flatten(0, 1).float(), outp.flatten(0, 1)).item()
                losses[module_name][i-1] += 1./num_eval_steps * loss
                acc = (preds.argmax(-1) == targets).float().mean()
                accs[module_name][i-1] += 1./num_eval_steps * acc
                targets = targets.roll(-1, dims=1)

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
    model = load_model(args.millions_of_params, args.mode)
    probes = make_probes(model, args.num_tokens_predicted)
    probes = train_probes(model, 1000, args.batchsize, args.num_tokens_predicted, probes)
    results = eval_probes(model, args.batchsize, args.num_tokens_predicted, probes)
    if args.verbose:
        print(results)
    
    results.write_csv(f"probes_results{args.model_name}.csv")


if __name__ == "__main__":
    main()
