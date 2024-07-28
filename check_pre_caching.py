"""Check out if the model pre-caches."""

import argparse
import json
from dataclasses import dataclass

import torch
from torch import nn
import safetensors.torch
import polars as pl

from main import SpeedyLangNet, LatentAttentionBlock, make_net, get_batch, hyp, data


hyp = hyp
metadata = None


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_file",
        type=str,
        help="The .safetensors-file to load. Include fileending. TYPE: str"
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


def load_model(model_file: str) -> SpeedyLangNet:
    global metadata
    with open(model_file.split(".safetensors")[0]+".metadata.json", "r") as f:
        metadata = json.loads(f.read())
        for key, value in metadata.items():
            metadata[key] = bool(value) if key == "linear_value" else int(value)
    model = make_net(metadata)
    safetensors.torch.load_model(model, model_file)
    return model.to("cuda")


@dataclass
class Probe:
    probe: nn.Linear
    optimizer: torch.optim.AdamW


def make_probes(model: SpeedyLangNet, num_tokens_predicted: int) -> dict[int, list[Probe]]:
    global metadata
    probes = {}
    attn_num = 0
    for module in model.modules():
        if not isinstance(module, LatentAttentionBlock):
            continue
        
        probes[attn_num] = []
        for _ in range(num_tokens_predicted):
            probe = nn.Linear(
                in_features=metadata["width"],
                out_features=metadata["num_tokens"],
                bias=False,
                device="cuda",
                dtype=torch.bfloat16,
            )
            optimizer = torch.optim.AdamW(probe.parameters())
            probes[attn_num].append(Probe(probe=probe, optimizer=optimizer))
        attn_num += 1

    return probes


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
        sequence = get_batch(data, key="train", batchsize=batchsize, length=metadata['max_sequence_length'])
        inputs = sequence
        targets = sequence.roll(-1, dims=1)

        attn_mask = model.make_mask(inputs, "causal")
        with torch.no_grad():
            x = model.net_dict['embedding'](inputs)
        for attn_num, attn_block in enumerate(model.net_dict['attn_layers']):
            with torch.no_grad():
                x = attn_block(x, attn_mask)
                preds = model.net_dict['norm'](x)
            for i in range(1, num_tokens_predicted+1):
                probe = probes[attn_num][i-1]
                probe.optimizer.zero_grad()
                loss = loss_fn(probe.probe(preds).flatten(0, 1), targets.flatten(0, 1))
                loss.backward()
                probe.optimizer.step()
                targets = targets.roll(-1, dims=1)

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
        sequence = get_batch(data, key='eval', batchsize=batchsize, length=hyp['misc']['sequence_length']['max'])
        inputs = sequence
        targets = sequence.roll(-1, dims=1)

        attn_mask = model.make_mask(inputs, "causal")
        x = model.net_dict['embedding'](inputs)

        for attn_num, attn_block in enumerate(model.net_dict['attn_layers']):
            x = attn_block(x, attn_mask)
            x = model.net_dict['norm'](x)
            for i in range(1, num_tokens_predicted+1):
                probe = probes[attn_num][i-1]
                preds = probe.probe(x)
                loss = loss_fn(preds.flatten(0, 1).float(), targets.flatten(0, 1)).item()
                losses[attn_num][i-1] += 1./num_eval_steps * loss
                acc = (preds.argmax(-1) == targets).float().mean()
                accs[attn_num][i-1] += 1./num_eval_steps * acc
                targets = targets.roll(-1, dims=1)

    # Flatten the results and turn them into a DataFrame
    results = {
        "attn_num": [],
        "token_num": [],
        "loss": [],
        "accuracy": []
    }
    for attn_num in range(len(model.net_dict['attn_layers'])):
        for token_num in range(num_tokens_predicted):
            results["attn_num"].append(attn_num)
            results["token_num"].append(token_num)
            results["loss"].append(losses[attn_num][token_num])
            results["accuracy"].append(accs[attn_num][token_num])

    return pl.DataFrame(results)


def main():
    args = get_args()
    model = load_model(args.model_file)
    probes = make_probes(model, args.num_tokens_predicted)
    probes = train_probes(model, 1000, args.batchsize, args.num_tokens_predicted, probes)
    results = eval_probes(model, args.batchsize, args.num_tokens_predicted, probes)
    if args.verbose:
        print(results)
    
    results.write_csv(f"probes_results{args.model_file[:-len('.safetensors')]}.csv")


if __name__ == "__main__":
    main()
