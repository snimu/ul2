
import argparse
import json

import tiktoken
import torch
import safetensors.torch
from rich import print
import polars as pl

import main


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def autoregress_with_branching(
    net: torch.nn.Module,
    inputs: torch.Tensor,  # (b, s_in)
    targets: torch.Tensor,  # (b, s_out), s_out > s_in
    k: int,  # top-k tokens to choose from when branching
    branch_every: int,
    num_steps: int,
) -> pl.DataFrame:
    net.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    
    b, s_in = inputs.shape
    _, s_out = targets.shape
    
    results = dict(completion=[], losses=[], choices=[])
    
    def branch(sequence, choices, losses, step):
        if step >= num_steps:
            return
        
        # Autoregress for branch_every steps
        for i in range(branch_every):
            if step + i >= num_steps:
                break
            
            x = sequence[:, :s_in + step + i]
            y: torch.Tensor = net(x)
            probs = y[:, -1]
            
            if step + i < s_out - s_in:
                target = targets[:, s_in + step + i]
                loss = torch.nn.functional.cross_entropy(probs, target, reduction='none')
                losses.extend(loss.tolist())
            
            top_k_probs, top_k_indices = torch.topk(probs, k)
            
            if i == branch_every - 1 and step + i + 1 < num_steps:
                # Branch
                for j in range(k):
                    new_sequence = sequence.clone()
                    new_sequence[:, s_in + step + i] = top_k_indices[:, j]
                    new_choices = choices + [j + 1]
                    branch(new_sequence, new_choices, losses.copy(), step + i + 1)
            else:
                # Continue with top-1
                sequence[:, s_in + step + i] = top_k_indices[:, 0]
                choices.append(1)
    
        if step + branch_every >= num_steps:
            # End of sequence reached
            completion = tokenizer.decode(sequence[0, s_in:].tolist())
            results["completion"].append(completion)
            results["losses"].append(losses)
            results["choices"].append(choices)
    
    # Initialize sequences with inputs
    initial_sequence = torch.cat([
        inputs,
        torch.zeros((b, num_steps), dtype=inputs.dtype, device=DEVICE)
    ], dim=1)
    
    input_text = tokenizer.decode(inputs[0].tolist())
    branch(initial_sequence, [], [], 0)
    results = pl.DataFrame(results)
    results = results.with_columns(  # Add the input text as a column
        pl.Series(name="input_text", dtype=pl.Utf8, values=[input_text])
    ).select(["input_text", "completion", "losses", "choices"])  # Reorder columns
    
    return results


@torch.no_grad()
def eval_autoregress_with_branching_wikitext(
        model: torch.nn.Module, args: argparse.Namespace
) -> pl.DataFrame:
    num_eval_sequences = main.hyp['opt']['num_eval_tokens']//main.hyp['misc']['sequence_length']['max']

    results = None
    for _ in range(num_eval_sequences):
        sequence = main.get_batch(main.data, key='eval', batchsize=1, length=main.hyp['misc']['sequence_length']['max'])
        inputs, targets = main.get_causal_data(sequence, no_special_tokens=True)
        inputs = inputs[:args.num_input_tokens]
        targets = targets[:args.num_input_tokens + args.num_steps]
        loc_results = autoregress_with_branching(
            model,
            inputs=inputs,
            targets=targets,
            k=args.top_k,
            branch_every=args.branch_every,
            num_steps=args.num_steps,
        )
        if results is None:
            results = loc_results
        else:
            results = pl.concat([results, loc_results])
    
    return results


@torch.no_grad()
def eval_autoregress_with_branching_fineweb(
        model: torch.nn.Module, args: argparse.Namespace
) -> pl.DataFrame:
    dl = main.load_fineweb("fineweb", "val")
    results = None
    for batch in dl:
        inputs_batch, targets_batch = main.get_causal_data(batch, no_special_tokens=True)
        for inputs, targets in zip(inputs_batch, targets_batch):
            inputs = inputs[:args.num_input_tokens].unsqueeze(0)
            targets = targets[:args.num_input_tokens + args.num_steps].unsqueeze(0)
            loc_results = autoregress_with_branching(
                model,
                inputs=inputs,
                targets=targets,
                k=args.top_k,
                branch_every=args.branch_every,
                num_steps=args.num_steps,
            )
            if results is None:
                results = loc_results
            else:
                results = pl.concat([results, loc_results])
    
    return results


@torch.no_grad()
def load_model(model_file: str) -> torch.nn.Module:
    with open(model_file.split(".safetensors")[0]+".metadata.json", "r") as f:
        metadata = json.loads(f.read())
        for key, value in metadata.items():
            metadata[key] = bool(value) if key == "linear_value" else int(value)
    model = main.make_net(metadata)
    safetensors.torch.load_model(model, model_file)
    return model.to("cuda")



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_tests",
        action="store_true",
        help="Run the tests. FLAG."
    )
    parser.add_argument(
        "--model_file",
        type=str,
        help="The .safetensors-file to load. Include fileending. TYPE: str"
    )
    parser.add_argument(
        "--top_k",
        type=int, default=3,
        help="The number of tokens predicted by the probe. TYPE: int; DEFAULT: 3"
    )
    parser.add_argument(
        "--branch_every",
        type=int, default=2,
        help="The batch-size. TYPE: int; DEFAULT: 5"
    )
    parser.add_argument(
        "--num_steps",
        type=int, default=6,
        help="The batch-size. TYPE: int; DEFAULT: 5"
    )
    parser.add_argument("--num_input_tokens", type=int, default=256)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the results. FLAG."
    )
    parser.add_argument(
        "--dataset",
        type=str, default="wikitext", choices=["wikitext", "fineweb"],
        help="The dataset to evaluate on. TYPE: str; DEFAULT: wikitext"
    )

    args = parser.parse_args()
    return args


class MockNet(torch.nn.Module):
    def __init__(self, in_size: int=10, out_size: int=100):
        super().__init__()
        self.emb = torch.nn.Embedding(in_size, out_size, dtype=torch.bfloat16)
    
    def forward(self, x):
        x = self.emb(x)
        return x


def _check_autoregress_with_branching():
    print("Testing autoregress_with_branching")
    
    net = MockNet(10, 10)
    inputs = torch.randint(0, 10, (2, 5))
    targets = torch.randint(0, 10, (2, 11))
    num_steps = 5
    results = autoregress_with_branching(net, inputs, targets, 3, 1, num_steps)
    print(results)

    print("Passed")


def _check_eval_autoregress_with_branching():
    print("Testing eval_autoregress_with_branching")
    
    ntok = main.hyp['misc']['num_tokens'] + main.hyp['misc']['num_special_tokens']
    net = MockNet(ntok, ntok)
    args = get_args()
    results = eval_autoregress_with_branching_wikitext(net, args)
    print(results)

    print("Passed")


def main_loop():
    args = get_args()
    if args.run_tests:
        _check_autoregress_with_branching()
        _check_eval_autoregress_with_branching()
        return
    model  = load_model(args.model_file)
    if args.dataset == "wikitext":
        results = eval_autoregress_with_branching_wikitext(model, args)
    elif args.dataset == "fineweb":
        results = eval_autoregress_with_branching_fineweb(model, args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    if args.verbose:
        print(results)
    
    results.write_csv(f"autoregress_with_branching_{args.dataset}_{args.model_file[:-len('.safetensors')]}.csv")


if __name__ == "__main__":
    main_loop()
