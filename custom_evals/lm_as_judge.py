# /// script
# requires-python = "==3.12"
# dependencies = [
#   "dspy",
#   "polars",
#   "tqdm",
# ]
# ///

"""Judge completions using a language model"""

import argparse
import functools
import json
from collections import Counter
from typing import Literal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import dspy
import polars as pl


class Preference(dspy.Signature):
    """
    Two different LLMs completed the same input text. Which one is better?
    
    Judge grammar, syntax, and content. Excessive repetition is bad; Veering off subject is bad. Etc.
    """
    input_text: str = dspy.InputField(desc="The input text that the LLMs completed")
    completion_1: str = dspy.InputField()
    completion_2: str = dspy.InputField()
    better_completion: Literal["completion_1", "completion_2"] = dspy.OutputField()


def get_preference(
        input_text: str, 
        completion_r: str, 
        completion_c: str,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        n: int = 10,
        loop: tqdm = None,
        description: str = "",
        cot: bool = False,
) -> dict[Literal["R", "C"], int]:
    # DSPy setup
    module = dspy.ChainOfThought if cot else dspy.Predict
    calculate_preference = module(Preference, temperature=temperature, cache=False)
    calculate_preference = functools.partial(calculate_preference, input_text=input_text)
    dspy.settings.configure(lm=dspy.LM(f"openai/{model}"))
    # Init counter
    counts = Counter({"R": 0, "C": 0})
    # Run n times for statistically meaningful result
    for i in range(n):
        # Provide preference in both possible orders, anonymously to avert confounders
        # Ordering 1
        # Give some feedback
        if loop is not None:
            desc = description + f"; preference {i+1}/{n}, Ordering 1/2"
            loop.set_description(desc)
        # Calculate preference
        preference = calculate_preference(
            completion_1=completion_r, completion_2=completion_c
        ).better_completion
        if preference == "completion_1":
            counts["R"] += 1
        else:
            counts["C"] += 1

        # Ordering 2
        # Give some feedback
        if loop is not None:
            desc = description + f"; preference {i+1}/{n}, Ordering 2/2"
            loop.set_description(desc)
        # Calculate preference
        preference = calculate_preference(
            completion_1=completion_c, completion_2=completion_r
        ).better_completion
        if preference == "completion_1":
            counts["C"] += 1
        else:
            counts["R"] += 1
    return dict(counts)


def ablate_preferences(
        gen_tokens: int = 1024,
        gen_steps: int = 1,
        gen_temperature: float = 1.0,
        judge_temperature: float = 1.0,
        judge_model: str = "gpt-4o-mini",
        n: int = 10,
        cot: bool = False,
):
    df = pl.read_csv(
        Path("..") / "results" / "evals" / "100BT" / "custom" 
        / f"temp-{gen_temperature}_toks-{gen_tokens}_steps-{gen_steps}.csv"
    )
    input_texts = df["query"].unique()
    results = {"summary": None, "preferences": []}
    loop = tqdm(input_texts)
    for input_text in loop:
        completions_c = df.filter(pl.col("query") == input_text).get_column("completion").unique()
        completions_r = df.filter(pl.col("query") == input_text).get_column("completion").unique()
        assert len(completions_c) == len(completions_r)
        for completion_num, (completion_c, completion_r) in enumerate(zip(completions_c, completions_r)):
            description = f"Completion {completion_num+1}/{len(completions_c)}"
            preference = get_preference(
                input_text=input_text, 
                completion_r=completion_r, 
                completion_c=completion_c,
                model=judge_model,
                temperature=judge_temperature,
                n=n,
                loop=loop,
                description=description,
                cot=cot,
            )
            results["preferences"].append({
                "query": input_text,
                "completion_c": completion_c,
                "completion_r": completion_r,
                "preference": preference,
            })

    # Summarize results
    summary = Counter()
    for preference in results["preferences"]:
        summary["R"] += preference["preference"]["R"]
        summary["C"] += preference["preference"]["C"]
    summary = dict(summary)
    summary["percent_r"] = summary["R"] / (summary["R"] + summary["C"])
    summary["percent_c"] = summary["C"] / (summary["R"] + summary["C"])
    results["summary"] = summary
    return results


class GrammaticalCorrectness(dspy.Signature):
    """
    Given a query and a completion by an LLM, judge the grammatical correctness of the completion
    on a scale from 0 to 10.
    """
    query: str = dspy.InputField()
    completion: str = dspy.InputField()
    score: float = dspy.OutputField(desc="0 is worst, 10 is best")


class Repetitiveness(dspy.Signature):
    """
    Given a query and a completion by an LLM, judge the repetitiveness of the completion
    on a scale from 0 to 10.
    """
    query: str = dspy.InputField()
    completion: str = dspy.InputField()
    score: float = dspy.OutputField(desc="0 is best, 10 is worst")


class Coherence(dspy.Signature):
    """
    Given a query and a completion by an LLM, judge the coherence of the completion
    on a scale from 0 to 10.
    """
    query: str = dspy.InputField()
    completion: str = dspy.InputField()
    score: float = dspy.OutputField(desc="0 is worst, 10 is best")


class Factuality(dspy.Signature):
    """
    Given a query and a completion by an LLM, judge the factuality of the completion
    on a scale from 0 to 10.
    """
    query: str = dspy.InputField()
    completion: str = dspy.InputField()
    score: float = dspy.OutputField(desc="0 is worst, 10 is best")


def ablate_scores(
        gen_tokens: int = 1024,
        gen_steps: int = 1,
        gen_temperature: float = 1.0,
        judge_temperature: float = 1.0,
        judge_model: str = "gpt-4o-mini",
        n: int = 10,
        cot: bool = False,
        savefile: str = "results.csv",
):
    df = pl.read_csv(
        Path("..") / "results" / "evals" / "100BT" / "custom" 
        / f"temp-{gen_temperature}_toks-{gen_tokens}_steps-{gen_steps}.csv"
    )
    input_texts = df["query"].unique()
    loop = tqdm(input_texts)

    def get_score(signature, query, completion):
        module = dspy.ChainOfThought if cot else dspy.Predict
        calculate_score = module(signature, temperature=judge_temperature, cache=False)
        dspy.settings.configure(lm=dspy.LM(f"openai/{judge_model}"))
        return calculate_score(query=query, completion=completion).score

    for input_text in loop:
        completions_c = df.filter(pl.col("query") == input_text).get_column("completion").unique()
        completions_r = df.filter(pl.col("query") == input_text).get_column("completion").unique()
        assert len(completions_c) == len(completions_r)

        for completion_num, (completion_c, completion_r) in enumerate(zip(completions_c, completions_r)):
            description = f"Completion {completion_num+1}/{len(completions_c)}"

            with ThreadPoolExecutor(max_workers=n) as pool:
                loop.set_description(description + "; grammar C")
                grammar_futures_c = [
                    pool.submit(get_score, GrammaticalCorrectness, input_text, completion_c)
                    for _ in range(n)
                ]
                grammar_scores_c = [future.result() for future in grammar_futures_c]

                loop.set_description(description + "; grammar R")
                grammar_futures_r = [
                    pool.submit(get_score, GrammaticalCorrectness, input_text, completion_r)
                    for _ in range(n)
                ]
                grammar_scores_r = [future.result() for future in grammar_futures_r]

                loop.set_description(description + "; repetitiveness C")
                repetitiveness_futures_c = [
                    pool.submit(get_score, Repetitiveness, input_text, completion_c)
                    for _ in range(n)
                ]
                repetitiveness_scores_c = [future.result() for future in repetitiveness_futures_c]

                loop.set_description(description + "; repetitiveness R")
                repetitiveness_futures_r = [
                    pool.submit(get_score, Repetitiveness, input_text, completion_r)
                    for _ in range(n)
                ]
                repetitiveness_scores_r = [future.result() for future in repetitiveness_futures_r]

                loop.set_description(description + "; coherence C")
                coherence_futures_c = [
                    pool.submit(get_score, Coherence, input_text, completion_c)
                    for _ in range(n)
                ]
                coherence_scores_c = [future.result() for future in coherence_futures_c]

                loop.set_description(description + "; coherence R")
                coherence_futures_r = [
                    pool.submit(get_score, Coherence, input_text, completion_r)
                    for _ in range(n)
                ]
                coherence_scores_r = [future.result() for future in coherence_futures_r]

                loop.set_description(description + "; factuality C")
                factuality_futures_c = [
                    pool.submit(get_score, Factuality, input_text, completion_c)
                    for _ in range(n)
                ]
                factuality_scores_c = [future.result() for future in factuality_futures_c]

                loop.set_description(description + "; factuality R")
                factuality_futures_r = [
                    pool.submit(get_score, Factuality, input_text, completion_r)
                    for _ in range(n)
                ]
                factuality_scores_r = [future.result() for future in factuality_futures_r]

            results = {
                "mode": ["c"] * len(grammar_scores_c) + ["r"] * len(grammar_scores_r),
                "model": [judge_model] * (len(grammar_scores_c) + len(grammar_scores_r)),
                "temperature": [judge_temperature] * (len(grammar_scores_c) + len(grammar_scores_r)),
                "grammatical_correctness": grammar_scores_c + grammar_scores_r,
                "repetitiveness": repetitiveness_scores_c + repetitiveness_scores_r,
                "coherence": coherence_scores_c + coherence_scores_r,
                "factuality": factuality_scores_c + factuality_scores_r,
                "query": [input_text] * (len(grammar_scores_c) + len(grammar_scores_r)),
                "completion": [completion_c] * len(grammar_scores_c) + [completion_r] * len(grammar_scores_r),
            }
            df = pl.DataFrame(results)
            if Path(savefile).exists():
                with open(savefile, "ab") as f:
                    df.write_csv(f, include_header=False)
            else:
                df.write_csv(savefile)

            # Print summary
            df = pl.read_csv(savefile)  # Get all data
            summary_c = df.filter(pl.col("mode") == "c")
            summary_r = df.filter(pl.col("mode") == "r")
            loop.write(
                f"{'-' * 10}\nmode=c\ngrammar={summary_c['grammatical_correctness'].mean():.2f}\n"
                f"repetitiveness={summary_c['repetitiveness'].mean():.2f}\n"
                f"coherence={summary_c['coherence'].mean():.2f}\n"
                f"factuality={summary_c['factuality'].mean():.2f}\n"
                f"\nmode=r\ngrammar={summary_r['grammatical_correctness'].mean():.2f}\n"
                f"repetitiveness={summary_r['repetitiveness'].mean():.2f}\n"
                f"coherence={summary_r['coherence'].mean():.2f}\n"
                f"factuality={summary_r['factuality'].mean():.2f}\n"
            )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gen-tokens",
        type=int, default=1024,
        help="The number of generated tokens. TYPE: int; DEFAULT: 1024"
    )
    parser.add_argument(
        "--gen-steps",
        type=int, default=1,
        help="The number of generation steps. TYPE: int; DEFAULT: 1"
    )
    parser.add_argument(
        "--gen-temperature",
        type=float, default=1.0,
        help="The temperature to use for generation. TYPE: float; DEFAULT: 1.0"
    )
    parser.add_argument(
        "--judge-temperature",
        type=float, default=1.0,
        help="The temperature to use for judging. TYPE: float; DEFAULT: 1.0"
    )
    parser.add_argument(
        "--judge-model",
        type=str, default="gpt-4o-mini",
        help="The model to use for judging. TYPE: str; DEFAULT: gpt-4o-mini"
    )
    parser.add_argument(
        "--n",
        type=int, default=10,
        help="The number of times to run the model. TYPE: int; DEFAULT: 10"
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use chain-of-thought prompting. TYPE: FLAG"
    )
    parser.add_argument(
        "--mode",
        choices=["preferences", "scores"],
        default="preferences",
        help="The mode to run. TYPE: str; DEFAULT: preferences"
    )

    return parser.parse_args()


def main():
    args = get_args()
    if args.mode == "preferences":
        results = ablate_preferences(
            gen_tokens=args.gen_tokens,
            gen_steps=args.gen_steps,
            gen_temperature=args.gen_temperature,
            judge_temperature=args.judge_temperature,
            judge_model=args.judge_model,
            n=args.n,
            cot=args.cot,
        )
        print(results["summary"])
        savefile = f"preferences_{args.judge_model}_{args.n}tries_{args.gen_tokens}_{args.gen_steps}_{args.n}{'_WithCoT' if args.cot else ''}.csv"
        savefile = Path("..") / "results" / "evals" / "100BT" / "custom" / savefile
        with open(savefile, "w") as f:
            json.dump(results, f)
    elif args.mode == "scores":
        savefile = f"scores_{args.judge_model}_{args.n}tries_{args.gen_tokens}_{args.gen_steps}_{args.n}{'_WithCoT' if args.cot else ''}.csv"
        savefile = Path("..") / "results" / "evals" / "100BT" / "custom" / savefile
        ablate_scores(
            gen_tokens=args.gen_tokens,
            gen_steps=args.gen_steps,
            gen_temperature=args.gen_temperature,
            judge_temperature=args.judge_temperature,
            judge_model=args.judge_model,
            n=args.n,
            cot=args.cot,
            savefile=savefile,
        )


if __name__ == "__main__":
    main()
