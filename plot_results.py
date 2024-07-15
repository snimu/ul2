
import ast
from typing import Literal

import colorsys
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from rich import print
import rich.table
import rich.text
import rich.style
from tabulate import tabulate


def close_plt() -> None:
    plt.cla()
    plt.clf()
    plt.close()


def series_to_array(series: pl.Series | str) -> np.ndarray:
    try:
        return np.array(ast.literal_eval(series[0]))
    except SyntaxError:
        try:
            return np.array(ast.literal_eval(series))
        except ValueError:
            series = series.replace("nan", "0")
            series = series.replace("inf", str(float(np.finfo(np.float16).max)))
            return np.array(ast.literal_eval(series))


def format_num_params(num_params: int, round_to_digits: int = 1) -> str:
    if num_params < 1_000:
        pnum = str(round(num_params, max(0, round_to_digits)))
        scalar = ""
    elif num_params < 1_000_000:
        pnum = f"{round(num_params/1_000, max(0, round_to_digits))}"
        scalar = "k"
    elif num_params < 1_000_000_000:
        pnum = f"{round(num_params/1_000_000, max(0, round_to_digits))}"
        scalar = "M"
    else:
        pnum = f"{round(num_params/1_000_000_000, max(0, round_to_digits))}"
        scalar = "B"

    before_dot = pnum.split(".")[0]
    after_dot = pnum.split(".")[1] if "." in pnum else ""
    after_dot = "" if after_dot and (round_to_digits <= 0) else after_dot
    after_dot = "" if after_dot and (int(after_dot) == 0) else after_dot
    after_dot = "." + after_dot if after_dot else ""

    return f"{before_dot}{after_dot}{scalar}"


def load_xs_ys_avg_y(
        file: str,
        model_scale: float | None = None,
        depth: int | None = None,
        width: int | None = None,
        num_params: int | None = None,
        linear_value: bool | None = None,
        num_heads: int | None = None,
        run_num: int | None = None,
        seed: int | None = None,
        ul2: bool | None = None,
        causal_denoisers: bool | None = None,
        noncausal_masking: bool | None = None,
        randomize_denoiser_settings: bool | None = None,
        randomize_mask_width: bool | None = None,
        no_special_tokens: bool | None = None,
        alternate_denoisers: bool | None = None,
        causal_divider: float | None = None,
        s_divider: float | None = None,
        r_divider: float | None = None,
        x_divider: float | None = None,
        to_plot: Literal["val_loss", "train_losses", "val_accs", "train_accs", "val_pplxs", "train_pplxs"] = "val_loss_causal",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "step",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y, and average y from a CSV file."""
    filters = (pl.col("last_val_loss").ge(0))  # initial condition -> always true

    if model_scale is not None:
        filters &= (pl.col("model_scale") == model_scale)
    if depth is not None:
        filters &= (pl.col("depth") == depth)
    if width is not None:
        filters &= (pl.col("width") == width)
    if num_params is not None:
        filters &= (pl.col("num_params") == num_params)
    if linear_value is not None:
        filters &= (pl.col("linear_value") == linear_value)
    if num_heads is not None:
        filters &= (pl.col("num_heads") == num_heads)
    if run_num is not None:
        filters &= (pl.col("run_num") == run_num)
    if seed is not None:
        filters &= (pl.col("seed") == seed)
    if ul2 is not None:
        filters &= (pl.col("ul2") == ul2)
    if causal_denoisers is not None:
        filters &= (pl.col("causal_denoisers") == causal_denoisers)
    if noncausal_masking is not None:
        filters &= (pl.col("noncausal_masking") == noncausal_masking)
    if randomize_denoiser_settings is not None:
        filters &= (pl.col("randomize_denoiser_settings") == randomize_denoiser_settings)
    if randomize_mask_width is not None:
        filters &= (pl.col("randomize_mask_width") == randomize_mask_width)
    if no_special_tokens is not None:
        filters &= (pl.col("no_special_tokens") == no_special_tokens)
    if alternate_denoisers is not None:
        filters &= (pl.col("alternate_denoisers") == alternate_denoisers)
    if causal_divider is not None:
        filters &= (pl.col("causal_divider") == causal_divider)
    if s_divider is not None:
        filters &= (pl.col("s_divider") == s_divider)
    if r_divider is not None:
        filters &= (pl.col("r_divider") == r_divider)
    if x_divider is not None:
        filters &= (pl.col("x_divider") == x_divider)

    df = pl.scan_csv(file).filter(filters).collect()
    df.sort("run_num")
    arrays = [series_to_array(df[to_plot][i]) for i in range(len(df[to_plot]))]

    if plot_over == "step":
        return load_steps_ys_avg_ys(df, arrays)
    elif plot_over == "epoch":
        return load_epochs_ys_avg_ys(df, arrays)
    elif plot_over == "token":
        return load_tokens_ys_avg_ys(df, arrays)
    elif plot_over == "time_sec":
        return load_time_ys_avg_ys(df, arrays)
    else:
        raise ValueError(f"{plot_over} not a valid x-value")


def load_steps_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_len = min([len(a) for a in arrays])
    ys = np.array([list(a[:min_len]) for a in arrays])
    num_datapoints = len(ys[0])
    xs = ((np.arange(num_datapoints) + 1) * 12.5).astype(int)
    avg_ys = np.mean(ys, axis=0)
    return xs, ys, avg_ys


def load_epochs_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [series_to_array(df["epoch"][i]) for i in range(len(df["epoch"]))]
    return interpolate_linearly(xs, arrays)


def load_tokens_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [series_to_array(df["tokens_seen"][i]) for i in range(len(df["tokens_seen"]))]
    return interpolate_linearly(xs, arrays)


def load_time_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [series_to_array(df["cumulative_time"][i]) for i in range(len(df["cumulative_time"]))]
    return interpolate_linearly(xs, arrays)


def interpolate_linearly(
        xs: list[np.ndarray], ys: list[np.ndarray], num_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Determine the maximum x value across all datasets
    max_x = max(x_vals.max() for x_vals in xs)
    
    # Generate a single set of new x values for all datasets
    new_x_vals = np.linspace(0, max_x, num_samples)

    new_ys = []
    for x_vals, y_vals in zip(xs, ys):
        # Interpolate y to the common set of new x values
        new_y_vals = np.interp(new_x_vals, x_vals, y_vals)
        new_ys.append(new_y_vals)

    # Convert new_ys to a 2D numpy array for easy manipulation
    new_ys = np.array(new_ys)
    
    # Calculate the average y values across all datasets
    avg_ys = np.nanmean(new_ys, axis=0)

    return new_x_vals, new_ys, avg_ys


def get_unique_settings(file: str, targets: list[str]) -> list[str | int | float | bool]:
    settings = []
    
    # Load the unique combinations of the targets
    combinations = (
        pl.scan_csv(file)
        .select(*[pl.col(target) for target in targets])
        .collect()
        .unique()
    )
    # Sort combinations alphabetically by content, target by target (for consistency in plotting)
    for target in targets:
        combinations = combinations.sort(target)
    # Create a list of settings
    for features in zip(
            *[combinations[target] for target in targets]
    ):
        settings.append(tuple(features))

    return settings


def generate_distinct_colors(n):
    """
    Generates n visually distinct colors.

    Parameters:
        n (int): The number of distinct colors to generate.

    Returns:
        list: A list of n visually distinct colors in hex format.
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Fixing saturation and lightness/value to 0.9 for bright colors
        # You can adjust these values for different color variations
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    
    return colors


def unique_num_params(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("num_params")
        .collect()
        ["num_params"]
        .unique()
        .to_numpy()
    )


def unique_widths(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("width")
        .collect()
        ["width"]
        .unique()
        .to_numpy()
    )


def unique_depths(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("depth")
        .collect()
        ["depth"]
        .unique()
        .to_numpy()
    )


def plot_metric_curves(
        file: str,
        depth: int | None = 8,
        width: int | None = 384,
        num_heads: int | None = None,
        linear_value: bool | None = False,
        ul2: bool | None = None,
        causal_denoisers: bool | None = None,
        noncausal_masking: bool | None = None,
        randomize_denoiser_settings: bool | None = None,
        randomize_mask_width: bool | None = None,
        no_special_tokens: bool | None = None,
        alternate_denoisers: bool | None = None,
        causal_divider: float | None = None,
        s_divider: float | None = None,
        r_divider: float | None = None,
        x_divider: float | None = None,
        to_plot: Literal["val_loss", "train_losses", "val_accs", "train_accs", "val_pplxs"] = "val_loss",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "epoch",
        show: bool = True,
        loglog: bool = False,
        plot_all: bool = False,
) -> None:
    
    settings = get_unique_settings(
        file,
        [
            "num_heads", "linear_value", "depth", "width",
            "ul2", "causal_denoisers", "noncausal_masking", "randomize_denoiser_settings",
            "randomize_mask_width", "causal_divider", "s_divider",
            "r_divider", "x_divider", "no_special_tokens", "alternate_denoisers",
        ],
    )

    for i, user_param in enumerate((
        num_heads, linear_value, depth, width, 
        ul2, causal_denoisers, noncausal_masking, randomize_denoiser_settings, 
        randomize_mask_width, causal_divider, s_divider, r_divider, x_divider,
        no_special_tokens, alternate_denoisers,
    )):
        if user_param is not None:
            settings = [setting for setting in settings if setting[i] == user_param or (i>4 and setting[4] is False)]

    colors = generate_distinct_colors(len(settings))

    for color, (
            num_heads_, linear_value_, depth_, width_,
            ul2_, causal_denoisers_, noncausal_masking_, randomize_denoiser_settings_, randomize_mask_width_,
            causal_divider_, s_divider_, r_divider_, x_divider_,
            no_special_tokens_, alternate_denoisers_,
    ) in zip(colors, settings):
        xs, ys, avg_ys = load_xs_ys_avg_y(
            file,
            depth=depth_,
            width=width_,
            num_heads=num_heads_,
            linear_value=linear_value_,
            ul2=ul2_,
            causal_denoisers=causal_denoisers_,
            noncausal_masking=noncausal_masking_,
            randomize_denoiser_settings=randomize_denoiser_settings_,
            randomize_mask_width=randomize_mask_width_,
            no_special_tokens=no_special_tokens_,
            alternate_denoisers=alternate_denoisers_,
            causal_divider=causal_divider_,
            s_divider=s_divider_,
            r_divider=r_divider_,
            x_divider=x_divider_,
            to_plot=to_plot,
            plot_over=plot_over,
        )
        if plot_all:
            for y in ys:
                if loglog:
                    plt.loglog(xs, y, color=color, alpha=0.2)
                else:
                    plt.plot(xs, y, color=color, alpha=0.2)

        num_params = pl.scan_csv(file).filter(
            (pl.col("num_heads") == num_heads_)
            & (pl.col("linear_value") == linear_value_)
            & (pl.col("depth") == depth_)
            & (pl.col("ul2") == ul2_)
            & (pl.col("causal_denoisers") == causal_denoisers_)
            & (pl.col("noncausal_masking") == noncausal_masking_)
            & (pl.col("randomize_denoiser_settings") == randomize_denoiser_settings_)
            & (pl.col("randomize_mask_width") == randomize_mask_width_)
            & (pl.col("causal_divider") == causal_divider_)
            & (pl.col("no_special_tokens") == no_special_tokens_)
            & (pl.col("alternate_denoisers") == alternate_denoisers_)
            & (pl.col("s_divider") == s_divider_)
            & (pl.col("r_divider") == r_divider_)
            & (pl.col("x_divider") == x_divider_)

        ).collect()["num_params"][0]
        
        if ul2_:
            label = f"UL2; C-S-R-X-div: {causal_divider_}-{s_divider_}-{r_divider_}-{x_divider_}"
            if not causal_denoisers_:
                label += ", nonC dens"
            if noncausal_masking_:
                label += ", nonC mask"
            if randomize_denoiser_settings_:
                label += ", rand den sets"
            if randomize_mask_width_:
                label += ", rand w"
            if no_special_tokens_:
                label += ", no tok"
            if alternate_denoisers_:
                label += ", alternate dens"
        else:
            label = "standard training"
        if loglog:
            plt.loglog(xs, avg_ys, color=color if plot_all else None, label=label)
        else:
            plt.plot(xs, avg_ys, color=color if plot_all else None, label=label)


    fig = plt.gcf()
    fig.set_size_inches(12, 7)

    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.legend()
    plt.grid()
    plt.title(f"{to_plot} vs {plot_over} ({format_num_params(num_params)}; depth: {depth_}, width: {width_})")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        # You should probably adjust the filename
        plt.savefig(f"{to_plot}_vs_{plot_over}.png", dpi=300)
    close_plt()  # in case you call this function multiple times with different settings


def count_mean_of_n_best_values(
        file: str,
        n: int,
        tablefmt: Literal["markdown", "latex", "cli"] = "cli",
        best: Literal["min", "max"] = "min",
        ul2: bool | None = None,
        depth: int | None = None,
        width: int | None = None,
        num_heads: int | None = None,
        linear_value: bool | None = False,
        causal_denoisers: bool | None = None,
        noncausal_masking: bool | None = None,
        randomize_denoiser_settings: bool | None = None,
        randomize_mask_width: bool | None = None,
        no_special_tokens: bool | None = None,
        alternate_denoisers: bool | None = None,
        causal_divider: float | None = None,
        s_divider: float | None = None,
        r_divider: float | None = None,
        x_divider: float | None = None,
        from_point: float | None = None,
        to_point: float | None = None,
        to_plot: Literal["val_loss", "train_losses", "val_accs", "train_accs", "val_pplxs"] = "val_loss_causal",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "epoch",
): 
    settings = get_unique_settings(
        file,
        [
            "num_heads", "linear_value", "depth", "width",
            "ul2", "causal_denoisers", "noncausal_masking", "randomize_denoiser_settings",
            "randomize_mask_width", "causal_divider", "s_divider",
            "r_divider", "x_divider", "no_special_tokens", "alternate_denoisers",
        ],
    )

    for i, user_param in enumerate((
        num_heads, linear_value, depth, width, 
        ul2, causal_denoisers, noncausal_masking, randomize_denoiser_settings, 
        randomize_mask_width, causal_divider, s_divider, r_divider, x_divider,
        no_special_tokens, alternate_denoisers,
    )):
        if user_param is not None:
            settings = [setting for setting in settings if setting[i] == user_param or (i>4 and setting[4] is False)]

    results = {
        "id": [], 
        "#curves": [], 
        "#params": [], 
        "depth": [], 
        "width": [],
        "ul2": [], 
        "causal dens": [], 
        "nonC mask": [],
        "rand setts": [],
        "rand w": [],
        "no toks": [],
        "alternate dens": [],
        "C div": [],
        "S div": [],
        "R div": [],
        "X div": [],
        f"mean (best {n})": [],
    }
    for i, (
        num_heads_, linear_value_, depth_, width_,
        ul2_, causal_denoisers_, noncausal_masking_, randomize_denoiser_settings_,
        randomize_mask_width_, causal_divider_, s_divider_,
        r_divider_, x_divider_,
        no_special_tokens_, alternate_denoisers_,
    ) in enumerate(settings):
        num_params = pl.scan_csv(file).filter(
            (pl.col("num_heads") == num_heads_)
            & (pl.col("linear_value") == linear_value_)
            & (pl.col("depth") == depth_)
            & (pl.col("ul2") == ul2_)
            & (pl.col("causal_denoisers") == causal_denoisers_)
            & (pl.col("noncausal_masking") == noncausal_masking_)
            & (pl.col("randomize_denoiser_settings") == randomize_denoiser_settings_)
            & (pl.col("randomize_mask_width") == randomize_mask_width_)
            & (pl.col("causal_divider") == causal_divider_)
            & (pl.col("no_special_tokens") == no_special_tokens_)
            & (pl.col("alternate_denoisers") == alternate_denoisers_)
            & (pl.col("s_divider") == s_divider_)
            & (pl.col("r_divider") == r_divider_)
            & (pl.col("x_divider") == x_divider_)
        ).collect()["num_params"][0]
        xs, ys, avg_ys = load_xs_ys_avg_y(
            file,
            num_heads=num_heads_,
            linear_value=linear_value_,
            depth=depth_,
            width=width_,
            ul2=ul2_,
            causal_denoisers=causal_denoisers_,
            noncausal_masking=noncausal_masking_,
            randomize_denoiser_settings=randomize_denoiser_settings_,
            randomize_mask_width=randomize_mask_width_,
            no_special_tokens=no_special_tokens_,
            alternate_denoisers=alternate_denoisers_,
            causal_divider=causal_divider_,
            s_divider=s_divider_,
            r_divider=r_divider_,
            x_divider=x_divider_,
            to_plot=to_plot,
            plot_over=plot_over,
        )

        # Filter the values to only include the ones in the specified range
        mask = np.ones_like(avg_ys, dtype=bool)
        if from_point is not None:
            mask = mask & (xs >= from_point)
        if to_point is not None:
            mask = mask & (xs <= to_point)
        avg_ys = avg_ys[mask]

        # Select the best values
        if best == "min":
            best_n_mean = np.mean(np.sort(avg_ys)[:n])
        else:
            best_n_mean = np.mean(np.flip(np.sort(avg_ys), axis=0)[:n])

        # Store the results
        results["id"].append(i)
        results["#curves"].append(len(ys))
        results["#params"].append(format_num_params(num_params))
        results["depth"].append(depth_)
        results["width"].append(width_)
        results["ul2"].append(ul2_)
        results["causal dens"].append(causal_denoisers_)
        results["nonC mask"].append(noncausal_masking_)
        results["rand setts"].append(randomize_denoiser_settings_)
        results["rand w"].append(randomize_mask_width_)
        results["no toks"].append(no_special_tokens_)
        results["alternate dens"].append(alternate_denoisers_)
        results["C div"].append(causal_divider_)
        results["S div"].append(s_divider_)
        results["R div"].append(r_divider_)
        results["X div"].append(x_divider_)
        results[f"mean (best {n})"].append(best_n_mean)

    colors = generate_distinct_colors(len(results["id"]))
    colors = [colors[i] for i in np.argsort(results[f"mean (best {n})"])]
    df = pl.DataFrame(results).sort(by=f"mean (best {n})", descending=best=="max")
    perc = pl.Series(r"% of best", [f"{num:,.1f}%" for num in df[f"mean (best {n})"]/df[f"mean (best {n})"][0]*100])
    df = df.with_columns(perc)
    data = df.to_numpy().tolist()
    headers = df.columns

    if tablefmt == "cli":
        def fmt(item) -> str:
            if not isinstance(item, float):
                return str(item)
            sig_digits = min(3, len(str(item).split(".")[1]))
            return f"{item:,.{sig_digits}f}"
        
        table = rich.table.Table(*headers, title=f"{to_plot} (from {plot_over} {from_point or xs[0].item():.1f} to {to_point or xs[-1].item():.1f})")
        for color, row in zip(colors, data):
            table.add_row(
                *[fmt(item) for item in row], 
                style=rich.style.Style(color=color),
            )
    else:
        data.insert(0, headers)
        table = tabulate(data, headers="firstrow", tablefmt="pipe" if tablefmt == "markdown" else tablefmt)

    print(table, "\n\n")


if __name__ == "__main__":
    results_seven = "results/results_seven.csv"
    # plot_metric_curves(
    #     file=results_seven,
    #     depth=35,
    #     width=None,
    #     num_heads=None,
    #     linear_value=False,
    #     ul2=None,
    #     causal_denoisers=True,
    #     randomize_denoiser_settings=True,
    #     randomize_mask_width= True,
    #     to_plot="val_loss_causal",
    #     plot_over="epoch",
    #     x_divider=1.0,
    #     show=True,
    #     loglog=False,
    #     plot_all=False,
    # )

    for to_plot in (
            "val_pplx_causal", "val_pplx_s", "val_pplx_r", "val_pplx_x",
            # "val_loss_causal", "val_loss_s", "val_loss_r", "val_loss_x",
    ):
        n = 5
        count_mean_of_n_best_values(
            file=results_seven,
            n=n,
            best="min",
            ul2=None,
            causal_denoisers=None,
            randomize_denoiser_settings=None,
            randomize_mask_width=None,
            alternate_denoisers=True,
            depth=35, # 21, 35, 43
            causal_divider=None,
            s_divider=None,
            r_divider=None,
            x_divider=None,
            from_point=None,
            to_point=None,
            to_plot=to_plot,
            plot_over="epoch",
            tablefmt="cli",
        )
