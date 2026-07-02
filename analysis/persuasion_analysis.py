#!/usr/bin/env python3
"""Comprehensive analysis pipeline for LLM persuasion study results."""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from results_io import FROZEN_CLEAN_PATH, load_dataset_rows
from results_schema import normalize_answer as schema_normalize_answer
from results_schema import parse_confidence as schema_parse_confidence

ANSWER_KEY_PATH = ROOT / "question.json"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

CONDITIONS = [
    "prior",
    "strength_alone",
    "majority_plain_alone",
    "majority_vague_alone",
    "credibility_alone",
    "strength_vs_majority",
    "majority_vs_credibility",
    "strength_vs_credibility",
]

PERSUASION_CONDITIONS = [c for c in CONDITIONS if c != "prior"]
COMPETITIVE_CONDITIONS = [
    "strength_vs_majority",
    "majority_vs_credibility",
    "strength_vs_credibility",
]

WINNER_SIDES = {
    "strength_vs_majority": ("strength", "majority"),
    "majority_vs_credibility": ("majority", "credibility"),
    "strength_vs_credibility": ("strength", "credibility"),
}

CONDITION_LABELS = {
    "prior": "Prior",
    "strength_alone": "Strength (alone)",
    "majority_plain_alone": "Majority plain (alone)",
    "majority_vague_alone": "Majority vague (alone)",
    "credibility_alone": "Credibility (alone)",
    "strength_vs_majority": "Strength vs majority",
    "majority_vs_credibility": "Majority vs credibility",
    "strength_vs_credibility": "Strength vs credibility",
}


def slugify_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def parse_confidence(value) -> float:
    score = schema_parse_confidence(value)
    return np.nan if score is None else score


def normalize_answer(value) -> str | None:
    return schema_normalize_answer(value)


def load_answer_key(path: Path = ANSWER_KEY_PATH) -> dict[str, str]:
    with path.open() as f:
        data = json.load(f)
    return {
        row["question_id"]: normalize_answer(row["correct_answer"])
        for row in data
    }


def load_results() -> pd.DataFrame:
    rows = load_dataset_rows(root=ROOT)
    if not rows:
        raise FileNotFoundError(
            f"No dataset found at {ROOT / FROZEN_CLEAN_PATH} and no rows under results/"
        )

    df = pd.DataFrame(rows)
    if "source_file" in df.columns:
        df = df.drop(columns=["source_file"])
    df = df.drop_duplicates(subset=["model", "question_id", "condition"], keep="last")
    df["confidence_num"] = df["confidence"].map(parse_confidence)
    df["prior_confidence_num"] = df["prior_confidence"].map(parse_confidence)
    df["answer_norm"] = df["answer"].map(normalize_answer)
    df["prior_answer_norm"] = df["prior_answer"].map(normalize_answer)
    return df


def enrich_with_correctness(df: pd.DataFrame, answer_key: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["correct_answer"] = out["question_id"].map(answer_key)
    out["is_correct"] = out["answer_norm"] == out["correct_answer"]
    out["prior_is_correct"] = out["prior_answer_norm"] == out["correct_answer"]
    out["confidence_change"] = out["confidence_num"] - out["prior_confidence_num"]
    out["confidence_increased"] = out["confidence_change"] > 0
    out["confidence_decreased"] = out["confidence_change"] < 0
    out["confidence_unchanged"] = out["confidence_change"] == 0

    # Transition labels for persuasion conditions
    def transition(row):
        if row["condition"] == "prior":
            return "baseline"
        pc, cc = row["prior_is_correct"], row["is_correct"]
        if pc and cc:
            return "C→C"
        if pc and not cc:
            return "C→W"
        if not pc and cc:
            return "W→C"
        return "W→W"

    out["transition"] = out.apply(transition, axis=1)
    return out


def holm_bonferroni(p_values: Iterable[float]) -> list[float]:
    pvals = list(p_values)
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (n - rank) * pvals[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return np.nan
    return diff.mean() / diff.std(ddof=1)


def cramers_v(table: np.ndarray) -> float:
    chi2, _, _, _ = stats.chi2_contingency(table)
    n = table.sum()
    if n == 0:
        return np.nan
    r, k = table.shape
    return math.sqrt(chi2 / (n * min(r - 1, k - 1)))


def wilson_ci_pct(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    from statsmodels.stats.proportion import proportion_confint

    low, high = proportion_confint(successes, total, alpha=alpha, method="wilson")
    return 100.0 * low, 100.0 * high


def categorical_barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    ax: plt.Axes,
    palette: str | list[str],
    **kwargs,
) -> None:
    """Seaborn barplot with hue=x to avoid future palette deprecation warnings."""
    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=x,
        palette=palette,
        dodge=False,
        legend=False,
        ax=ax,
        **kwargs,
    )


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return np.nan
    k = min(b, c)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * (0.5**n)
    return min(1.0, 2 * p)


def md_table(df: pd.DataFrame, float_fmt: str = ".2f") -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(
                lambda v: "" if pd.isna(v) else format(v, float_fmt)
            )
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


@dataclass
class AnalysisBundle:
    scope: str
    df: pd.DataFrame
    out_dir: Path


def ensure_dirs(path: Path) -> Path:
    fig_dir = path / "figures"
    path.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def flip_analysis(bundle: AnalysisBundle) -> pd.DataFrame:
    df = bundle.df
    rows = []
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        total = len(sub)
        flipped = int(sub["flipped"].sum())
        rate = 100.0 * flipped / total if total else np.nan
        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "n_questions": total,
                "n_flipped": flipped,
                "flip_rate_pct": rate,
            }
        )
    table = pd.DataFrame(rows).sort_values("flip_rate_pct", ascending=False)
    (bundle.out_dir / "01_flip_analysis.md").write_text(
        "# Flip Analysis\n\n" + md_table(table) + "\n"
    )
    return table


def confidence_analysis(bundle: AnalysisBundle) -> pd.DataFrame:
    df = bundle.df
    rows = []
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        conf = sub["confidence_num"].dropna()
        if condition == "prior":
            change = pd.Series(dtype=float)
            inc = dec = same = 0
        else:
            change = sub["confidence_change"].dropna()
            inc = int(sub["confidence_increased"].sum())
            dec = int(sub["confidence_decreased"].sum())
            same = int(sub["confidence_unchanged"].sum())
        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "mean_confidence": conf.mean(),
                "median_confidence": conf.median(),
                "std_confidence": conf.std(ddof=0),
                "mean_confidence_change": change.mean() if len(change) else np.nan,
                "n_increased": inc,
                "n_decreased": dec,
                "n_unchanged": same,
            }
        )
    table = pd.DataFrame(rows)
    (bundle.out_dir / "02_confidence_analysis.md").write_text(
        "# Confidence Analysis\n\n" + md_table(table, ".3f") + "\n"
    )

    fig_dir = bundle.out_dir / "figures"
    persuasion = table[table["condition"] != "prior"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    categorical_barplot(persuasion, "label", "mean_confidence", axes[0], palette="Blues_d")
    axes[0].set_title("Mean Confidence by Condition")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean confidence (1–10)")
    axes[0].tick_params(axis="x", rotation=35)
    for label in axes[0].get_xticklabels():
        label.set_ha("right")

    categorical_barplot(
        persuasion,
        "label",
        "mean_confidence_change",
        axes[1],
        palette="RdBu_r",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Mean Confidence Change from Prior")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Δ confidence")
    axes[1].tick_params(axis="x", rotation=35)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")

    fig.tight_layout()
    fig.savefig(fig_dir / "confidence_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = persuasion.melt(
        id_vars=["label"],
        value_vars=["n_increased", "n_decreased", "n_unchanged"],
        var_name="direction",
        value_name="count",
    )
    direction_labels = {
        "n_increased": "Increased",
        "n_decreased": "Decreased",
        "n_unchanged": "Unchanged",
    }
    plot_df["direction"] = plot_df["direction"].map(direction_labels)
    sns.barplot(data=plot_df, x="label", y="count", hue="direction", ax=ax)
    ax.set_title("Confidence Change Direction")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(fig_dir / "confidence_direction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return table


def winner_inference(comp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Binomial tests and Wilson CIs for competitive winner rates in this sample."""
    rows = []
    for condition in COMPETITIVE_CONDITIONS:
        sub = comp[comp["condition"] == condition]
        side_a, side_b = WINNER_SIDES[condition]
        a = int((sub["winner"] == side_a).sum())
        b = int((sub["winner"] == side_b).sum())
        n = a + b
        if n == 0:
            continue
        leading = side_a if a >= b else side_b
        leading_wins = max(a, b)
        ci_lo, ci_hi = wilson_ci_pct(leading_wins, n)
        p_value = float(stats.binomtest(leading_wins, n, 0.5, alternative="two-sided").pvalue)
        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "comparison": f"{side_a} vs {side_b}",
                "leading_strategy": leading,
                "leading_wins": leading_wins,
                "n_trials": n,
                "leading_win_pct": 100.0 * leading_wins / n,
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
                "binom_p_value": p_value,
            }
        )

    by_condition = pd.DataFrame(rows)
    strategy_counts = {
        strategy: int((comp["winner"] == strategy).sum())
        for strategy in ["strength", "majority", "credibility"]
    }
    total = int(sum(strategy_counts.values()))
    overall = {}
    if total > 0:
        observed = np.array(list(strategy_counts.values()))
        chi2, chi_p = stats.chisquare(observed)
        overall = {
            "n_trials": total,
            "strategy_counts": strategy_counts,
            "chi2": float(chi2),
            "chi2_p_value": float(chi_p),
        }
    return by_condition, overall


def winner_analysis(bundle: AnalysisBundle) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = bundle.df
    comp = df[df["condition"].isin(COMPETITIVE_CONDITIONS)].copy()
    comp = comp[comp["winner"].notna()]

    by_condition_rows = []
    for condition in COMPETITIVE_CONDITIONS:
        sub = comp[comp["condition"] == condition]
        total = len(sub)
        for strategy in ["strength", "majority", "credibility"]:
            wins = int((sub["winner"] == strategy).sum())
            by_condition_rows.append(
                {
                    "condition": condition,
                    "label": CONDITION_LABELS[condition],
                    "strategy": strategy,
                    "wins": wins,
                    "win_pct": 100.0 * wins / total if total else np.nan,
                }
            )

    by_condition = pd.DataFrame(by_condition_rows)

    overall_rows = []
    for strategy in ["strength", "majority", "credibility"]:
        wins = int((comp["winner"] == strategy).sum())
        total = len(comp)
        overall_rows.append(
            {
                "strategy": strategy,
                "wins": wins,
                "win_pct": 100.0 * wins / total if total else np.nan,
            }
        )
    overall = pd.DataFrame(overall_rows).sort_values("win_pct", ascending=False)
    overall["rank"] = range(1, len(overall) + 1)
    total_comp = len(comp)
    for idx, row in overall.iterrows():
        ci_lo, ci_hi = wilson_ci_pct(int(row["wins"]), total_comp)
        overall.loc[idx, "ci_95_lo"] = ci_lo
        overall.loc[idx, "ci_95_hi"] = ci_hi

    inference_by_condition, inference_overall = winner_inference(comp)

    text = "# Winner Analysis\n\n## By competitive condition\n\n"
    text += md_table(by_condition) + "\n\n## Overall strategy ranking\n\n"
    text += md_table(overall, ".2f") + "\n\n"
    text += "## Inferential tests (this sample)\n\n"
    text += (
        "Per condition: two-sided binomial test of whether the more frequent side differs from 50%; "
        "95% Wilson confidence intervals (CI) for the leading side's win rate.\n\n"
    )
    if len(inference_by_condition):
        text += md_table(inference_by_condition, ".4g") + "\n\n"
    if inference_overall:
        counts = inference_overall["strategy_counts"]
        text += (
            f"**Overall pooled distribution** (*n* = {inference_overall['n_trials']} competitive trials): "
            f"strength {counts['strength']}, majority {counts['majority']}, "
            f"credibility {counts['credibility']}. "
            f"Chi-square goodness-of-fit vs. uniform three-way split: "
            f"χ² = {inference_overall['chi2']:.3f}, *p* = {inference_overall['chi2_p_value']:.4g}. "
            "This tests equality of pooled win counts, not pairwise dominance in a single matchup.\n"
        )
    (bundle.out_dir / "03_winner_analysis.md").write_text(text)

    fig_dir = bundle.out_dir / "figures"
    fig, ax = plt.subplots(figsize=(8, 5))
    categorical_barplot(overall, "strategy", "win_pct", ax, palette="viridis")
    ax.set_title("Overall Persuasion Strategy Win Rate")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Win %")
    fig.tight_layout()
    fig.savefig(fig_dir / "winner_overall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=by_condition, x="label", y="win_pct", hue="strategy", ax=ax)
    ax.set_title("Win Rate by Competitive Condition")
    ax.set_xlabel("")
    ax.set_ylabel("Win %")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / "winner_by_condition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return by_condition, overall, inference_by_condition, inference_overall


def correctness_analysis(bundle: AnalysisBundle) -> pd.DataFrame:
    df = bundle.df
    rows = []
    transition_tables = {}
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        n = len(sub)
        acc = 100.0 * sub["is_correct"].mean() if n else np.nan
        if condition == "prior":
            acc_change = 0.0
        else:
            acc_change = 100.0 * (sub["is_correct"].mean() - sub["prior_is_correct"].mean())
        cc = int((sub["transition"] == "C→C").sum())
        cw = int((sub["transition"] == "C→W").sum())
        wc = int((sub["transition"] == "W→C").sum())
        ww = int((sub["transition"] == "W→W").sum())
        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "accuracy_pct": acc,
                "accuracy_change_from_prior_pct": acc_change,
                "C_to_C": cc,
                "C_to_W": cw,
                "W_to_C": wc,
                "W_to_W": ww,
            }
        )
        if condition != "prior":
            transition_tables[condition] = np.array([[cc, cw], [wc, ww]])

    table = pd.DataFrame(rows)
    text = "# Correctness Analysis\n\n" + md_table(table) + "\n\n## Transition matrices\n\n"
    for condition, mat in transition_tables.items():
        text += f"### {CONDITION_LABELS[condition]}\n\n"
        text += "|  | Post: Correct | Post: Wrong |\n"
        text += "|---|---:|---:|\n"
        text += f"| Prior: Correct | {mat[0,0]} | {mat[0,1]} |\n"
        text += f"| Prior: Wrong | {mat[1,0]} | {mat[1,1]} |\n\n"
    (bundle.out_dir / "04_correctness_analysis.md").write_text(text)

    fig_dir = bundle.out_dir / "figures"
    persuasion = table[table["condition"] != "prior"].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=persuasion, x="label", y="accuracy_pct", ax=ax, color="#4C72B0")
    ax.set_title("Accuracy by Condition")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_by_condition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    heatmap_df = persuasion.set_index("label")[["C_to_C", "C_to_W", "W_to_C", "W_to_W"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title("Answer Transition Counts")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(fig_dir / "transition_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for condition in PERSUASION_CONDITIONS:
        sub = df[df["condition"] == condition]
        mat = np.array([
            [int((sub["transition"] == "C→C").sum()), int((sub["transition"] == "C→W").sum())],
            [int((sub["transition"] == "W→C").sum()), int((sub["transition"] == "W→W").sum())],
        ])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(
            mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Post: Correct", "Post: Wrong"],
            yticklabels=["Prior: Correct", "Prior: Wrong"],
            ax=ax,
            cbar=False,
        )
        ax.set_title(CONDITION_LABELS[condition])
        fig.tight_layout()
        safe_name = condition.replace("_", "-")
        fig.savefig(fig_dir / f"confusion_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return table


def persuasion_benefit_analysis(bundle: AnalysisBundle) -> pd.DataFrame:
    df = bundle.df[bundle.df["condition"] != "prior"]
    rows = []
    for condition in PERSUASION_CONDITIONS:
        sub = df[df["condition"] == condition]
        harmful = int((sub["transition"] == "C→W").sum())
        helpful = int((sub["transition"] == "W→C").sum())
        n = len(sub)
        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "harmful_rate_pct": 100.0 * harmful / n if n else np.nan,
                "helpful_rate_pct": 100.0 * helpful / n if n else np.nan,
                "net_persuasion_gain_pct": 100.0 * (helpful - harmful) / n if n else np.nan,
                "harmful_n": harmful,
                "helpful_n": helpful,
            }
        )
    table = pd.DataFrame(rows).sort_values("net_persuasion_gain_pct", ascending=False)
    (bundle.out_dir / "05_persuasion_benefit.md").write_text(
        "# Persuasion Benefit Analysis\n\n" + md_table(table) + "\n"
    )
    return table


def calibration_analysis(bundle: AnalysisBundle) -> pd.DataFrame:
    df = bundle.df
    rows = []
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        correct_conf = sub.loc[sub["is_correct"], "confidence_num"].mean()
        incorrect_conf = sub.loc[~sub["is_correct"], "confidence_num"].mean()
        gap = correct_conf - incorrect_conf if pd.notna(correct_conf) and pd.notna(incorrect_conf) else np.nan
        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "mean_confidence_when_correct": correct_conf,
                "mean_confidence_when_incorrect": incorrect_conf,
                "calibration_gap": gap,
            }
        )
    table = pd.DataFrame(rows)

    prior_gap = table.loc[table["condition"] == "prior", "calibration_gap"].iloc[0]
    persuasion_gap = table.loc[table["condition"] != "prior", "calibration_gap"].mean()
    if pd.notna(prior_gap) and pd.notna(persuasion_gap):
        if persuasion_gap < prior_gap:
            overconfidence_note = "Persuasion conditions show a smaller calibration gap on average, indicating relatively higher confidence on incorrect answers (increased overconfidence)."
        elif persuasion_gap > prior_gap:
            overconfidence_note = "Persuasion conditions show a larger calibration gap on average, indicating relatively lower confidence on incorrect answers (reduced overconfidence / increased underconfidence on errors)."
        else:
            overconfidence_note = "Persuasion does not materially change the calibration gap relative to the prior baseline."
    else:
        overconfidence_note = "Insufficient data to assess overconfidence shift."

    text = "# Confidence Calibration\n\n" + md_table(table, ".3f") + "\n\n"
    text += f"**Interpretation:** {overconfidence_note}\n"
    (bundle.out_dir / "06_calibration.md").write_text(text)
    return table


def statistical_tests(bundle: AnalysisBundle) -> pd.DataFrame:
    df = bundle.df
    prior = df[df["condition"] == "prior"].set_index(["question_id", "model"], drop=False)

    rows = []
    for condition in PERSUASION_CONDITIONS:
        sub = df[df["condition"] == condition].set_index(["question_id", "model"], drop=False)
        common_idx = prior.index.intersection(sub.index)
        p_prior = prior.loc[common_idx, "is_correct"].astype(int)
        p_cond = sub.loc[common_idx, "is_correct"].astype(int)
        b = int(((p_prior == 1) & (p_cond == 0)).sum())
        c = int(((p_prior == 0) & (p_cond == 1)).sum())
        mcnemar_p = mcnemar_exact(b, c)

        conf_prior = prior.loc[common_idx, "confidence_num"].to_numpy()
        conf_cond = sub.loc[common_idx, "confidence_num"].to_numpy()
        mask = ~np.isnan(conf_prior) & ~np.isnan(conf_cond)
        if mask.sum() >= 5:
            wilcoxon = stats.wilcoxon(conf_cond[mask], conf_prior[mask])
            wilcoxon_p = wilcoxon.pvalue
            d = cohens_d_paired(conf_cond[mask], conf_prior[mask])
        else:
            wilcoxon_p = np.nan
            d = np.nan

        rows.append(
            {
                "test": f"McNemar: prior vs {condition}",
                "comparison": condition,
                "statistic": f"b={b}, c={c}",
                "p_value": mcnemar_p,
                "effect_size": np.nan,
                "effect_label": "n/a",
            }
        )
        rows.append(
            {
                "test": f"Wilcoxon: confidence prior vs {condition}",
                "comparison": condition,
                "statistic": "signed-rank",
                "p_value": wilcoxon_p,
                "effect_size": d,
                "effect_label": "Cohen's d",
            }
        )

    flip_table = []
    for condition in PERSUASION_CONDITIONS:
        sub = df[df["condition"] == condition]
        flip_table.append([int(sub["flipped"].sum()), int((~sub["flipped"]).sum())])
    flip_array = np.array(flip_table)
    fisher_rows = []
    if flip_array.sum() > 0 and flip_array.shape[0] > 1:
        if (flip_array < 5).any():
            chi_p = np.nan
            fisher_note = "Some flip-count cells are sparse (<5); pairwise Fisher's exact tests reported below."
            ref = flip_array[0]
            for i, condition in enumerate(PERSUASION_CONDITIONS):
                if i == 0:
                    continue
                _, p = stats.fisher_exact([ref, flip_array[i]])
                fisher_rows.append({
                    "test": f"Fisher: flip rate {PERSUASION_CONDITIONS[0]} vs {condition}",
                    "comparison": condition,
                    "statistic": "2x2 exact",
                    "p_value": p,
                    "effect_size": np.nan,
                    "effect_label": "n/a",
                })
        else:
            chi2, chi_p, _, _ = stats.chi2_contingency(flip_array)
            fisher_note = (
                f"Global chi-square on flip counts across conditions: "
                f"chi2={chi2:.3f}, p={chi_p:.4g}, Cramer's V={cramers_v(flip_array):.3f}"
            )
    else:
        chi_p = np.nan
        fisher_note = "Insufficient variation for global flip-rate chi-square."

    rows.extend(fisher_rows)

    table = pd.DataFrame(rows)
    table["p_holm"] = holm_bonferroni(table["p_value"].fillna(1.0).tolist())

    text = "# Statistical Tests\n\n" + md_table(table, ".4g") + "\n\n"
    text += f"**Flip-rate omnibus test:** {fisher_note}\n"
    (bundle.out_dir / "07_statistical_tests.md").write_text(text)
    return table


def model_comparison(all_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    models = sorted(all_df["model"].unique())
    rows = []
    for model in models:
        sub = all_df[all_df["model"] == model]
        persuasion = sub[sub["condition"] != "prior"]
        comp = sub[sub["condition"].isin(COMPETITIVE_CONDITIONS)]
        strength_wins = int((comp["winner"] == "strength").sum())
        majority_wins = int((comp["winner"] == "majority").sum())
        credibility_wins = int((comp["winner"] == "credibility").sum())
        harmful = int((persuasion["transition"] == "C→W").sum())
        helpful = int((persuasion["transition"] == "W→C").sum())
        rows.append(
            {
                "model": model,
                "flip_rate_pct": 100.0 * persuasion["flipped"].mean(),
                "accuracy_pct": 100.0 * sub["is_correct"].mean(),
                "prior_accuracy_pct": 100.0 * sub[sub["condition"] == "prior"]["is_correct"].mean(),
                "mean_confidence": persuasion["confidence_num"].mean(),
                "strength_win_pct": 100.0 * strength_wins / len(comp) if len(comp) else np.nan,
                "majority_win_pct": 100.0 * majority_wins / len(comp) if len(comp) else np.nan,
                "credibility_win_pct": 100.0 * credibility_wins / len(comp) if len(comp) else np.nan,
                "harmful_rate_pct": 100.0 * harmful / len(persuasion) if len(persuasion) else np.nan,
                "helpful_rate_pct": 100.0 * helpful / len(persuasion) if len(persuasion) else np.nan,
                "susceptibility_score": (
                    persuasion["flipped"].mean()
                    + (harmful - helpful) / max(len(persuasion), 1)
                ),
            }
        )

    table = pd.DataFrame(rows).sort_values("susceptibility_score", ascending=True)
    table["persuasion_resistance_rank"] = range(1, len(table) + 1)
    table["persuasion_susceptibility_rank"] = table["persuasion_resistance_rank"].iloc[::-1].values

    text = "# Model Comparison\n\n"
    text += "Ranked from **most resistant** (lowest susceptibility score) to **most susceptible**.\n\n"
    text += md_table(
        table[
            [
                "persuasion_resistance_rank",
                "model",
                "flip_rate_pct",
                "accuracy_pct",
                "mean_confidence",
                "strength_win_pct",
                "majority_win_pct",
                "credibility_win_pct",
                "susceptibility_score",
            ]
        ].rename(columns={"persuasion_resistance_rank": "rank_resistant_to_susceptible"})
    )
    text += "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "08_model_comparison.md").write_text(text)

    if len(models) > 1:
        flip_counts = []
        for model in models:
            sub = all_df[(all_df["model"] == model) & (all_df["condition"] != "prior")]
            flip_counts.append([int(sub["flipped"].sum()), int((~sub["flipped"]).sum())])
        arr = np.array(flip_counts)
        if (arr < 5).any():
            test_line = "Pairwise Fisher's exact tests recommended due to sparse flip counts."
        else:
            chi2, p, _, _ = stats.chi2_contingency(arr)
            test_line = f"Chi-square across models (flip vs no-flip): chi2={chi2:.3f}, p={p:.4g}, Cramer's V={cramers_v(arr):.3f}"
        (out_dir / "08_model_comparison_tests.md").write_text(
            f"# Model Comparison Statistical Tests\n\n{test_line}\n"
        )

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sns.barplot(data=table, x="model", y="flip_rate_pct", ax=axes[0, 0], color="#C44E52")
    axes[0, 0].set_title("Flip Rate by Model")
    axes[0, 0].tick_params(axis="x", rotation=25)
    sns.barplot(data=table, x="model", y="accuracy_pct", ax=axes[0, 1], color="#4C72B0")
    axes[0, 1].set_title("Overall Accuracy by Model")
    axes[0, 1].tick_params(axis="x", rotation=25)
    sns.barplot(data=table, x="model", y="mean_confidence", ax=axes[1, 0], color="#55A868")
    axes[1, 0].set_title("Mean Confidence by Model")
    axes[1, 0].tick_params(axis="x", rotation=25)
    win_df = table.melt(
        id_vars=["model"],
        value_vars=["strength_win_pct", "majority_win_pct", "credibility_win_pct"],
        var_name="strategy",
        value_name="win_pct",
    )
    win_df["strategy"] = win_df["strategy"].str.replace("_win_pct", "")
    sns.barplot(data=win_df, x="model", y="win_pct", hue="strategy", ax=axes[1, 1])
    axes[1, 1].set_title("Winner Preferences by Model")
    axes[1, 1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return table


def publication_figures(bundle: AnalysisBundle, flip_table: pd.DataFrame, correctness_table: pd.DataFrame, benefit_table: pd.DataFrame) -> None:
    fig_dir = bundle.out_dir / "figures"
    persuasion_flip = flip_table[flip_table["condition"] != "prior"].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    categorical_barplot(persuasion_flip, "label", "flip_rate_pct", ax, palette="rocket")
    ax.set_title("Flip Rate by Persuasion Condition")
    ax.set_xlabel("")
    ax.set_ylabel("Flip rate (%)")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(fig_dir / "flip_rate_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    persuasion_acc = correctness_table[correctness_table["condition"] != "prior"].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(persuasion_acc))
    width = 0.35
    ax.bar(x - width / 2, persuasion_acc["accuracy_pct"], width, label="Post-persuasion", color="#4C72B0")
    prior_acc = persuasion_acc["accuracy_pct"] - persuasion_acc["accuracy_change_from_prior_pct"]
    ax.bar(x + width / 2, prior_acc, width, label="Prior", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(persuasion_acc["label"], rotation=35, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy: Prior vs Persuasion Conditions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    categorical_barplot(benefit_table, "label", "net_persuasion_gain_pct", ax, palette="vlag")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Net Persuasion Gain by Condition")
    ax.set_xlabel("")
    ax.set_ylabel("Net gain (helpful − harmful, %)")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(fig_dir / "net_persuasion_gain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def research_questions(
    flip_table: pd.DataFrame,
    winner_overall: pd.DataFrame,
    correctness_table: pd.DataFrame,
    benefit_table: pd.DataFrame,
    confidence_table: pd.DataFrame,
    model_table: pd.DataFrame,
    stats_table: pd.DataFrame,
    winner_inference_overall: dict,
    winner_inference_by_condition: pd.DataFrame,
    out_path: Path,
) -> str:
    persuasion_flip = flip_table[flip_table["condition"] != "prior"].sort_values("flip_rate_pct", ascending=False)
    top_flip = persuasion_flip.iloc[0]
    bottom_flip = persuasion_flip.iloc[-1]
    top_flip_ci = wilson_ci_pct(int(top_flip["n_flipped"]), int(top_flip["n_questions"]))

    top_strategy = winner_overall.iloc[0]
    top_strategy_ci = (
        float(top_strategy["ci_95_lo"]),
        float(top_strategy["ci_95_hi"]),
    )

    persuasion_acc = correctness_table[correctness_table["condition"] != "prior"]
    mean_acc_change = persuasion_acc["accuracy_change_from_prior_pct"].mean()
    net_benefit = benefit_table["net_persuasion_gain_pct"].mean()

    conf_change_mean = confidence_table[confidence_table["condition"] != "prior"]["mean_confidence_change"].mean()
    mean_flip = persuasion_flip["flip_rate_pct"].mean()

    maj_plain = flip_table[flip_table["condition"] == "majority_plain_alone"]["flip_rate_pct"].iloc[0]
    cred_alone = flip_table[flip_table["condition"] == "credibility_alone"]["flip_rate_pct"].iloc[0]
    strength_alone = flip_table[flip_table["condition"] == "strength_alone"]["flip_rate_pct"].iloc[0]
    maj_vague = flip_table[flip_table["condition"] == "majority_vague_alone"]["flip_rate_pct"].iloc[0]

    most_resistant = model_table.iloc[0]
    most_susceptible = model_table.iloc[-1]

    sig_mcnemar = stats_table[
        stats_table["test"].str.startswith("McNemar") & (stats_table["p_holm"] < 0.05)
    ]

    text = "# Research Questions\n\n"
    text += (
        f"**RQ1 — Which persuasion strategy changes model beliefs most frequently?** "
        f"In this sample, {CONDITION_LABELS[top_flip['condition']]} had the highest flip rate "
        f"({top_flip['flip_rate_pct']:.1f}%, 95% CI [{top_flip_ci[0]:.1f}, {top_flip_ci[1]:.1f}]%), "
        f"compared with {CONDITION_LABELS[bottom_flip['condition']]} "
        f"({bottom_flip['flip_rate_pct']:.1f}%). See flip-rate omnibus test in statistical tests.\n\n"
    )

    rq2 = (
        f"**RQ2 — When persuasion strategies conflict, which strategy is selected most often?** "
        f"In this sample, **{top_strategy['strategy']}** had the highest pooled win rate "
        f"({top_strategy['win_pct']:.1f}%, 95% CI [{top_strategy_ci[0]:.1f}, {top_strategy_ci[1]:.1f}]% "
        f"across all competitive trials)."
    )
    if winner_inference_overall:
        rq2 += (
            f" A chi-square test against a uniform three-way split gave "
            f"χ² = {winner_inference_overall['chi2']:.2f}, "
            f"*p* = {winner_inference_overall['chi2_p_value']:.4g} "
            f"(pooled *n* = {winner_inference_overall['n_trials']}); this does not establish pairwise dominance "
            f"in any single matchup."
        )
    rq2 += "\n\n"
    text += rq2

    text += (
        f"**RQ3 — Does persuasion improve or reduce factual accuracy?** "
        f"In this sample, mean accuracy change from prior across persuasion conditions was "
        f"{mean_acc_change:+.2f} percentage points; mean net persuasion gain was "
        f"{net_benefit:+.2f} percentage points (helpful − harmful). "
        f"Condition-level significance is reported via Holm-corrected McNemar tests.\n\n"
    )
    text += (
        f"**RQ4 — Does persuasion mainly change beliefs or simply reduce confidence?** "
        f"In this sample, mean flip rate was {mean_flip:.1f}% while mean confidence change was "
        f"{conf_change_mean:+.2f} points, suggesting persuasion "
        f"{'primarily shifted answers' if abs(mean_flip) > abs(conf_change_mean) * 10 else 'affected both answers and confidence materially'} "
        f"in this benchmark.\n\n"
    )
    text += (
        f"**RQ5 — Are some models more resistant to persuasion than others?** "
        f"In this sample, the lowest flip rate was {most_resistant['model']} "
        f"({most_resistant['flip_rate_pct']:.1f}%) and the highest was {most_susceptible['model']} "
        f"({most_susceptible['flip_rate_pct']:.1f}%). "
    )
    if len(model_table) > 1:
        text += "See model-comparison tests for omnibus significance.\n\n"
    else:
        text += "Only one model is present in the dataset; cross-model significance cannot be assessed.\n\n"

    maj_cred_row = winner_inference_by_condition[
        winner_inference_by_condition["condition"] == "majority_vs_credibility"
    ]
    str_maj_row = winner_inference_by_condition[
        winner_inference_by_condition["condition"] == "strength_vs_majority"
    ]

    rq6 = (
        f"**RQ6 — Does credibility outperform majority?** "
        f"In this sample, credibility-alone flip rate was {cred_alone:.1f}% vs majority-plain "
        f"{maj_plain:.1f}% and majority-vague {maj_vague:.1f}%."
    )
    if len(maj_cred_row):
        row = maj_cred_row.iloc[0]
        rq6 += (
            f" In majority-vs-credibility matchups, {row['leading_strategy']} led "
            f"({row['leading_win_pct']:.1f}%, 95% CI [{row['ci_95_lo']:.1f}, {row['ci_95_hi']:.1f}]%; "
            f"binomial *p* = {row['binom_p_value']:.4g})."
        )
    rq6 += "\n\n"
    text += rq6

    rq7 = (
        f"**RQ7 — Does strong evidence consistently outperform majority?** "
        f"In this sample, strength-alone flip rate was {strength_alone:.1f}% vs majority-plain "
        f"{maj_plain:.1f}% and majority-vague {maj_vague:.1f}%."
    )
    if len(str_maj_row):
        row = str_maj_row.iloc[0]
        rq7 += (
            f" In strength-vs-majority matchups, {row['leading_strategy']} led "
            f"({row['leading_win_pct']:.1f}%, 95% CI [{row['ci_95_lo']:.1f}, {row['ci_95_hi']:.1f}]%; "
            f"binomial *p* = {row['binom_p_value']:.4g})."
        )
    rq7 += "\n\n"
    text += rq7

    if len(sig_mcnemar):
        text += "## Statistically significant accuracy shifts (Holm-corrected McNemar)\n\n"
        for _, row in sig_mcnemar.iterrows():
            text += f"- {row['comparison']}: p={row['p_holm']:.4g}\n"

    out_path.write_text(text)
    return text


def write_results_section(
    scope: str,
    flip_table: pd.DataFrame,
    confidence_table: pd.DataFrame,
    winner_overall: pd.DataFrame,
    correctness_table: pd.DataFrame,
    benefit_table: pd.DataFrame,
    calibration_table: pd.DataFrame,
    stats_table: pd.DataFrame,
    model_table: pd.DataFrame,
    winner_inference_overall: dict,
    winner_inference_by_condition: pd.DataFrame,
    out_path: Path,
) -> None:
    persuasion_flip = flip_table[flip_table["condition"] != "prior"].sort_values("flip_rate_pct", ascending=False)
    top3_parts = []
    for _, row in persuasion_flip.head(3).iterrows():
        ci_lo, ci_hi = wilson_ci_pct(int(row["n_flipped"]), int(row["n_questions"]))
        top3_parts.append(
            f"{row['label']} ({row['flip_rate_pct']:.1f}%, 95% CI [{ci_lo:.1f}, {ci_hi:.1f}]%)"
        )
    top3_flip = ", ".join(top3_parts)
    top_strategy = winner_overall.iloc[0]
    top_ci = (float(top_strategy["ci_95_lo"]), float(top_strategy["ci_95_hi"]))
    persuasion_acc = correctness_table[correctness_table["condition"] != "prior"]
    mean_acc_change = persuasion_acc["accuracy_change_from_prior_pct"].mean()
    best_benefit = benefit_table.iloc[0]
    worst_benefit = benefit_table.iloc[-1]
    if worst_benefit["net_persuasion_gain_pct"] < 0:
        harm_phrase = (
            f"while the largest net harm was associated with {worst_benefit['label']} "
            f"({worst_benefit['net_persuasion_gain_pct']:+.1f}%)"
        )
    else:
        harm_phrase = (
            f"while the smallest net gain was observed for {worst_benefit['label']} "
            f"({worst_benefit['net_persuasion_gain_pct']:+.1f}%)"
        )

    prior_cal = calibration_table[calibration_table["condition"] == "prior"]["calibration_gap"].iloc[0]
    mean_persuasion_cal = calibration_table[calibration_table["condition"] != "prior"]["calibration_gap"].mean()

    sig_tests = stats_table[(stats_table["p_holm"] < 0.05) & stats_table["p_value"].notna()]
    sig_summary = (
        f"{len(sig_tests)} Holm-corrected tests reached p<0.05"
        if len(sig_tests)
        else "No Holm-corrected tests reached p<0.05"
    )

    sections = [
        f"# Results ({scope})\n",
        "## Belief updating under persuasion",
        (
            f"In this sample, we measured belief change as the proportion of trials in which the model's "
            f"yes/no answer differed from its prior response. Across persuasion conditions, flip rates "
            f"ranged from {persuasion_flip['flip_rate_pct'].min():.1f}% to "
            f"{persuasion_flip['flip_rate_pct'].max():.1f}%. "
            f"The highest flip rates occurred for {top3_flip}."
        ),
        "## Competitive strategy selection (head-to-head trials)",
        (
            f"In this sample, **{top_strategy['strategy']}** was the most frequently selected strategy "
            f"in pooled competitive trials ({top_strategy['win_pct']:.1f}%, 95% CI "
            f"[{top_ci[0]:.1f}, {top_ci[1]:.1f}]%)."
            + (
                f" A chi-square test against a uniform three-way split gave χ² = "
                f"{winner_inference_overall['chi2']:.2f}, *p* = {winner_inference_overall['chi2_p_value']:.4g}."
                if winner_inference_overall
                else ""
            )
            + " Per-matchup binomial tests and CIs are in the winner analysis table."
        ),
        "## Effects on factual accuracy",
        (
            f"In this sample, persuasion conditions changed mean accuracy by "
            f"{mean_acc_change:+.2f} percentage points relative to the prior baseline. "
            f"The largest net factual benefit was observed for "
            f"{best_benefit['label']} (net gain {best_benefit['net_persuasion_gain_pct']:+.1f}%), "
            f"{harm_phrase}."
        ),
        "## Confidence and calibration",
        (
            f"Mean confidence shifted by "
            f"{confidence_table[confidence_table['condition'] != 'prior']['mean_confidence_change'].mean():+.2f} "
            f"points on average. The prior calibration gap (confidence when correct minus confidence when "
            f"incorrect) was {prior_cal:.2f}; under persuasion conditions the mean gap was "
            f"{mean_persuasion_cal:.2f}, indicating "
            f"{'increased overconfidence on errors' if mean_persuasion_cal < prior_cal else 'reduced overconfidence on errors'} "
            f"relative to baseline."
        ),
        "## Model differences",
        (
            f"Models varied in susceptibility from {model_table.iloc[0]['model']} "
            f"(flip rate {model_table.iloc[0]['flip_rate_pct']:.1f}%, rank 1 most resistant) "
            f"to {model_table.iloc[-1]['model']} "
            f"(flip rate {model_table.iloc[-1]['flip_rate_pct']:.1f}%, most susceptible)."
            if len(model_table) > 1
            else (
                f"Analysis included {model_table.iloc[0]['model']} "
                f"(flip rate {model_table.iloc[0]['flip_rate_pct']:.1f}%)."
            )
        ),
        "## Statistical significance",
        (
            f"Paired McNemar tests compared each persuasion condition's accuracy against the prior; "
            f"Wilcoxon signed-rank tests assessed paired confidence shifts. After Holm–Bonferroni correction, "
            f"{sig_summary}."
        ),
        "## Summary",
        (
            f"In this sample, persuasion altered model beliefs (mean flip rate "
            f"{persuasion_flip['flip_rate_pct'].mean():.1f}%) with heterogeneous effects on accuracy "
            f"and confidence. Competitive selection patterns and susceptibility differences are "
            f"reported with uncertainty intervals in the accompanying tables."
        ),
    ]

    out_path.write_text("\n\n".join(sections) + "\n")


def run_scope(scope: str, df: pd.DataFrame, out_dir: Path) -> dict:
    ensure_dirs(out_dir)
    bundle = AnalysisBundle(scope=scope, df=df, out_dir=out_dir)

    flip_table = flip_analysis(bundle)
    confidence_table = confidence_analysis(bundle)
    _, winner_overall, winner_inference_by_condition, winner_inference_overall = winner_analysis(bundle)
    correctness_table = correctness_analysis(bundle)
    benefit_table = persuasion_benefit_analysis(bundle)
    calibration_table = calibration_analysis(bundle)
    stats_table = statistical_tests(bundle)
    publication_figures(bundle, flip_table, correctness_table, benefit_table)

    return {
        "flip_table": flip_table,
        "confidence_table": confidence_table,
        "winner_overall": winner_overall,
        "winner_inference_by_condition": winner_inference_by_condition,
        "winner_inference_overall": winner_inference_overall,
        "correctness_table": correctness_table,
        "benefit_table": benefit_table,
        "calibration_table": calibration_table,
        "stats_table": stats_table,
    }


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    answer_key = load_answer_key()
    all_df = enrich_with_correctness(load_results(), answer_key)

    aggregated_dir = OUTPUT_ROOT / "aggregated"
    agg = run_scope("All models (aggregated)", all_df, aggregated_dir)

    per_model = {}
    for model in sorted(all_df["model"].unique()):
        model_df = all_df[all_df["model"] == model]
        model_dir = OUTPUT_ROOT / "models" / slugify_model(model)
        per_model[model] = run_scope(model, model_df, model_dir)

    model_table = model_comparison(all_df, OUTPUT_ROOT / "model_comparison")

    research_questions(
        agg["flip_table"],
        agg["winner_overall"],
        agg["correctness_table"],
        agg["benefit_table"],
        agg["confidence_table"],
        model_table,
        agg["stats_table"],
        agg["winner_inference_overall"],
        agg["winner_inference_by_condition"],
        OUTPUT_ROOT / "09_research_questions.md",
    )

    write_results_section(
        "All Models",
        agg["flip_table"],
        agg["confidence_table"],
        agg["winner_overall"],
        agg["correctness_table"],
        agg["benefit_table"],
        agg["calibration_table"],
        agg["stats_table"],
        model_table,
        agg["winner_inference_overall"],
        agg["winner_inference_by_condition"],
        OUTPUT_ROOT / "RESULTS.md",
    )

    for model, results in per_model.items():
        write_results_section(
            model,
            results["flip_table"],
            results["confidence_table"],
            results["winner_overall"],
            results["correctness_table"],
            results["benefit_table"],
            results["calibration_table"],
            results["stats_table"],
            model_table[model_table["model"] == model],
            results["winner_inference_overall"],
            results["winner_inference_by_condition"],
            OUTPUT_ROOT / "models" / slugify_model(model) / "RESULTS.md",
        )

    index = [
        "# Persuasion Study Analysis Output",
        "",
        f"- Records analyzed: **{len(all_df)}**",
        f"- Models: **{', '.join(sorted(all_df['model'].unique()))}**",
        f"- Questions: **{all_df['question_id'].nunique()}**",
        "",
        "## Aggregated",
        "- [Flip analysis](aggregated/01_flip_analysis.md)",
        "- [Confidence analysis](aggregated/02_confidence_analysis.md)",
        "- [Winner analysis](aggregated/03_winner_analysis.md)",
        "- [Correctness analysis](aggregated/04_correctness_analysis.md)",
        "- [Persuasion benefit](aggregated/05_persuasion_benefit.md)",
        "- [Calibration](aggregated/06_calibration.md)",
        "- [Statistical tests](aggregated/07_statistical_tests.md)",
        "- [Figures](aggregated/figures/)",
        "- [Paper Results](RESULTS.md)",
        "",
        "## Per-model",
    ]
    for model in sorted(all_df["model"].unique()):
        slug = slugify_model(model)
        index.append(f"- **{model}**: [outputs](models/{slug}/)")
    index.extend([
        "",
        "## Cross-model",
        "- [Model comparison](model_comparison/08_model_comparison.md)",
        "- [Research questions](09_research_questions.md)",
    ])
    (OUTPUT_ROOT / "README.md").write_text("\n".join(index) + "\n")

    print(f"Analysis complete. Outputs written to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
