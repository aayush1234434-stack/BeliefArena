#!/usr/bin/env python3
"""
Analyze belief-update experiment results from JSONL files.

Loads all ``*.jsonl`` files under ``results/``, computes descriptive statistics,
runs statistical tests, and writes tables, figures, and a paper-ready summary
to ``results/analysis/``.

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir results --questions question.json
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
ANALYSIS_DIR = RESULTS_DIR / "analysis"
QUESTIONS_FILE = Path("question.json")
ALL_MODELS_LABEL = "ALL_MODELS"

CONDITIONS = [
    "prior",
    "strength_alone",
    "majority_vague_alone",
    "strength_vs_majority",
    "majority_vs_credibility",
    "strength_vs_credibility",
    "credibility_alone",
    "majority_plain_alone",
]

NON_PRIOR_CONDITIONS = [c for c in CONDITIONS if c != "prior"]

CONFLICT_CONDITIONS = [
    "strength_vs_majority",
    "majority_vs_credibility",
    "strength_vs_credibility",
]

WINNER_SIDES = {
    "strength_vs_majority": ("strength", "majority"),
    "majority_vs_credibility": ("majority", "credibility"),
    "strength_vs_credibility": ("strength", "credibility"),
}

TRANSITION_LABELS = ["correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong"]
TRANSITION_SHORT = ["CC", "CW", "WC", "WW"]

ALPHA = 0.05

# Publication-style defaults
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (8, 5),
    }
)
sns.set_theme(style="whitegrid", context="paper")


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------


def load_jsonl_files(results_dir: Path) -> pd.DataFrame:
    """Load and concatenate every JSONL file in ``results_dir``."""
    paths = sorted(results_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No JSONL files found in {results_dir}. "
            "Place per-model result files there before running analysis."
        )

    frames = []
    for path in paths:
        rows = []
        with path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
        if rows:
            frame = pd.DataFrame(rows)
            frame["source_file"] = path.name
            frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["answer"] = df["answer"].astype(str).str.strip()
    df["prior_answer"] = df["prior_answer"].astype(str).str.strip()
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITIONS, ordered=True)
    return df


def load_gold_answers(questions_file: Path) -> pd.DataFrame:
    """Return question_id -> correct_answer mapping from ``question.json``."""
    with questions_file.open() as handle:
        questions = json.load(handle)
    gold = pd.DataFrame(questions)[["question_id", "correct_answer"]]
    gold["correct_answer"] = gold["correct_answer"].str.strip().str.lower()
    return gold


def parse_confidence(value) -> float | np.nan:
    """Parse model confidence strings (e.g. ``'8'``, ``'2.'``) to floats."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    number = float(match.group(1))
    return number if 1 <= number <= 10 else np.nan


def prepare_data(df: pd.DataFrame, gold: pd.DataFrame | None) -> pd.DataFrame:
    """Add numeric confidence, correctness, and paired prior fields."""
    out = df.copy()
    out["confidence_num"] = out["confidence"].map(parse_confidence)
    out["prior_confidence_num"] = out["prior_confidence"].map(parse_confidence)
    out["confidence_change"] = out["confidence_num"] - out["prior_confidence_num"]

    if gold is not None and not gold.empty:
        out = out.merge(gold, on="question_id", how="left")
        out["answer_norm"] = out["answer"].str.lower()
        out["prior_answer_norm"] = out["prior_answer"].str.lower()
        out["is_correct"] = out["answer_norm"] == out["correct_answer"]
        out["prior_is_correct"] = out["prior_answer_norm"] == out["correct_answer"]
        out["accuracy_change"] = out["is_correct"].astype(int) - out["prior_is_correct"].astype(int)
        out["transition"] = pd.Series(
            np.select(
                [
                    out["prior_is_correct"] & out["is_correct"],
                    out["prior_is_correct"] & ~out["is_correct"],
                    ~out["prior_is_correct"] & out["is_correct"],
                    ~out["prior_is_correct"] & ~out["is_correct"],
                ],
                TRANSITION_LABELS,
                default="",
            ),
            index=out.index,
            dtype="string",
        )
        out.loc[out["transition"] == "", "transition"] = pd.NA
    else:
        out["is_correct"] = np.nan
        out["prior_is_correct"] = np.nan
        out["accuracy_change"] = np.nan
        out["transition"] = np.nan

    return out


def model_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Yield (label, subset) for each model plus an all-models aggregate."""
    groups = [(model, df[df["model"] == model]) for model in sorted(df["model"].unique())]
    groups.append((ALL_MODELS_LABEL, df))
    return groups


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


def compute_flip_rates(subset: pd.DataFrame) -> pd.DataFrame:
    """Flip rate (%) per condition."""
    rows = []
    for condition in CONDITIONS:
        cond_df = subset[subset["condition"] == condition]
        n = len(cond_df)
        if n == 0:
            continue
        flip_pct = 100.0 * cond_df["flipped"].mean() if condition != "prior" else 0.0
        rows.append(
            {
                "condition": condition,
                "n": n,
                "flips": int(cond_df["flipped"].sum()) if condition != "prior" else 0,
                "flip_rate_pct": flip_pct,
            }
        )
    return pd.DataFrame(rows)


def compute_winner_stats(subset: pd.DataFrame) -> pd.DataFrame:
    """Winner counts and percentages for conflict conditions."""
    rows = []
    for condition in CONFLICT_CONDITIONS:
        cond_df = subset[(subset["condition"] == condition) & subset["winner"].notna()]
        n = len(cond_df)
        if n == 0:
            continue
        for winner, count in cond_df["winner"].value_counts().items():
            rows.append(
                {
                    "condition": condition,
                    "winner": winner,
                    "count": int(count),
                    "pct": 100.0 * count / n,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def compute_confidence_stats(subset: pd.DataFrame) -> pd.DataFrame:
    """Mean/median confidence and mean change vs. prior per condition."""
    rows = []
    for condition in CONDITIONS:
        cond_df = subset[subset["condition"] == condition]
        conf = cond_df["confidence_num"].dropna()
        change = cond_df["confidence_change"].dropna()
        if conf.empty:
            continue
        rows.append(
            {
                "condition": condition,
                "n": len(cond_df),
                "n_confidence": len(conf),
                "mean_confidence": conf.mean(),
                "median_confidence": conf.median(),
                "mean_confidence_change": change.mean() if condition != "prior" else 0.0,
                "median_confidence_change": change.median() if condition != "prior" else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_accuracy_stats(subset: pd.DataFrame) -> pd.DataFrame | None:
    """Accuracy and accuracy change relative to prior, when gold labels exist."""
    if subset["is_correct"].isna().all():
        return None

    rows = []
    for condition in CONDITIONS:
        cond_df = subset[subset["condition"] == condition].dropna(subset=["is_correct"])
        if cond_df.empty:
            continue
        rows.append(
            {
                "condition": condition,
                "n": len(cond_df),
                "accuracy_pct": 100.0 * cond_df["is_correct"].mean(),
                "mean_accuracy_change": cond_df["accuracy_change"].mean(),
            }
        )
    return pd.DataFrame(rows)


def compute_transition_stats(subset: pd.DataFrame) -> pd.DataFrame | None:
    """Correct/Wrong transition counts from prior to each condition."""
    if subset["transition"].isna().all():
        return None

    rows = []
    for condition in NON_PRIOR_CONDITIONS:
        cond_df = subset[subset["condition"] == condition].dropna(subset=["transition"])
        if cond_df.empty:
            continue
        counts = cond_df["transition"].value_counts()
        row = {"condition": condition, "n": len(cond_df)}
        for label in TRANSITION_LABELS:
            row[label] = int(counts.get(label, 0))
            row[f"{label}_pct"] = 100.0 * row[label] / len(cond_df)
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all descriptive tables keyed by ``{model}_{table}``."""
    tables: dict[str, pd.DataFrame] = {}
    for model_label, subset in model_groups(df):
        prefix = _safe_name(model_label)
        tables[f"{prefix}_flip_rates"] = compute_flip_rates(subset).assign(model=model_label)
        tables[f"{prefix}_winner_stats"] = compute_winner_stats(subset).assign(model=model_label)
        tables[f"{prefix}_confidence_stats"] = compute_confidence_stats(subset).assign(model=model_label)

        accuracy = compute_accuracy_stats(subset)
        if accuracy is not None:
            tables[f"{prefix}_accuracy_stats"] = accuracy.assign(model=model_label)

        transitions = compute_transition_stats(subset)
        if transitions is not None:
            tables[f"{prefix}_transitions"] = transitions.assign(model=model_label)

    return tables


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def _cohens_d_paired(diff: np.ndarray) -> float:
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return np.nan
    return diff.mean() / diff.std(ddof=1)


def _cramers_v(chi2: float, n: int, k: int) -> float:
    if n == 0 or k <= 1:
        return np.nan
    return np.sqrt(chi2 / (n * (k - 1)))


def _odds_ratio_mcnemar(b: int, c: int) -> float:
    """Odds ratio for discordant McNemar pairs (b=CW, c=WC style)."""
    if b == 0 and c == 0:
        return np.nan
    if b == 0:
        return np.inf
    if c == 0:
        return 0.0
    return b / c


def test_flip_rates(subset: pd.DataFrame) -> pd.DataFrame:
    """Binomial test: flip rate differs from 0 for each non-prior condition."""
    rows = []
    for condition in NON_PRIOR_CONDITIONS:
        cond_df = subset[subset["condition"] == condition]
        n = len(cond_df)
        if n == 0:
            continue
        flips = int(cond_df["flipped"].sum())
        # H0: flip probability = 0
        p_value = stats.binomtest(flips, n, p=0.0, alternative="greater").pvalue
        rows.append(
            {
                "test": "binom_flip_gt_zero",
                "comparison": f"{condition} vs prior",
                "n": n,
                "statistic": flips,
                "flip_rate_pct": 100.0 * flips / n,
                "p_value": p_value,
                "effect_size": flips / n,
                "effect_name": "flip_proportion",
            }
        )
    return pd.DataFrame(rows)


def test_accuracy_mcnemar(subset: pd.DataFrame) -> pd.DataFrame:
    """McNemar test for prior vs. each condition accuracy (paired by question)."""
    if subset["is_correct"].isna().all():
        return pd.DataFrame()

    rows = []
    prior = (
        subset[subset["condition"] == "prior"][["question_id", "is_correct"]]
        .drop_duplicates("question_id")
        .set_index("question_id")["is_correct"]
    )

    for condition in NON_PRIOR_CONDITIONS:
        cond = (
            subset[subset["condition"] == condition][["question_id", "is_correct"]]
            .drop_duplicates("question_id")
            .set_index("question_id")["is_correct"]
        )
        paired = pd.concat([prior, cond], axis=1, keys=["prior", "cond"]).dropna()
        if paired.empty:
            continue

        # b = prior correct, condition wrong; c = prior wrong, condition correct
        b = int((paired["prior"] & ~paired["cond"]).sum())
        c = int((~paired["prior"] & paired["cond"]).sum())
        table = np.array([[0, b], [c, 0]])
        if b + c == 0:
            p_value = 1.0
            statistic = 0.0
        else:
            result = mcnemar(table, exact=(b + c) < 25)
            p_value = float(result.pvalue)
            statistic = float(result.statistic) if result.statistic is not None else np.nan

        rows.append(
            {
                "test": "mcnemar_accuracy",
                "comparison": f"prior vs {condition}",
                "n": len(paired),
                "statistic": statistic,
                "discordant_correct_to_wrong": b,
                "discordant_wrong_to_correct": c,
                "p_value": p_value,
                "effect_size": _odds_ratio_mcnemar(b, c),
                "effect_name": "odds_ratio_discordant",
            }
        )
    return pd.DataFrame(rows)


def test_winner_distribution(subset: pd.DataFrame) -> pd.DataFrame:
    """Chi-square or Fisher test: winner side frequencies differ from 50/50."""
    rows = []
    for condition in CONFLICT_CONDITIONS:
        cond_df = subset[(subset["condition"] == condition) & subset["winner"].notna()]
        counts = cond_df["winner"].value_counts()
        side_a, side_b = WINNER_SIDES[condition]
        a = int(counts.get(side_a, 0))
        b = int(counts.get(side_b, 0))
        n = a + b
        if n == 0:
            continue

        table = np.array([[a, b]])
        if n < 20:
            # Fisher exact on 2x2 by adding complementary category implicitly
            _, p_value = stats.fisher_exact([[a, b], [b, a]])
            statistic = np.nan
            test_name = "fisher_winner_balance"
        else:
            chi2, p_value, _, _ = stats.chisquare([a, b], f_exp=[n / 2, n / 2])
            statistic = chi2
            test_name = "chi2_winner_balance"

        rows.append(
            {
                "test": test_name,
                "comparison": condition,
                "n": n,
                "statistic": statistic,
                f"{side_a}_count": a,
                f"{side_b}_count": b,
                f"{side_a}_pct": 100.0 * a / n,
                f"{side_b}_pct": 100.0 * b / n,
                "p_value": p_value,
                "effect_size": _cramers_v(statistic if not np.isnan(statistic) else 0, n, 2),
                "effect_name": "cramers_v",
            }
        )
    return pd.DataFrame(rows)


def test_confidence_change(subset: pd.DataFrame) -> pd.DataFrame:
    """Paired t-test or Wilcoxon on confidence change vs. 0 (per question)."""
    rows = []
    for condition in NON_PRIOR_CONDITIONS:
        cond_df = subset[subset["condition"] == condition]
        changes = cond_df["confidence_change"].dropna().to_numpy()
        if len(changes) < 3:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
            shapiro_p = stats.shapiro(changes).pvalue if 3 <= len(changes) <= 5000 else np.nan

        if not np.isnan(shapiro_p) and shapiro_p < ALPHA:
            stat, p_value = stats.wilcoxon(changes, alternative="two-sided", zero_method="wilcox")
            test_name = "wilcoxon_confidence_change"
            effect = stat
            effect_name = "wilcoxon_W"
        else:
            stat, p_value = stats.ttest_1sample(changes, popmean=0.0)
            test_name = "paired_t_confidence_change"
            effect = _cohens_d_paired(changes)
            effect_name = "cohens_d"

        rows.append(
            {
                "test": test_name,
                "comparison": f"{condition} vs prior",
                "n": len(changes),
                "statistic": stat,
                "mean_change": changes.mean(),
                "median_change": np.median(changes),
                "p_value": p_value,
                "effect_size": effect,
                "effect_name": effect_name,
                "normality_shapiro_p": shapiro_p,
            }
        )
    return pd.DataFrame(rows)


def apply_multiple_testing_correction(tests: pd.DataFrame) -> pd.DataFrame:
    """Bonferroni-adjusted p-values across all tests in a model block."""
    if tests.empty:
        return tests
    out = tests.copy()
    m = len(out)
    out["p_value_bonferroni"] = np.minimum(out["p_value"] * m, 1.0)
    out["significant_raw"] = out["p_value"] < ALPHA
    out["significant_bonferroni"] = out["p_value_bonferroni"] < ALPHA
    return out


def run_statistical_tests(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run all statistical tests per model and for all models combined."""
    all_tests: dict[str, pd.DataFrame] = {}
    for model_label, subset in model_groups(df):
        prefix = _safe_name(model_label)
        parts = [
            test_flip_rates(subset),
            test_accuracy_mcnemar(subset),
            test_winner_distribution(subset),
            test_confidence_change(subset),
        ]
        combined = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        if combined.empty:
            continue
        combined = combined.assign(model=model_label)
        combined = apply_multiple_testing_correction(combined)
        all_tests[f"{prefix}_statistical_tests"] = combined
    return all_tests


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _safe_name(label: str) -> str:
    return re.sub(r"[^\w.-]+", "_", label)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as CSV and Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False, float_format="%.4f")
    md = df.copy()
    for col in md.select_dtypes(include=[np.floating]).columns:
        md[col] = md[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    path.with_suffix(".md").write_text(md.to_markdown(index=False))


def save_all_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    for name, table in tables.items():
        save_table(table, output_dir / "tables" / name)


def _short_model_name(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model


def plot_flip_rates(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    frames = [t for k, t in tables.items() if k.endswith("_flip_rates")]
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    data = data[data["condition"] != "prior"]

    g = sns.catplot(
        data=data,
        x="condition",
        y="flip_rate_pct",
        hue="model",
        kind="bar",
        height=5,
        aspect=1.6,
        palette="colorblind",
    )
    g.set_axis_labels("Condition", "Flip rate (%)")
    g.set_xticklabels(rotation=35, ha="right")
    g.fig.suptitle("Answer flip rate relative to prior", y=1.02)
    g.savefig(output_dir / "figures" / "flip_rates_bar.png", bbox_inches="tight")
    plt.close(g.fig)


def plot_winner_bars(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    frames = [t for k, t in tables.items() if k.endswith("_winner_stats")]
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)

    g = sns.catplot(
        data=data,
        x="condition",
        y="pct",
        hue="winner",
        col="model",
        kind="bar",
        col_wrap=2,
        height=4,
        aspect=1.2,
        palette="Set2",
        sharey=True,
    )
    g.set_axis_labels("Condition", "Winner share (%)")
    g.set_titles("{col_name}")
    g.fig.suptitle("Conflict winner frequencies", y=1.03)
    g.savefig(output_dir / "figures" / "winner_frequencies_bar.png", bbox_inches="tight")
    plt.close(g.fig)


def plot_accuracy_bars(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    frames = [t for k, t in tables.items() if k.endswith("_accuracy_stats")]
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)

    g = sns.catplot(
        data=data,
        x="condition",
        y="accuracy_pct",
        hue="model",
        kind="bar",
        height=5,
        aspect=1.6,
        palette="colorblind",
    )
    g.set_axis_labels("Condition", "Accuracy (%)")
    g.set_xticklabels(rotation=35, ha="right")
    g.fig.suptitle("Accuracy by condition", y=1.02)
    g.savefig(output_dir / "figures" / "accuracy_bar.png", bbox_inches="tight")
    plt.close(g.fig)


def plot_confidence_heatmap(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    frames = [t for k, t in tables.items() if k.endswith("_confidence_stats")]
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    pivot = data.pivot_table(index="model", columns="condition", values="mean_confidence")
    pivot = pivot.reindex(columns=CONDITIONS)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, ax=ax, cbar_kws={"label": "Mean confidence"})
    ax.set_title("Mean confidence by model and condition")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "confidence_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_transition_heatmap(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    frames = [t for k, t in tables.items() if k.endswith("_transitions")]
    if not frames:
        return

    for table_name, data in ((k, t) for k, t in tables.items() if k.endswith("_transitions")):
        model_label = data["model"].iloc[0]
        pct_cols = [f"{label}_pct" for label in TRANSITION_LABELS]
        matrix = data.set_index("condition")[pct_cols]
        matrix.columns = TRANSITION_SHORT

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5, ax=ax, cbar_kws={"label": "% of trials"})
        ax.set_title(f"Prior→condition accuracy transitions ({_short_model_name(model_label)})")
        ax.set_ylabel("Condition")
        fig.tight_layout()
        safe = _safe_name(model_label)
        fig.savefig(output_dir / "figures" / f"transitions_heatmap_{safe}.png", bbox_inches="tight")
        plt.close(fig)


def plot_confidence_change_bars(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    frames = [t for k, t in tables.items() if k.endswith("_confidence_stats")]
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    data = data[data["condition"] != "prior"]

    g = sns.catplot(
        data=data,
        x="condition",
        y="mean_confidence_change",
        hue="model",
        kind="bar",
        height=5,
        aspect=1.6,
        palette="colorblind",
    )
    g.set_axis_labels("Condition", "Mean confidence change vs. prior")
    g.set_xticklabels(rotation=35, ha="right")
    g.ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    g.fig.suptitle("Confidence shift relative to prior", y=1.02)
    g.savefig(output_dir / "figures" / "confidence_change_bar.png", bbox_inches="tight")
    plt.close(g.fig)


def generate_figures(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    plot_flip_rates(tables, output_dir)
    plot_winner_bars(tables, output_dir)
    plot_accuracy_bars(tables, output_dir)
    plot_confidence_heatmap(tables, output_dir)
    plot_transition_heatmap(tables, output_dir)
    plot_confidence_change_bars(tables, output_dir)


def _format_pct(value: float) -> str:
    return f"{value:.1f}%"


def _significant_findings(tests: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame]) -> list[str]:
    """Collect bullet points for statistically significant (Bonferroni) results."""
    bullets: list[str] = []

    for name, test_df in tests.items():
        model = test_df["model"].iloc[0] if not test_df.empty else name
        model_short = _short_model_name(model)
        sig = test_df[test_df["significant_bonferroni"]]

        for _, row in sig.iterrows():
            comparison = row["comparison"]
            p = row["p_value_bonferroni"]
            test = row["test"]

            if test.startswith("binom"):
                bullets.append(
                    f"**{model_short}**: Flip rate for {comparison} was significantly above zero "
                    f"({row['flip_rate_pct']:.1f}%, Bonferroni *p* = {p:.4f})."
                )
            elif test.startswith("mcnemar"):
                bullets.append(
                    f"**{model_short}**: Accuracy differed between {comparison} "
                    f"(McNemar, Bonferroni *p* = {p:.4f}; "
                    f"CW = {int(row['discordant_correct_to_wrong'])}, "
                    f"WC = {int(row['discordant_wrong_to_correct'])})."
                )
            elif "winner" in test:
                bullets.append(
                    f"**{model_short}**: Winner distribution in {comparison} deviated from chance "
                    f"(Bonferroni *p* = {p:.4f})."
                )
            elif "confidence" in test:
                bullets.append(
                    f"**{model_short}**: Mean confidence change for {comparison} was significant "
                    f"(mean Δ = {row['mean_change']:.2f}, Bonferroni *p* = {p:.4f})."
                )

    # Descriptive highlights even if not significant
    for name, flip_df in tables.items():
        if not name.endswith("_flip_rates"):
            continue
        model = flip_df["model"].iloc[0]
        model_short = _short_model_name(model)
        top = flip_df[flip_df["condition"] != "prior"].sort_values("flip_rate_pct", ascending=False).head(1)
        if not top.empty:
            row = top.iloc[0]
            bullets.append(
                f"**{model_short}**: Highest flip rate observed for `{row['condition']}` "
                f"({_format_pct(row['flip_rate_pct'])})."
            )

    for name, winner_df in tables.items():
        if not name.endswith("_winner_stats"):
            continue
        model = winner_df["model"].iloc[0]
        model_short = _short_model_name(model)
        for condition in CONFLICT_CONDITIONS:
            sub = winner_df[winner_df["condition"] == condition]
            if sub.empty:
                continue
            leader = sub.sort_values("pct", ascending=False).iloc[0]
            bullets.append(
                f"**{model_short}**: In `{condition}`, `{leader['winner']}` won "
                f"{_format_pct(leader['pct'])} of conflict trials."
            )

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in bullets:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def generate_results_summary(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    tests: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Write ``results_summary.md`` with an ACL/EMNLP-style Results section."""
    n_questions = df["question_id"].nunique()
    n_models = df["model"].nunique()
    n_trials = len(df)

    sig_count = sum(int(t["significant_bonferroni"].sum()) for t in tests.values())
    bullets = _significant_findings(tests, tables)

    lines = [
        "# Results Summary",
        "",
        f"*Auto-generated from {n_trials} trials across {n_questions} questions "
        f"and {n_models} model(s).*",
        "",
        "## Results",
        "",
        (
            "We evaluated how large language models update yes/no beliefs when exposed "
            "to competing evidence cues (argument strength, vague majority, credible sourcing, "
            "and their pairwise conflicts). For each question, models first answered without "
            "evidence (prior), then under eight evidence conditions. We report flip rates "
            "relative to the prior, conflict-winner frequencies, confidence shifts, and—where "
            "gold labels are available—accuracy transitions."
        ),
        "",
    ]

    # Aggregate descriptive paragraph per model
    for model_label, subset in model_groups(df):
        if model_label == ALL_MODELS_LABEL and n_models == 1:
            continue
        model_short = _short_model_name(model_label)
        flip_key = f"{_safe_name(model_label)}_flip_rates"
        if flip_key not in tables:
            continue
        flip_df = tables[flip_key]
        non_prior = flip_df[flip_df["condition"] != "prior"]
        mean_flip = non_prior["flip_rate_pct"].mean()
        max_row = non_prior.loc[non_prior["flip_rate_pct"].idxmax()]

        conf_key = f"{_safe_name(model_label)}_confidence_stats"
        conf_df = tables.get(conf_key)
        mean_conf = conf_df["mean_confidence"].mean() if conf_df is not None else np.nan

        acc_key = f"{_safe_name(model_label)}_accuracy_stats"
        acc_df = tables.get(acc_key)
        prior_acc = np.nan
        if acc_df is not None:
            prior_rows = acc_df[acc_df["condition"] == "prior"]
            if not prior_rows.empty:
                prior_acc = prior_rows.iloc[0]["accuracy_pct"]

        lines.append(
            f"For **{model_short}**, the average flip rate across evidence conditions was "
            f"{mean_flip:.1f}% (max: {max_row['condition']} at {max_row['flip_rate_pct']:.1f}%). "
            f"Mean confidence averaged {mean_conf:.2f} on the 1–10 scale"
            + (f"; prior accuracy was {prior_acc:.1f}%." if not np.isnan(prior_acc) else ".")
        )

    lines.extend(["", "### Statistically significant findings", ""])
    if sig_count == 0:
        lines.append(
            "After Bonferroni correction, no comparisons reached significance at *p* < 0.05. "
            "Descriptive patterns are reported below."
        )
    else:
        lines.append(
            f"{sig_count} comparison(s) survived Bonferroni correction (*p* < 0.05):"
        )

    lines.append("")
    for bullet in bullets[:20]:
        lines.append(f"- {bullet}")

    lines.extend(
        [
            "",
            "### Materials",
            "",
            "Full tables, figures, and test statistics are saved under `results/analysis/`.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze belief-update experiment results.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Directory with JSONL result files.")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE, help="Path to question.json for gold labels.")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR, help="Directory for analysis outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading JSONL files from {args.results_dir}...")
    df = load_jsonl_files(args.results_dir)

    gold = None
    if args.questions.exists():
        print(f"Loading gold answers from {args.questions}...")
        gold = load_gold_answers(args.questions)
    else:
        print(f"Warning: {args.questions} not found; skipping accuracy analyses.")

    df = prepare_data(df, gold)

    print("Computing descriptive statistics...")
    tables = build_summary_tables(df)

    print("Running statistical tests...")
    tests = run_statistical_tests(df)
    tables.update(tests)

    print(f"Saving tables to {output_dir / 'tables'}...")
    save_all_tables(tables, output_dir)

    print(f"Generating figures in {output_dir / 'figures'}...")
    generate_figures(tables, output_dir)

    print(f"Writing summary to {output_dir / 'results_summary.md'}...")
    generate_results_summary(df, tables, tests, output_dir / "results_summary.md")

    print("Done.")


if __name__ == "__main__":
    main()
