#!/usr/bin/env python3
"""Fit mixed-effects and clustered models for the persuasion study paper tables."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from results_io import FROZEN_CLEAN_PATH, load_dataset_rows
from results_schema import CONDITION_NAMES, normalize_answer, parse_confidence

DEFAULT_DATA = ROOT / FROZEN_CLEAN_PATH
DEFAULT_QUESTIONS = ROOT / "question.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "output" / "paper"

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

PERSUASION_CONDITIONS = [name for name in CONDITION_NAMES if name != "prior"]


def load_frame(data_path: Path | None, questions_path: Path) -> pd.DataFrame:
    if data_path is not None and data_path.exists():
        with data_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        rows = load_dataset_rows(root=ROOT)
        if not rows:
            raise FileNotFoundError(
                f"No dataset found at {DEFAULT_DATA} and no rows under results/"
            )
    with questions_path.open(encoding="utf-8") as handle:
        answer_key = {
            row["question_id"]: normalize_answer(row["correct_answer"])
            for row in json.load(handle)
        }

    df = pd.DataFrame(rows)
    df["confidence_num"] = df["confidence"].map(parse_confidence)
    df["answer_norm"] = df["answer"].map(normalize_answer)
    df["is_correct"] = df.apply(
        lambda row: row["answer_norm"] == answer_key[row["question_id"]],
        axis=1,
    )
    df["flipped"] = df["flipped"].astype(int)
    df["is_correct_int"] = df["is_correct"].astype(int)
    df["qwen"] = (df["model"] == "qwen/qwen3.5-122b-a10b").astype(int)
    df["condition_cat"] = pd.Categorical(df["condition"], categories=CONDITION_NAMES)
    return df


def per_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_NAMES:
        sub = df[df["condition"] == condition]
        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "n_trials": len(sub),
                "flip_rate_pct": 100.0 * sub["flipped"].mean(),
                "accuracy_pct": 100.0 * sub["is_correct"].mean(),
                "mean_confidence": sub["confidence_num"].mean(),
            }
        )
    return pd.DataFrame(rows)


def is_interaction_term(term: str) -> bool:
    return ":qwen" in term


def tidy_mixedlm(fit, outcome: str, *, model_type: str) -> list[dict]:
    rows = []
    ci = fit.conf_int()
    for name in fit.fe_params.index:
        coef = fit.fe_params[name]
        se = fit.bse_fe[name]
        z = coef / se
        rows.append(
            {
                "outcome": outcome,
                "term": name,
                "estimate": coef,
                "std_error": se,
                "statistic": z,
                "p_value": 2 * (1 - stats.norm.cdf(abs(z))),
                "ci_95_lo": ci.loc[name, 0],
                "ci_95_hi": ci.loc[name, 1],
                "model_type": model_type,
            }
        )
    rows.append(
        {
            "outcome": outcome,
            "term": "question_id random intercept variance",
            "estimate": float(fit.cov_re.iloc[0, 0]),
            "std_error": np.nan,
            "statistic": np.nan,
            "p_value": np.nan,
            "ci_95_lo": np.nan,
            "ci_95_hi": np.nan,
            "model_type": model_type,
        }
    )
    return rows


def tidy_gee(fit, outcome: str, *, model_type: str) -> list[dict]:
    rows = []
    ci = fit.conf_int()
    for name in fit.params.index:
        rows.append(
            {
                "outcome": outcome,
                "term": name,
                "estimate": fit.params[name],
                "std_error": fit.bse[name],
                "statistic": fit.tvalues[name],
                "p_value": fit.pvalues[name],
                "ci_95_lo": ci.loc[name, 0],
                "ci_95_hi": ci.loc[name, 1],
                "model_type": model_type,
            }
        )
    return rows


def wald_interaction_omnibus_mixedlm(fit, outcome: str) -> dict:
    names = list(fit.fe_params.index)
    interaction_names = [name for name in names if is_interaction_term(name)]
    if not interaction_names:
        return {
            "outcome": outcome,
            "test": "Wald omnibus (condition × model)",
            "df": 0,
            "statistic": np.nan,
            "p_value": np.nan,
        }

    restriction = ", ".join(f"{name} = 0" for name in interaction_names)
    result = fit.wald_test(restriction)
    stat = float(np.squeeze(result.statistic))
    p_value = float(np.squeeze(result.pvalue))
    return {
        "outcome": outcome,
        "test": "Wald omnibus (condition × model)",
        "df": len(interaction_names),
        "statistic": stat,
        "p_value": p_value,
    }


def wald_interaction_omnibus_gee(fit, outcome: str) -> dict:
    names = list(fit.params.index)
    interaction_names = [name for name in names if is_interaction_term(name)]
    if not interaction_names:
        return {
            "outcome": outcome,
            "test": "Wald omnibus (condition × model)",
            "df": 0,
            "statistic": np.nan,
            "p_value": np.nan,
        }

    restriction = ", ".join(f"{name} = 0" for name in interaction_names)
    result = fit.wald_test(restriction)
    stat = float(np.squeeze(result.statistic))
    p_value = float(np.squeeze(result.pvalue))
    return {
        "outcome": outcome,
        "test": "Wald omnibus (condition × model)",
        "df": len(interaction_names),
        "statistic": stat,
        "p_value": p_value,
    }


def fit_main_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    df_conf = df.dropna(subset=["confidence_num"])

    fit_conf = MixedLM.from_formula(
        "confidence_num ~ C(condition_cat, Treatment(reference='prior')) + qwen",
        groups="question_id",
        data=df_conf,
    ).fit(reml=True, method="lbfgs")
    rows.extend(
        tidy_mixedlm(
            fit_conf,
            "confidence",
            model_type="LMM main effects (random intercept: question)",
        )
    )

    fit_acc = MixedLM.from_formula(
        "is_correct_int ~ C(condition_cat, Treatment(reference='prior')) + qwen",
        groups="question_id",
        data=df,
    ).fit(reml=True, method="lbfgs")
    rows.extend(
        tidy_mixedlm(
            fit_acc,
            "accuracy",
            model_type="LMM main effects (random intercept: question)",
        )
    )

    persuasion = df[df["condition"] != "prior"].copy()
    persuasion["condition_cat"] = pd.Categorical(
        persuasion["condition"],
        categories=PERSUASION_CONDITIONS,
    )
    fit_flip = GEE.from_formula(
        "flipped ~ C(condition_cat, Treatment(reference='majority_plain_alone')) + qwen",
        groups="question_id",
        cov_struct=Exchangeable(),
        family=Binomial(),
        data=persuasion,
    ).fit()
    rows.extend(
        tidy_gee(
            fit_flip,
            "belief_flip",
            model_type="GEE main effects (exchangeable by question)",
        )
    )

    return pd.DataFrame(rows)


def fit_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    omnibus_rows: list[dict] = []
    df_conf = df.dropna(subset=["confidence_num"])

    fit_conf = MixedLM.from_formula(
        "confidence_num ~ C(condition_cat, Treatment(reference='prior')) * qwen",
        groups="question_id",
        data=df_conf,
    ).fit(reml=True, method="lbfgs")
    rows.extend(
        tidy_mixedlm(
            fit_conf,
            "confidence",
            model_type="LMM with condition × model (random intercept: question)",
        )
    )
    omnibus_rows.append(wald_interaction_omnibus_mixedlm(fit_conf, "confidence"))

    fit_acc = MixedLM.from_formula(
        "is_correct_int ~ C(condition_cat, Treatment(reference='prior')) * qwen",
        groups="question_id",
        data=df,
    ).fit(reml=True, method="lbfgs")
    rows.extend(
        tidy_mixedlm(
            fit_acc,
            "accuracy",
            model_type="LMM with condition × model (random intercept: question)",
        )
    )
    omnibus_rows.append(wald_interaction_omnibus_mixedlm(fit_acc, "accuracy"))

    persuasion = df[df["condition"] != "prior"].copy()
    persuasion["condition_cat"] = pd.Categorical(
        persuasion["condition"],
        categories=PERSUASION_CONDITIONS,
    )
    fit_flip = GEE.from_formula(
        "flipped ~ C(condition_cat, Treatment(reference='majority_plain_alone')) * qwen",
        groups="question_id",
        cov_struct=Exchangeable(),
        family=Binomial(),
        data=persuasion,
    ).fit()
    rows.extend(
        tidy_gee(
            fit_flip,
            "belief_flip",
            model_type="GEE with condition × model (exchangeable by question)",
        )
    )
    omnibus_rows.append(wald_interaction_omnibus_gee(fit_flip, "belief_flip"))

    coef_table = pd.DataFrame(rows)
    omnibus_table = pd.DataFrame(omnibus_rows)
    interaction_only = coef_table[coef_table["term"].map(is_interaction_term)].copy()
    return coef_table, omnibus_table, interaction_only


def humanize_term(term: str) -> str:
    if term == "qwen":
        return "Qwen (vs Llama)"
    if term == "Intercept":
        return "Intercept"
    if ":qwen" in term:
        base = term.split(":")[0]
        base = re.sub(r"C\(condition_cat, Treatment\(reference='[^']+'\)\)\[T\.([^\]]+)\]", r"\1", base)
        return f"{CONDITION_LABELS.get(base, base)} × Qwen"
    match = re.search(r"\[T\.([^\]]+)\]", term)
    if match:
        condition = match.group(1)
        return CONDITION_LABELS.get(condition, condition)
    return term


def write_interaction_markdown(
    interaction_table: pd.DataFrame,
    omnibus_table: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# Condition × Model Interaction Tests",
        "",
        "Models include all condition and model main effects plus condition × model interactions.",
        "Reference condition: prior (confidence, accuracy) or majority plain alone (belief flip).",
        "Reference model: Llama 3.1 8B Instruct (`qwen` = 0).",
        "",
        "## Omnibus Wald tests (interaction block)",
        "",
        "| Outcome | df | Statistic | *p* |",
        "|---------|---:|----------:|----:|",
    ]
    for _, row in omnibus_table.iterrows():
        p = row["p_value"]
        p_text = f"{p:.4g}" if pd.notna(p) else "—"
        stat = row["statistic"]
        stat_text = f"{stat:.3f}" if pd.notna(stat) else "—"
        lines.append(
            f"| {row['outcome']} | {int(row['df']) if pd.notna(row['df']) else '—'} "
            f"| {stat_text} | {p_text} |"
        )

    lines.extend(["", "## Interaction coefficients", ""])
    for outcome in interaction_table["outcome"].unique():
        sub = interaction_table[interaction_table["outcome"] == outcome]
        lines.append(f"### {outcome}")
        lines.append("")
        lines.append("| Term | Estimate | SE | *z*/*t* | *p* | 95% CI |")
        lines.append("|------|----------|-----|---------|-----|--------|")
        for _, row in sub.iterrows():
            label = humanize_term(row["term"])
            p = row["p_value"]
            p_text = f"{p:.4g}" if pd.notna(p) and p >= 0.0001 else (
                "&lt; .0001" if pd.notna(p) else "—"
            )
            stat = row["statistic"]
            stat_text = f"{stat:.2f}" if pd.notna(stat) else "—"
            lo = row["ci_95_lo"]
            hi = row["ci_95_hi"]
            ci = f"[{lo:.3f}, {hi:.3f}]" if pd.notna(lo) and pd.notna(hi) else "—"
            lines.append(
                f"| {label} | {row['estimate']:.4f} | {row['std_error']:.4f} "
                f"| {stat_text} | {p_text} | {ci} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit paper regression tables.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    df = load_frame(args.data, args.questions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_condition_summary(df).to_csv(args.out_dir / "per_condition_summary.csv", index=False)

    main_effects = fit_main_effects(df)
    main_effects.to_csv(args.out_dir / "mixed_effects_coefficients.csv", index=False)

    interaction_coefs, omnibus, interaction_only = fit_interactions(df)
    interaction_coefs.to_csv(args.out_dir / "mixed_effects_interactions_full.csv", index=False)
    interaction_only.to_csv(args.out_dir / "mixed_effects_interactions.csv", index=False)
    omnibus.to_csv(args.out_dir / "mixed_effects_interaction_omnibus.csv", index=False)
    write_interaction_markdown(
        interaction_only,
        omnibus,
        args.out_dir / "06_condition_model_interactions.md",
    )

    print(f"Wrote tables to {args.out_dir}")


if __name__ == "__main__":
    main()
