import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from question_texts import enrich_questions
from results_io import (
    LEGACY_RESULTS_PATH,
    RESULTS_DIR,
    migrate_legacy_results,
    model_metadata_path,
    model_results_path,
    write_model_metadata,
)
from results_schema import (
    SYSTEM_PROMPT,
    WINNER_RULES,
    build_user_prompt,
    compute_trial_outcomes,
    is_valid_row,
    parse_model_response,
    parse_validated_model_response,
)

DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"
MODEL = DEFAULT_MODEL
RESULTS_FILE = model_results_path(MODEL)

PROVIDER = "nvidia"
API_ENDPOINT = "https://integrate.api.nvidia.com/v1"
TEMPERATURE = 0.7
MAX_TOKENS = 64
PROMPT_VERSION = "v1"
RPM_LIMIT = 30  # NVIDIA API requests-per-minute cap
MIN_REQUEST_INTERVAL = 60.0 / RPM_LIMIT
MAX_RETRIES = 20

CONDITIONS = {
    "prior": [],
    "strength_alone": ["strength_correct"],
    "majority_vague_alone": ["majority_vague_wrong"],
    "strength_vs_majority": ["strength_correct", "majority_vague_wrong"],
    "majority_vs_credibility": ["majority_plain_correct", "credibility_wrong"],
    "strength_vs_credibility": ["strength_correct", "credibility_wrong"],
    "credibility_alone": ["credibility_correct"],
    "majority_plain_alone": ["majority_plain_wrong"],
}

_client = None
_RateLimitError = None
_APIStatusError = None
_APIConnectionError = None
_last_request_at = 0.0
TRANSIENT_STATUS_CODES = {500, 502, 503, 504}


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def require_api_key() -> str:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print(
            "Error: NVIDIA_API_KEY environment variable is required.\n"
            "  export NVIDIA_API_KEY='your-key-here'",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return api_key


def build_run_metadata(model: str, *, run_started_at: str, run_completed_at: str | None = None) -> dict:
    return {
        "model": model,
        "provider": PROVIDER,
        "api_endpoint": API_ENDPOINT,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "prompt_version": PROMPT_VERSION,
        "rpm_limit": RPM_LIMIT,
        "run_started_at": run_started_at,
        "run_completed_at": run_completed_at,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run belief-update experiment trials.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="NVIDIA API model id (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for per-model JSONL result files",
    )
    parser.add_argument(
        "--migrate-legacy",
        action="store_true",
        help=f"Split {LEGACY_RESULTS_PATH} into per-model files under results/ and exit",
    )
    return parser.parse_args()


def throttle():
    """Space requests evenly to stay under RPM_LIMIT."""
    global _last_request_at
    now = time.time()
    if _last_request_at > 0:
        wait = MIN_REQUEST_INTERVAL - (now - _last_request_at)
        if wait > 0:
            time.sleep(wait)
    _last_request_at = time.time()


def get_client():
    global _client, _RateLimitError, _APIStatusError, _APIConnectionError
    if _client is None:
        log(f"Importing openai (python: {sys.executable})...")
        from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

        _RateLimitError = RateLimitError
        _APIStatusError = APIStatusError
        _APIConnectionError = APIConnectionError
        log("Creating API client...")
        _client = OpenAI(
            base_url=API_ENDPOINT,
            api_key=require_api_key(),
            max_retries=0,
            timeout=30.0,
        )
        log("API client ready.")
    return _client


def validate_question(question_data):
    for field in ("question_id", "question_text", "correct_answer"):
        if field not in question_data:
            raise KeyError(
                f"{question_data.get('question_id', '?')}: missing required field '{field}'"
            )
    for condition, fields in CONDITIONS.items():
        for field in fields:
            if field not in question_data:
                raise KeyError(
                    f"{question_data['question_id']}: missing field '{field}' "
                    f"for condition '{condition}'"
                )


def load_completed_questions(results_file: Path, model: str) -> set[str]:
    """Return question IDs that already have all valid conditions for this model."""
    progress = load_question_progress(results_file, model)
    required = set(CONDITIONS)
    return {
        qid
        for qid, done in progress.items()
        if required <= set(done)
        and all(is_valid_row(done[condition]) for condition in required)
    }


def load_question_progress(results_file: Path, model: str) -> dict[str, dict[str, dict]]:
    """Return saved rows keyed by question_id then condition (latest row wins)."""
    progress: dict[str, dict[str, dict]] = {}
    if not results_file.exists():
        return progress

    with results_file.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("model") != model:
                continue
            progress.setdefault(row["question_id"], {})[row["condition"]] = row
    return progress


def ask_model(question_data, fields, label="?", max_retries=MAX_RETRIES):
    client = get_client()
    prompt = build_user_prompt(question_data, fields)
    last_error = None
    for attempt in range(max_retries):
        try:
            throttle()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = response.choices[0].message.content
            if content is None or not str(content).strip():
                last_error = "empty response content"
                wait = min(30, 2 * (attempt + 1))
                log(f"Empty response on {label} — retry in {wait}s ({attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            raw_response = str(content).strip()
            parsed = parse_validated_model_response(raw_response)
            if parsed is None:
                answer, confidence = parse_model_response(raw_response)
                last_error = f"invalid response: answer={answer!r} confidence={confidence!r}"
                wait = min(30, 2 * (attempt + 1))
                log(
                    f"Invalid response on {label} ({last_error}) — retry in {wait}s "
                    f"({attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
                continue
            answer, confidence = parsed
            return {
                "prompt": prompt,
                "system_prompt": SYSTEM_PROMPT,
                "raw_response": raw_response,
                "answer": answer,
                "confidence": confidence,
            }
        except _RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 60
            log(f"Rate limited on {label} — waiting {wait}s for RPM window reset ({attempt + 1}/{max_retries})")
            time.sleep(wait)
            _last_request_at = 0.0
        except _APIStatusError as e:
            if e.status_code not in TRANSIENT_STATUS_CODES or attempt == max_retries - 1:
                raise
            wait = min(30, 5 * (attempt + 1))
            log(f"Server error {e.status_code} on {label} — retry in {wait}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)
        except _APIConnectionError as e:
            if attempt == max_retries - 1:
                raise
            wait = min(30, 5 * (attempt + 1))
            log(f"Connection error on {label} ({e}) — retry in {wait}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError(f"Failed to get a valid response for {label}: {last_error}")


def run_question(
    question_data,
    results_file: Path,
    question_num,
    total_questions,
    saved_rows: dict[str, dict] | None = None,
):
    correct_answer = question_data["correct_answer"]
    qid = question_data["question_id"]
    n_conditions = len(CONDITIONS)
    saved_rows = saved_rows or {}

    log(f"Question {question_num}/{total_questions} ({qid}) — starting")

    prior_row = saved_rows.get("prior")
    prior_trial = None
    if prior_row and is_valid_row(prior_row):
        prior_answer = prior_row["answer"]
        prior_confidence = prior_row.get("prior_confidence", prior_row.get("confidence"))
        log(f"  [{qid}] prior — reusing saved {prior_answer} {prior_confidence}")
    else:
        if prior_row:
            log(f"  [{qid}] prior — saved row invalid; re-requesting")
        log(f"  [{qid}] 1/{n_conditions} prior — requesting")
        prior_trial = ask_model(question_data, CONDITIONS["prior"], f"{qid}/prior")
        prior_answer = prior_trial["answer"]
        prior_confidence = prior_trial["confidence"]
        log(
            f"  [{qid}] 1/{n_conditions} prior — got {prior_answer} {prior_confidence} "
            f"(raw: {prior_trial['raw_response']!r})"
        )

    results = []
    for i, (condition, fields) in enumerate(CONDITIONS.items(), 1):
        saved = saved_rows.get(condition)
        if saved and is_valid_row(saved):
            log(f"  [{qid}] {i}/{n_conditions} {condition} — skipping (already saved)")
            results.append(saved)
            continue
        if saved:
            log(f"  [{qid}] {i}/{n_conditions} {condition} — saved row invalid; re-requesting")

        if condition == "prior":
            trial = prior_trial or {
                "prompt": build_user_prompt(question_data, fields),
                "system_prompt": SYSTEM_PROMPT,
                "raw_response": None,
                "answer": prior_answer,
                "confidence": prior_confidence,
            }
        else:
            log(f"  [{qid}] {i}/{n_conditions} {condition} — requesting")
            trial = ask_model(question_data, fields, f"{qid}/{condition}")
            log(
                f"  [{qid}] {i}/{n_conditions} {condition} — got {trial['answer']} "
                f"{trial['confidence']} (raw: {trial['raw_response']!r})"
            )

        answer = trial["answer"]
        confidence = trial["confidence"]
        flipped, winner = compute_trial_outcomes(
            answer,
            prior_answer,
            correct_answer,
            condition,
        )

        row = {
            "model": MODEL,
            "question_id": question_data["question_id"],
            "condition": condition,
            "prompt": trial["prompt"],
            "system_prompt": trial["system_prompt"],
            "raw_response": trial["raw_response"],
            "answer": answer,
            "confidence": confidence,
            "prior_answer": prior_answer,
            "prior_confidence": prior_confidence,
            "flipped": flipped,
            "winner": winner,
        }
        results.append(row)

        with results_file.open("a") as f:
            f.write(json.dumps(row) + "\n")

    log(f"Question {question_num}/{total_questions} ({qid}) — complete")
    return results


if __name__ == "__main__":
    args = parse_args()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting main.py...", flush=True)
    MODEL = args.model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE = model_results_path(MODEL, args.results_dir)

    if args.migrate_legacy:
        paths = migrate_legacy_results(results_dir=args.results_dir)
        if paths:
            log(f"Migrated {LEGACY_RESULTS_PATH} into {len(paths)} file(s):")
            for path in paths:
                log(f"  {path}")
        else:
            log(f"No legacy file found at {LEGACY_RESULTS_PATH}")
        raise SystemExit(0)

    require_api_key()

    # One-time convenience: import legacy rows if the per-model file is empty.
    if not RESULTS_FILE.exists() and LEGACY_RESULTS_PATH.exists():
        migrated = migrate_legacy_results(results_dir=args.results_dir)
        if RESULTS_FILE.exists():
            log(f"Imported existing rows from {LEGACY_RESULTS_PATH} into {RESULTS_FILE}")

    started_at = time.time()
    run_started_at = datetime.now().astimezone().isoformat()
    metadata_path = model_metadata_path(MODEL, args.results_dir)
    write_model_metadata(
        build_run_metadata(MODEL, run_started_at=run_started_at),
        MODEL,
        args.results_dir,
    )

    log(f"Model: {MODEL}")
    log(f"Results file: {RESULTS_FILE}")
    log(f"Metadata file: {metadata_path}")

    if "new_env" in sys.executable:
        log(
            "Warning: new_env is very slow importing openai (often 5–20 min). "
            "For faster runs: /Users/aayushsingh/anaconda3/bin/python main.py ..."
        )
    log("Initializing API client...")
    get_client()

    log("Loading question.json...")
    with open("question.json", "r") as f:
        data = enrich_questions(json.load(f))

    api_calls_per_question = len(CONDITIONS)
    log(
        f"Loaded {len(data)} questions × {len(CONDITIONS)} conditions "
        f"= {len(data) * len(CONDITIONS)} trials (~{len(data) * api_calls_per_question} API calls)"
    )
    log(f"Pacing: {RPM_LIMIT} RPM ({MIN_REQUEST_INTERVAL:.1f}s between requests)")

    log("Validating all questions...")
    for question_data in data:
        validate_question(question_data)
    log("Validation OK")

    completed = load_completed_questions(RESULTS_FILE, MODEL)
    progress = load_question_progress(RESULTS_FILE, MODEL)
    pending = [q for q in data if q["question_id"] not in completed]
    log(f"Progress: {len(completed)}/{len(data)} complete, {len(pending)} pending")
    if completed:
        log(f"Skipping: {', '.join(sorted(completed))}")
    if pending:
        log(f"To run: {', '.join(q['question_id'] for q in pending)}")
    else:
        log(f"Nothing to do — all questions already complete for {MODEL}.")
        write_model_metadata(
            build_run_metadata(
                MODEL,
                run_started_at=run_started_at,
                run_completed_at=datetime.now().astimezone().isoformat(),
            ),
            MODEL,
            args.results_dir,
        )
        raise SystemExit(0)

    total_results = 0
    for i, question_data in enumerate(data, 1):
        qid = question_data["question_id"]
        if qid in completed:
            continue
        total_results += len(
            run_question(
                question_data,
                RESULTS_FILE,
                i,
                len(data),
                saved_rows=progress.get(qid, {}),
            )
        )

    elapsed_min = (time.time() - started_at) / 60
    write_model_metadata(
        build_run_metadata(
            MODEL,
            run_started_at=run_started_at,
            run_completed_at=datetime.now().astimezone().isoformat(),
        ),
        MODEL,
        args.results_dir,
    )
    log(f"Done — {total_results} trials appended to {RESULTS_FILE} in {elapsed_min:.1f} min")
