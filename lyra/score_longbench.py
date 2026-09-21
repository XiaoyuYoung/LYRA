"""Score predictions produced by ``lyra.eval_longbench``.

LongBench-v1 follows the official per-task metrics.  LongBench-v2 follows its
official exact-choice accuracy and reports Overall, Easy/Hard, and
Short/Medium/Long.  Extra domain, truncation, and numeric token-length slices
make long-context failure modes easier to inspect.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent



DEFAULT_RESULTS = PROJECT_ROOT / "outputs" / "evaluation"


TASK_TO_CATEGORY = {
    "narrativeqa": "single_doc_qa",
    "qasper": "single_doc_qa",
    "multifieldqa_en": "single_doc_qa",
    "multifieldqa_zh": "single_doc_qa",
    "hotpotqa": "multi_doc_qa",
    "2wikimqa": "multi_doc_qa",
    "musique": "multi_doc_qa",
    "dureader": "multi_doc_qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "vcsum": "summarization",
    "trec": "few_shot_learning",
    "triviaqa": "few_shot_learning",
    "samsum": "few_shot_learning",
    "lsht": "few_shot_learning",
    "passage_retrieval_en": "synthetic",
    "passage_retrieval_zh": "synthetic",
    "passage_count": "synthetic",
    "lcc": "code_completion",
    "repobench-p": "code_completion",
}

FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-dir", type=Path, nargs="*", help="Score only these checkpoint result directories.")
    parser.add_argument("--output", type=Path, help="Aggregate JSON path; defaults to <results-dir>/summary.json.")
    parser.add_argument("--compensate-unparsed", action="store_true", help="Give unparsed v2 answers 25%% expected random accuracy (not official default).")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    text = text.lower()
    punctuation = set(string.punctuation)
    text = "".join(character for character in text if character not in punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalize_zh_answer(text: str) -> str:
    cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punctuation = set(string.punctuation + cn_punctuation)
    return "".join(character for character in text.lower() if not character.isspace() and character not in punctuation)


def token_f1(prediction: list[str], ground_truth: list[str]) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    overlap = sum(common.values())
    if overlap == 0 or not prediction or not ground_truth:
        return 0.0
    precision = overlap / len(prediction)
    recall = overlap / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_: Any) -> float:
    return token_f1(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def _jieba_tokens(text: str) -> list[str]:
    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError("Chinese LongBench metrics require `pip install jieba`.") from exc
    return list(jieba.cut(text, cut_all=False))


def qa_f1_zh_score(prediction: str, ground_truth: str, **_: Any) -> float:
    prediction_tokens = [normalize_zh_answer(token) for token in _jieba_tokens(prediction)]
    truth_tokens = [normalize_zh_answer(token) for token in _jieba_tokens(ground_truth)]
    prediction_tokens = [token for token in prediction_tokens if token]
    truth_tokens = [token for token in truth_tokens if token]
    return token_f1(prediction_tokens, truth_tokens)


def rouge_score(prediction: str, ground_truth: str, **_: Any) -> float:
    try:
        from rouge import Rouge
    except ImportError as exc:
        raise RuntimeError("LongBench summarization metrics require `pip install rouge`.") from exc
    try:
        return float(Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"])
    except Exception:
        return 0.0


def rouge_zh_score(prediction: str, ground_truth: str, **_: Any) -> float:
    return rouge_score(" ".join(_jieba_tokens(prediction)), " ".join(_jieba_tokens(ground_truth)))


def count_score(prediction: str, ground_truth: str, **_: Any) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(str(number) == str(ground_truth) for number in numbers) / len(numbers)


def retrieval_score(prediction: str, ground_truth: str, **_: Any) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    return 0.0 if not numbers else sum(number == matches[0] for number in numbers) / len(numbers)


def retrieval_zh_score(prediction: str, ground_truth: str, **_: Any) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    return 0.0 if not numbers else sum(number == matches[0] for number in numbers) / len(numbers)


def classification_score(prediction: str, ground_truth: str, all_classes: list[str] | None, **_: Any) -> float:
    matches = [class_name for class_name in (all_classes or []) if class_name in prediction]
    matches = [term for term in matches if not (term in ground_truth and term != ground_truth)]
    return 1.0 / len(matches) if ground_truth in matches else 0.0


def code_sim_score(prediction: str, ground_truth: str, **_: Any) -> float:
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    try:
        from fuzzywuzzy import fuzz
    except ImportError as exc:
        raise RuntimeError("LongBench code metrics require `pip install fuzzywuzzy`.") from exc
    return float(fuzz.ratio(candidate, ground_truth) / 100)


TASK_TO_METRIC: dict[str, Callable[..., float]] = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def round_percent(value: float) -> float:
    return round(100.0 * value, 2)


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def percent_or_none(values: list[float]) -> float | None:
    value = safe_mean(values)
    return None if value is None else round_percent(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            record_id = str(record.get("id", f"line:{line_number}"))
            if record_id in ids:
                raise ValueError(f"Duplicate id {record_id!r} in {path}")
            ids.add(record_id)
            records.append(record)
    return records


def numeric_length_bucket(length: int | float | None) -> str:
    if length is None:
        return "unknown"
    if length < 4000:
        return "0-4k"
    if length < 8000:
        return "4-8k"
    return "8k+"


def token_bucket(record: dict[str, Any]) -> str:
    tokens = int(record.get("original_input_tokens", record.get("input_tokens", 0)))
    if tokens < 4096:
        return "0-4k"
    if tokens < 8192:
        return "4-8k"
    if tokens < 16384:
        return "8-16k"
    if tokens < 32768:
        return "16-32k"
    return "32k+"


def input_was_truncated(record: dict[str, Any]) -> bool:
    """Read the explicit field while remaining compatible with older runs."""
    return bool(record.get("input_truncated", record.get("truncated", False)))


def score_longbench(run_dir: Path) -> dict[str, Any] | None:
    prediction_dir = run_dir / "longbench"
    if not prediction_dir.is_dir():
        return None
    task_scores: dict[str, float] = {}
    task_means: dict[str, float] = {}
    task_counts: dict[str, int] = {}
    task_input_truncated: dict[str, int] = {}
    task_output_truncated: dict[str, int] = {}
    category_scores: defaultdict[str, list[float]] = defaultdict(list)
    length_scores: defaultdict[str, list[float]] = defaultdict(list)
    token_scores: defaultdict[str, list[float]] = defaultdict(list)
    truncation_scores: defaultdict[str, list[float]] = defaultdict(list)
    all_sample_scores: list[float] = []

    for path in sorted(prediction_dir.glob("*.jsonl")):
        task = path.stem
        if task not in TASK_TO_METRIC:
            raise ValueError(f"No official metric registered for {task!r}")
        records = read_jsonl(path)
        scores: list[float] = []
        for record in records:
            prediction = str(record.get("pred", ""))
            if task in FIRST_LINE_TASKS:
                prediction = prediction.lstrip("\n").split("\n")[0]
            answers = record.get("answers") or []
            score = max(
                (
                    TASK_TO_METRIC[task](
                        prediction,
                        str(answer),
                        all_classes=record.get("all_classes"),
                    )
                    for answer in answers
                ),
                default=0.0,
            )
            scores.append(score)
            all_sample_scores.append(score)
            length_scores[numeric_length_bucket(record.get("length"))].append(score)
            token_scores[token_bucket(record)].append(score)
            truncation_scores["truncated" if input_was_truncated(record) else "not_truncated"].append(score)
        task_means[task] = safe_mean(scores) or 0.0
        task_scores[task] = round_percent(task_means[task])
        task_counts[task] = len(records)
        task_input_truncated[task] = sum(input_was_truncated(record) for record in records)
        task_output_truncated[task] = sum(bool(record.get("output_truncated")) for record in records)
        category_scores[TASK_TO_CATEGORY[task]].append(task_means[task])

    ordered_length = ("0-4k", "4-8k", "8k+", "unknown")
    ordered_tokens = ("0-4k", "4-8k", "8-16k", "16-32k", "32k+")
    return {
        "metric": "official LongBench per-task metric, percent",
        "macro_average": round_percent(mean(task_means.values())) if task_means else None,
        "micro_sample_average": percent_or_none(all_sample_scores),
        "samples_scored": sum(task_counts.values()),
        "tasks_scored": len(task_scores),
        "task_scores": task_scores,
        "task_counts": task_counts,
        "task_truncated_counts": task_input_truncated,
        "task_input_truncated_counts": task_input_truncated,
        "task_output_truncated_counts": task_output_truncated,
        "category_macro_scores": {key: percent_or_none(values) for key, values in sorted(category_scores.items())},
        "source_length_scores": {key: percent_or_none(length_scores[key]) for key in ordered_length if length_scores[key]},
        "input_token_scores": {key: percent_or_none(token_scores[key]) for key in ordered_tokens if token_scores[key]},
        "truncation_scores": {key: percent_or_none(values) for key, values in sorted(truncation_scores.items())},
        "truncated_samples": sum(task_input_truncated.values()),
        "input_truncated_samples": sum(task_input_truncated.values()),
        "output_truncated_samples": sum(task_output_truncated.values()),
    }


def extract_v2_answer(response: str) -> str | None:
    cleaned = response.replace("*", "").strip()
    patterns = (
        r"the correct answer is\s*\(?([A-D])\)?",
        r"(?:final\s+)?answer\s*(?:is|:)\s*\(?([A-D])\)?",
        r"^\s*\(?([A-D])\)?(?:[.。,:：\s]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def score_longbench_v2(run_dir: Path, compensate_unparsed: bool) -> dict[str, Any] | None:
    path = run_dir / "longbench_v2.jsonl"
    if not path.is_file():
        return None
    records = read_jsonl(path)
    dimensions: dict[str, defaultdict[str, list[float]]] = {
        "difficulty": defaultdict(list),
        "length": defaultdict(list),
        "domain": defaultdict(list),
        "input_tokens": defaultdict(list),
        "truncation": defaultdict(list),
        "output_truncation": defaultdict(list),
    }
    scores: list[float] = []
    unparsed = 0
    for record in records:
        pred = record.get("pred") or extract_v2_answer(str(record.get("response", "")))
        if pred is None:
            unparsed += 1
            score = 0.25 if compensate_unparsed else 0.0
        else:
            score = float(str(pred).upper() == str(record["answer"]).upper())
        scores.append(score)
        dimensions["difficulty"][str(record.get("difficulty", "unknown"))].append(score)
        dimensions["length"][str(record.get("length", "unknown"))].append(score)
        dimensions["domain"][str(record.get("domain", "unknown"))].append(score)
        dimensions["input_tokens"][token_bucket(record)].append(score)
        dimensions["truncation"]["truncated" if input_was_truncated(record) else "not_truncated"].append(score)
        dimensions["output_truncation"][
            "truncated" if record.get("output_truncated") else "not_truncated"
        ].append(score)

    def summarize(groups: defaultdict[str, list[float]]) -> dict[str, dict[str, float | int | None]]:
        return {
            key: {"accuracy": percent_or_none(values), "count": len(values)}
            for key, values in sorted(groups.items())
        }

    return {
        "metric": "exact multiple-choice accuracy, percent",
        "overall": percent_or_none(scores),
        "samples_scored": len(records),
        "unparsed_answers": unparsed,
        "compensate_unparsed": compensate_unparsed,
        "difficulty": summarize(dimensions["difficulty"]),
        "length": summarize(dimensions["length"]),
        "domain": summarize(dimensions["domain"]),
        "input_tokens": summarize(dimensions["input_tokens"]),
        "truncation": summarize(dimensions["truncation"]),
        "output_truncation": summarize(dimensions["output_truncation"]),
        "truncated_samples": sum(input_was_truncated(record) for record in records),
        "input_truncated_samples": sum(input_was_truncated(record) for record in records),
        "output_truncated_samples": sum(bool(record.get("output_truncated")) for record in records),
    }


def discover_run_dirs(results_dir: Path, explicit: list[Path] | None) -> list[Path]:
    if explicit:
        candidates = [path.expanduser().resolve() for path in explicit]
    else:
        root = results_dir.expanduser().resolve()
        candidates = []
        if (root / "longbench").is_dir() or (root / "longbench_v2.jsonl").is_file():
            candidates.append(root)
        if root.is_dir():
            candidates.extend(path.parent for path in root.rglob("run_config.json"))
            candidates.extend(path.parent for path in root.rglob("longbench_v2.jsonl"))
            candidates.extend(path.parent for path in root.rglob("longbench") if path.is_dir())
    runs = sorted(
        {
            path
            for path in candidates
            if (path / "longbench").is_dir() or (path / "longbench_v2.jsonl").is_file()
        }
    )
    return runs


def score_run(run_dir: Path, compensate_unparsed: bool) -> dict[str, Any]:
    manifest_path = run_dir / "run_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    contaminated = manifest.get("contaminated_training_benchmark")
    result = {
        "run_dir": str(run_dir),
        "checkpoint": manifest.get("checkpoint"),
        "contaminated_training_benchmark": contaminated,
        "warning": manifest.get(
            "warning",
            "Training-data contamination status is unknown because run_config.json is missing.",
        ),
        "longbench": score_longbench(run_dir),
        "longbench_v2": score_longbench_v2(run_dir, compensate_unparsed),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def print_table(results: list[dict[str, Any]]) -> None:
    print(
        "run\tLongBench_macro\tLongBench-v2\tV2_easy\tV2_hard\tV2_short\tV2_medium\tV2_long"
        "\tinput_truncated\toutput_truncated"
    )
    for result in results:
        v1 = result.get("longbench") or {}
        v2 = result.get("longbench_v2") or {}
        difficulty = v2.get("difficulty", {})
        length = v2.get("length", {})

        def accuracy(group: dict[str, Any], key: str) -> Any:
            return group.get(key, {}).get("accuracy", "-")

        input_truncated = (v1.get("input_truncated_samples", 0) or 0) + (
            v2.get("input_truncated_samples", 0) or 0
        )
        output_truncated = (v1.get("output_truncated_samples", 0) or 0) + (
            v2.get("output_truncated_samples", 0) or 0
        )
        print(
            "\t".join(
                map(
                    str,
                    (
                        Path(result["run_dir"]).name,
                        v1.get("macro_average", "-"),
                        v2.get("overall", "-"),
                        accuracy(difficulty, "easy"),
                        accuracy(difficulty, "hard"),
                        accuracy(length, "short"),
                        accuracy(length, "medium"),
                        accuracy(length, "long"),
                        input_truncated,
                        output_truncated,
                    ),
                )
            )
        )


def self_test() -> None:
    assert math.isclose(qa_f1_score("The red fox", "red fox"), 1.0)
    assert count_score("There are 7.", "7") == 1.0
    assert retrieval_score("Paragraph 12", "Paragraph 12") == 1.0
    assert retrieval_zh_score("段落3", "段落3") == 1.0
    assert classification_score("City", "City", ["City", "Country"]) == 1.0
    assert extract_v2_answer("The correct answer is (C)") == "C"
    assert extract_v2_answer("Answer: b") == "B"
    assert numeric_length_bucket(3999) == "0-4k"
    assert numeric_length_bucket(4000) == "4-8k"
    assert numeric_length_bucket(8000) == "8k+"
    with tempfile.TemporaryDirectory(prefix="longbench-score-test-") as temporary:
        run_dir = Path(temporary)
        (run_dir / "longbench").mkdir()
        v1_record = {
            "id": "hotpotqa:0",
            "pred": "Miller v. California",
            "answers": ["Miller v. California"],
            "all_classes": None,
            "length": 8616,
            "original_input_tokens": 9000,
            "input_truncated": False,
            "output_truncated": False,
        }
        (run_dir / "longbench" / "hotpotqa.jsonl").write_text(
            json.dumps(v1_record) + "\n", encoding="utf-8"
        )
        v2_record = {
            "id": "example",
            "response": "The correct answer is (C)",
            "pred": "C",
            "answer": "C",
            "difficulty": "hard",
            "length": "long",
            "domain": "Single-Document QA",
            "original_input_tokens": 20000,
            "input_truncated": True,
            "output_truncated": False,
        }
        (run_dir / "longbench_v2.jsonl").write_text(
            json.dumps(v2_record) + "\n", encoding="utf-8"
        )
        v1_result = score_longbench(run_dir)
        v2_result = score_longbench_v2(run_dir, compensate_unparsed=False)
        assert v1_result and v1_result["macro_average"] == 100.0
        assert v2_result and v2_result["overall"] == 100.0
        assert v2_result["length"]["long"]["accuracy"] == 100.0
        assert v1_result["output_truncated_samples"] == 0
        assert v2_result["input_truncated_samples"] == 1
    print("Self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_dirs = discover_run_dirs(args.results_dir, args.run_dir)
    if not run_dirs:
        raise RuntimeError("No prediction runs found")
    results = [score_run(run_dir, args.compensate_unparsed) for run_dir in run_dirs]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "compensate_unparsed": args.compensate_unparsed,
        "runs": results,
    }
    output = args.output or args.results_dir.expanduser().resolve() / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print_table(results)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
