"""Evaluation system for the IronStore assistant.

Runs every question in eval/dataset.json through the real app code (api/rag.py), lets four
LLM judges grade each answer, and writes a report to eval/results/.

    python eval/run_eval.py                 # all questions; LangSmith experiment if a key is set
    python eval/run_eval.py --local         # same, without LangSmith
    python eval/run_eval.py --only list sku # only some question types
    python eval/run_eval.py --limit 3       # quick smoke run

Judges (LLM-as-judge, gpt-4.1-mini, pass/fail + reason):
    correctness          answer vs reference answer      wrong or missing facts
    relevance            answer vs question              off-topic answers
    groundedness         answer vs retrieved chunks      hallucination
    retrieval_relevance  retrieved chunks vs question    search failures (n/a for unanswerable questions)

Cost: about 18 app answers + 72 short judge calls, a few cents.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# this PC tries IPv6 first and waits ~10 s per call; prefer IPv4 (no effect elsewhere)
_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: sorted(_getaddrinfo(*a, **k), key=lambda x: x[0] != socket.AF_INET)

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
sys.path.insert(0, str(ROOT / "api"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

if sys.platform == "win32":                     # keys saved with setx after this shell started
    import winreg
    for _name in ("OPENAI_API_KEY", "PINECONE_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT"):
        if not os.environ.get(_name):
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as _key:
                    os.environ[_name] = winreg.QueryValueEx(_key, _name)[0]
            except OSError:
                pass

from openai import OpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import rag  # noqa: E402

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
DATASET_NAME = "ironstore-app-eval"
KEYS = ["correctness", "relevance", "groundedness", "retrieval_relevance"]


# ---------- the app under test ----------

_passages = []
_original_retrieve = rag.retrieve


def _capturing_retrieve(question):
    """Same as rag.retrieve, but remembers the chunks the answer model will read (for the judges)."""
    kept, log = _original_retrieve(question)
    _passages[:] = [c["text"] for c in kept]
    return kept, log


rag.retrieve = _capturing_retrieve


def run_app(inputs):
    _passages.clear()
    history = [rag.Turn(**t) for t in inputs.get("history", [])]
    out = rag.ask(inputs["question"], history)
    return {"answer": out["answer"], "documents": list(_passages), "path": out.get("path"),
            "route": out.get("route"), "seconds": sum(out["timings"].values())}


# ---------- the judges ----------

class Grade(BaseModel):
    explanation: str = Field(description="One or two sentences explaining the grade.")
    passed: bool


_judge = None


def judge(instructions, content):
    global _judge
    _judge = _judge or OpenAI()
    return _judge.chat.completions.parse(
        model=JUDGE_MODEL, temperature=0, response_format=Grade,
        messages=[{"role": "system", "content": instructions}, {"role": "user", "content": content}],
    ).choices[0].message.parsed


def correctness(inputs, outputs, reference_outputs):
    g = judge("You grade an answer from a company assistant against a REFERENCE answer from the company documents.\n"
              "Pass if it is factually consistent with the reference and contains its key facts (numbers, prices, "
              "codes, names). Extra correct detail is fine. Follow any rule the reference states (e.g. what must or "
              "must not be listed, or that a warning is required). If the reference says the information is not in "
              "the documents, pass only if the answer says so without inventing.",
              f"QUESTION: {inputs['question']}\n\nREFERENCE: {reference_outputs['answer']}\n\n"
              f"ANSWER: {outputs['answer']}")
    return {"key": "correctness", "score": int(g.passed), "comment": g.explanation}


def relevance(inputs, outputs):
    g = judge("Grade whether the ANSWER addresses the QUESTION and helps answer it. Saying clearly that the "
              "information is not available is relevant for questions the documents cannot answer.",
              f"QUESTION: {inputs['question']}\n\nANSWER: {outputs['answer']}")
    return {"key": "relevance", "score": int(g.passed), "comment": g.explanation}


def groundedness(inputs, outputs):
    g = judge("Grade whether every factual claim in the ANSWER is supported by the FACTS (retrieved chunks). "
              "Fail if the answer states facts that are not in the FACTS. Saying something was not found is grounded. "
              "A standard warning that catalog information is outdated is allowed.",
              "FACTS:\n" + "\n\n".join(outputs["documents"]) + f"\n\nANSWER: {outputs['answer']}")
    return {"key": "groundedness", "score": int(g.passed), "comment": g.explanation}


def retrieval_relevance(inputs, outputs, reference_outputs):
    if reference_outputs.get("type") == "unanswerable":
        return {"key": "retrieval_relevance", "score": None, "comment": "n/a: nothing relevant exists"}
    g = judge("Grade whether the retrieved FACTS are relevant to the QUESTION: pass if they contain the information "
              "or clearly related material needed to answer it, even if some chunks are unrelated.",
              f"QUESTION: {inputs['question']}\n\nFACTS:\n" + "\n\n".join(outputs["documents"]))
    return {"key": "retrieval_relevance", "score": int(g.passed), "comment": g.explanation}


EVALUATORS = [correctness, relevance, groundedness, retrieval_relevance]


def grade(example, outputs):
    inputs, reference = example["inputs"], example["outputs"]
    results = []
    for evaluator in EVALUATORS:
        args = {"inputs": inputs, "outputs": outputs, "reference_outputs": reference}
        names = evaluator.__code__.co_varnames[:evaluator.__code__.co_argcount]
        results.append(evaluator(**{k: v for k, v in args.items() if k in names}))
    return results


# ---------- running ----------

def load_examples(only, limit):
    rows = json.loads((EVAL_DIR / "dataset.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if not only or r["type"] in only][:limit or None]
    return [{"inputs": {"question": r["question"], **({"history": r["history"]} if r.get("history") else {})},
             "outputs": {"answer": r["answer"], "type": r["type"]}} for r in rows]


def run_local(examples):
    rows = []
    for n, ex in enumerate(examples, 1):
        print(f"[{n}/{len(examples)}] {ex['inputs']['question'][:70]}", flush=True)
        t0 = time.time()
        outputs = run_app(ex["inputs"])
        rows.append(_row(ex, outputs, grade(ex, outputs), time.time() - t0))
    return rows, None


def run_langsmith(examples, label):
    from langsmith import Client, evaluate
    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        known = {e.inputs["question"] for e in client.list_examples(dataset_id=dataset.id)}
    else:
        dataset = client.create_dataset(dataset_name=DATASET_NAME,
                                        description="IronStore assistant: questions with reference answers (eval/dataset.json)")
        known = set()
    new = [e for e in examples if e["inputs"]["question"] not in known]
    if new:
        client.create_examples(dataset_id=dataset.id, inputs=[e["inputs"] for e in new],
                               outputs=[e["outputs"] for e in new],
                               metadata=[{"type": e["outputs"]["type"]} for e in new])
    wanted = {e["inputs"]["question"] for e in examples}
    data = [e for e in client.list_examples(dataset_id=dataset.id) if e.inputs["question"] in wanted]
    results = evaluate(run_app, data=data, evaluators=EVALUATORS, experiment_prefix=f"ironstore-app-{label}",
                       max_concurrency=1, metadata={"version": label})
    rows = []
    for r in results:
        ex = {"inputs": r["example"].inputs, "outputs": r["example"].outputs}
        grades = [{"key": e.key, "score": e.score, "comment": e.comment} for e in r["evaluation_results"]["results"]]
        seconds = (r["run"].end_time - r["run"].start_time).total_seconds()
        rows.append(_row(ex, r["run"].outputs or {}, grades, seconds))
    return rows, results.experiment_name


def _row(example, outputs, grades, seconds):
    row = {"type": example["outputs"]["type"], "question": example["inputs"]["question"],
           "answer": outputs.get("answer", ""), "path": outputs.get("path"), "seconds": round(seconds, 1)}
    for g in grades:
        row[g["key"]] = g["score"]
        row[g["key"] + "_why"] = g["comment"]
    return row


# ---------- report ----------

def pct(values):
    values = [v for v in values if v is not None]
    return f"{sum(values) / len(values):.0%}" if values else "n/a"


def write_report(rows, label, experiment):
    results_dir = EVAL_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    lines = [f"# IronStore assistant evaluation, {stamp.replace('_', ' ')}", "",
             f"Version: `{label}` · questions: {len(rows)} · judge: `{JUDGE_MODEL}`"
             + (f" · LangSmith experiment: `{experiment}`" if experiment else " · local run"), "",
             "## Pass rate per judge", "", "| Judge | Pass rate |", "|---|---|"]
    lines += [f"| {k} | {pct([r.get(k) for r in rows])} |" for k in KEYS]
    lines += [f"| avg seconds per question | {sum(r['seconds'] for r in rows) / max(1, len(rows)):.1f} |", "",
              "## Correctness by question type", "", "| Type | Questions | Correct |", "|---|---|---|"]
    for t in dict.fromkeys(r["type"] for r in rows):
        group = [r for r in rows if r["type"] == t]
        lines.append(f"| {t} | {len(group)} | {pct([r.get('correctness') for r in group])} |")
    fails = [(r, k) for r in rows for k in KEYS if r.get(k) == 0]
    lines += ["", "## Failed judgements", ""]
    if fails:
        lines += ["| Type | Judge | Question | Judge's reason |", "|---|---|---|---|"]
        lines += [f"| {r['type']} | {k} | {r['question']} | {str(r.get(k + '_why', '')).replace('|', '/')} |"
                  for r, k in fails]
    else:
        lines.append("None: every answer passed every judge.")
    lines += ["", "## All answers", ""]
    for r in rows:
        marks = " ".join(f"{k}={'n/a' if r.get(k) is None else r.get(k)}" for k in KEYS)
        lines += [f"**[{r['type']}] {r['question']}**  ", f"<sub>path: {r['path']} · {marks}</sub>", "",
                  "> " + r["answer"].replace("\n", "\n> "), ""]
    report = results_dir / f"{stamp}_{label}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "latest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="do not send results to LangSmith")
    parser.add_argument("--only", nargs="*", help="question types to run, e.g. list sku policy")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    args = parser.parse_args()
    for key in ("OPENAI_API_KEY", "PINECONE_API_KEY"):
        if not os.environ.get(key):
            sys.exit(f"{key} is not set (environment or .env).")

    try:
        label = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip() or "local"
        if subprocess.run(["git", "status", "--porcelain", "api"], cwd=ROOT, capture_output=True, text=True).stdout:
            label += "-dirty"                # app code changed since that commit
    except OSError:
        label = "local"

    examples = load_examples(args.only, args.limit)
    use_langsmith = not args.local and bool(os.environ.get("LANGSMITH_API_KEY"))
    if use_langsmith:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "ironstore-assistant")
    print(f"Evaluating {len(examples)} questions, version {label}, "
          f"{'LangSmith' if use_langsmith else 'local'} run\n")
    rows, experiment = run_langsmith(examples, label) if use_langsmith else run_local(examples)
    report = write_report(rows, label, experiment)

    print("\nPass rate per judge:")
    for k in KEYS:
        print(f"  {k:<20} {pct([r.get(k) for r in rows])}")
    print(f"\nReport: {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
