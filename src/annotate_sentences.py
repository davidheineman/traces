from concurrent.futures import ThreadPoolExecutor
import openai
import os
from pathlib import Path
from tqdm import tqdm
import re
import json
from concurrent.futures import as_completed
import spacy

EN_CORE_WEB_SM = spacy.load("en_core_web_sm")

ANNOTATED_DIR = Path("annotated")
OUT_DIR = Path("out")

TAGS = {
    "problem_setup": "Parsing or rephrasing the problem",
    "plan_generation": "Stating or deciding on a plan of action, meta-reasoning",
    "fact_retrieval": "Recalling facts, formulas, problem details without computation",
    "active_computation": "Algebra, calculations, or other manipulations toward the answer",
    "uncertainty_management": "Expressing confusion, re-evaluating, including backtracking",
    "result_consolidation": "Aggregating intermediate results, summarizing, or preparing",
    "self_checking": "Verifying previous steps, checking calculations, and re-confirmations",
    "final_answer_emission": "Explicitly stating the final answer",
}

PROMPT = """
Full chain of thought:

{trace}

Classify this sentence from the above chain of thought into one of these categories:

1. problem_setup: Parsing or rephrasing the problem
2. plan_generation: Stating or deciding on a plan of action, meta-reasoning
3. fact_retrieval: Recalling facts, formulas, problem details without computation
4. active_computation: Algebra, calculations, or other manipulations toward the answer
5. uncertainty_management: Expressing confusion, re-evaluating, including backtracking
6. result_consolidation: Aggregating intermediate results, summarizing, or preparing
7. self_checking: Verifying previous steps, checking calculations, and re-confirmations
8. final_answer_emission: Explicitly stating the final answer

Sentence: "{sentence}"

Respond with only the category name (e.g., "active_computation").
"""


def split_sentences(text):
    doc = EN_CORE_WEB_SM(text)
    sents = [sent.text for sent in doc.sents]
    return sents


def classify_sentence(trace, sentence):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": PROMPT.format(trace=trace, sentence=sentence)}
        ],
        temperature=0.1,
    )

    tag = response.choices[0].message.content.strip()

    # Validate the tag
    if tag not in TAGS.keys():
        tag = "invalid"

    return tag


def annotate_trace(trace):
    sentences = split_sentences(trace)

    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_sentence = {
            executor.submit(classify_sentence, trace, sentence): sentence
            for sentence in sentences
        }

        tags = []

        # progress bar
        for future in as_completed(future_to_sentence):
            tag = future.result()
            sentence = future_to_sentence[future]
            tags.append((sentence, tag))

    # Sort tags by original sentence order
    sentence_to_tag = dict(tags)
    tags = [sentence_to_tag[sentence] for sentence in sentences]

    annotated_trace = {"tags": tags, "sentences": sentences, "trace": trace}

    return annotated_trace


def main(model_name, alias):
    # load traces
    traces = []
    with open(OUT_DIR / model_name / f"{alias}_responses.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            traces.append(entry["output"][0]["text"])

    traces = traces[:100]

    annotated_traces = []
    for trace in tqdm(traces, desc="Annotating traces"):
        annotated_trace = annotate_trace(trace)
        annotated_traces += [annotated_trace]

    # save annotations
    (ANNOTATED_DIR / model_name).mkdir(parents=True, exist_ok=True)
    with open(ANNOTATED_DIR / model_name / f"{alias}_annotated.jsonl", "w") as f:
        for i, annotated_trace in enumerate(annotated_traces):
            entry = {"index": i, "annotated_trace": annotated_trace}
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Annotate reasoning traces with sentence-level tags")
    parser.add_argument("-m", "--model", required=True, help="Model name")
    parser.add_argument("-t", "--task", required=True, help="Task name")
    
    args = parser.parse_args()
    
    main(args.model, args.task)
