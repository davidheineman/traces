import json
import requests
import openai
import re
from pathlib import Path
import tiktoken
from transformers import AutoTokenizer
from tqdm import tqdm
from rich import print as pprint
from openai._types import NOT_GIVEN

# enc = tiktoken.encoding_for_model("gpt-4o-mini")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

quiet = True

ANNOTATED_DIR = Path("annotated")
OUT_DIR = Path("out")
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = """
Rewrite this entire chain of thought but create tags around each of the parts of the CoT. 

Use this set for the tags:

1. [problem_setup]: Parsing or rephrasing the problem
2. [plan_generation]: Stating or deciding on a plan of action, meta-reasoning
3. [fact_retrieval]: Recalling facts, formulas, problem details without computation
4. [active_computation]: Algebra, calculations, or other manipulations toward the answer
5. [uncertainty_management]: Expressing confusion, re-evaluating, including backtracking
6. [result_consolidation]: Aggregating intermediate results, summarizing, or preparing
7. [self_checking]: Verifying previous steps, checking calculations, and re-confirmations
8. [final_answer_emission]: Explicitly stating the final answer

Put the tags like this [final_answer_emission]text example[/final_answer_emission]. MAKE SURE TO GIVE EACH SENTENCE ITS OWN TAG

BEGIN TRACE:

```
{trace}
```

YOUR ANNOTATED TRACE:

```\n"""

TAGS = [
    "[problem_setup]",
    "[plan_generation]",
    "[fact_retrieval]",
    "[active_computation]",
    "[uncertainty_management]",
    "[result_consolidation]",
    "[self_checking]",
    "[final_answer_emission]",
]
CLOSE_TAGS = [tag.replace("[", "[/") for tag in TAGS]
ALL_TAGS = TAGS + CLOSE_TAGS


def generate_token_openai(content, allowed=None):
    client = openai.OpenAI()

    bias = NOT_GIVEN
    if allowed is not None:
        allowed_ids = [enc.encode(x)[0] for x in allowed]
        bias = {tid: 100 for tid in allowed_ids}

    response = client.completions.create(
        model="gpt-4o-mini",
        prompt=content,
        temperature=0.1,
        max_tokens=1,  # Generate only 1 token at a time
        logprobs=3,
        logit_bias=bias,
    )
    choice = response.choices[0]
    content = choice.text
    logprobs = choice.logprobs

    return content, logprobs


def generate_token_vllm(content, allowed=None):
    payload = {
        "prompt": content,
        "max_tokens": 1,
        "temperature": 0.1,
        "logprobs": 3,
    }
    
    # Add logit bias if allowed tokens are specified
    if allowed is not None:
        allowed_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in allowed]
        if not quiet:
            # Debug: print the allowed tokens
            allowed_tokens = [tokenizer.decode([tid]) for tid in allowed_ids]
            print(f"Allowed tokens: {allowed_tokens}")
        payload["logit_bias"] = {str(tid): 100 for tid in allowed_ids}
    
    # Make request to vLLM server
    response = requests.post(
        "http://localhost:8000/v1/completions",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    response.raise_for_status()
    result = response.json()
    
    choice = result["choices"][0]
    content = choice["text"]
    logprobs = choice["logprobs"]
    
    return content, logprobs


def is_open(generation):
    if "[" not in generation:
        return False
    last_open_index = generation.rfind("[")
    return "]" not in generation[last_open_index:]


def annotate_trace(trace):
    curr_trace = trace
    generation = ""

    trace_tokens = tokenizer.encode(trace, add_special_tokens=False)
    
    with tqdm(total=len(trace_tokens), desc="Annotating trace") as pbar:
        while len(curr_trace) > 0:
            content = PROMPT.format(trace=trace) + generation

            _open = is_open(content)

            if _open:
                # at this point it could be:
                # ...<                                          # open!
                # ...<uncertainty_                              # open!
                # ...<uncertainty_management>...<               # open!
                # ...<uncertainty_management>...</uncertainty_  # open!
                # Get content after and including the last <
                last_open_index = content.rfind("[")
                after_last_open = content[last_open_index:]

                # Find which tag this content is a prefix of
                allowed = []
                for tag in ALL_TAGS:
                    if tag.startswith(after_last_open):
                        allowed += [tag[len(after_last_open) :]]

                assert len(allowed) > 0, (allowed, generation)
            else:
                # ...</uncertainty_management>... # closed
                # ...<uncertainty_management>...  # closed
                all_tags = ALL_TAGS
                allowed = [
                    curr_trace[:150],  # only need enough chats for the first token
                    *all_tags,
                    *[" " + tag for tag in all_tags],  # allow " [tag]"
                    *["\n" + tag for tag in all_tags],  # allow "\n[tag]"
                ]

            # next_token, logprobs = generate_token_openai(content, allowed)
            next_token, logprobs = generate_token_vllm(content, allowed)

            if curr_trace.startswith(next_token):
                curr_trace = curr_trace[len(next_token) :]
                pbar.update(1) # Update tqdm

            generation += next_token
    
    return generation


def extract_thought(trace):
    # Extract only the content within the first <think> </think> tokens
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, trace, re.DOTALL)
    
    if not think_match:
        raise RuntimeError("No <think> tags found in trace")
    trace = think_match.group(1) # only keep first
    return trace


def main():
    # Load model outputs
    traces = []
    with open(OUT_DIR / "aime:cot_responses.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            traces.append(entry["output"][0]["text"])

    trace = traces[0]

    trace = extract_thought(trace)

    # Replace [ and ] => ( and )
    # Allows the tags to work with the decoding algorithm
    trace = trace.replace('[', '(').replace(']', ')')

    generation = annotate_trace(trace)

    # Save the annotated generation
    annotated_entry = {
        "original_trace": trace,
        "annotated_generation": generation
    }
    with open(ANNOTATED_DIR / "aime:cot_annotations.jsonl", "w") as f:
        f.write(json.dumps(annotated_entry) + "\n")


if __name__ == "__main__":
    main()
