from concurrent.futures import ThreadPoolExecutor
import openai
import os
from pathlib import Path
from tqdm import tqdm

ANNOTATED_DIR = Path('annotated')
OUT_DIR = Path('out')
os.makedirs(ANNOTATED_DIR, exist_ok=True)

PROMPT = """
Rewrite this entire chain of thought but create tags around each of the parts of the CoT. 

Use this set for the tags:

1. Problem Setup: Parsing or rephrasing the problem
2. Plan Generation: Stating or deciding on a plan of action, meta-reasoning
3. Fact Retrieval: Recalling facts, formulas, problem details without computation
4. Active Computation: Algebra, calculations, or other manipulations toward the answer
5. Uncertainty Management: Expressing confusion, re-evaluating, including backtracking
6. Result Consolidation: Aggregating intermediate results, summarizing, or preparing
7. Self Checking: Verifying previous steps, checking calculations, and re-confirmations
8. Final Answer Emission: Explicitly stating the final answer

Put the tags like this <final_answer_emission>text example</final_answer_emission>. make sure all text is tagged

BEGIN TRACE:

{trace}
"""

def annotate_trace(trace):
    client = openai.OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(trace=trace)
            }
        ],
        temperature=0.1
    )
    
    return response.choices[0].message.content


def main():
    traces = []

    import json
    
    with open(OUT_DIR / 'aime:cot_responses.jsonl', 'r') as f:
        for line in f:
            entry = json.loads(line)
            traces.append(entry['output'][0]['text'])

    traces = traces[:10]
    
    with ThreadPoolExecutor() as executor:
        annotated_traces = list(tqdm(executor.map(annotate_trace, traces), total=len(traces)))
    
    with open(ANNOTATED_DIR / 'aime:cot_annotated.jsonl', 'w') as f:
        for i, annotated_trace in enumerate(annotated_traces):
            entry = {
                'index': i,
                'annotated_trace': annotated_trace
            }
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    main()
