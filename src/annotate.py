from concurrent.futures import ThreadPoolExecutor
import openai

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
    
    with ThreadPoolExecutor() as executor:
        annotated_traces = list(executor.map(annotate_trace, traces))
