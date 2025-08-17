### setup

```sh
pip install -r requirements.txt
```

**Run eval**

```sh
minieval -t minerva_500:cot -m deepseek-ai/DeepSeek-R1-0528-Qwen3-8B -b vllm --writer.save_path out
```

**Additional models**

```sh
open-thoughts/OpenThinker3-7B
Qwen/Qwen3-235B-A22B-Thinking-2507
Qwen/Qwen3-30B-A3B-Thinking-2507
deepseek-ai/DeepSeek-R1
deepseek-ai/DeepSeek-R1-0528
deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
moonshotai/Kimi-K2-Instruct
openai/gpt-oss-20b
openai/gpt-oss-120b
```

**Run vLLM server**

```sh
vllm serve Qwen/Qwen3-32B --port 8000 --max-model-len 32768
```

**Annotate reasoning trace**

```python
# Custom decoder that re-generates the same output, but allows tagging:
    # "Compute 2+3=5.\n" ==> "[problem_setup]Compute 2+3=5.[/problem_setup]"
python src/annotate_constrained.py # currently 6 TPS on 4o mini (40 minutes for 1 13K token trace)

# run in background
nohup python src/annotate_constrained.py > /tmp/out.out 2>&1 &
```