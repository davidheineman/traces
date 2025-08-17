### setup

```sh
pip install -r requirements.txt

# GPT OSS requires pre-release vLLM
uv pip install --pre vllm==0.10.1+gptoss --extra-index-url https://wheels.vllm.ai/gpt-oss/ --extra-index-url https://download.pytorch.org/whl/nightly/cu128  --index-strategy unsafe-best-match
```

**Run eval**

```sh
minieval -t minerva_500:cot -m deepseek-ai/DeepSeek-R1-0528-Qwen3-8B -b vllm --writer.save_path out/DeepSeek-R1-0528-Qwen3-8B
minieval -t minerva_500:cot -m open-thoughts/OpenThinker3-7B -b vllm --writer.save_path out/OpenThinker3-7B
minieval -t minerva_500:cot -m Qwen/Qwen3-30B-A3B-Thinking-2507 -b vllm --writer.save_path out/Qwen3-30B-A3B-Thinking-2507
minieval -t minerva_500:cot -m openai/gpt-oss-20b -b vllm --writer.save_path out/gpt-oss-20b
```

<details>
<summary>Additional models</summary>

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

</details>

**Run vLLM server**

```sh
vllm serve Qwen/Qwen3-32B --port 8000 --max-model-len 32768
```

**Annotate reasoning trace (per-sentence)**

```python
# Use GPT to annotate (~1M input tokens per trace)
python src/annotate_sentences.py -t minerva_500:cot -m OpenThinker3-7B
python src/annotate_sentences.py -t minerva_500:cot -m DeepSeek-R1-0528-Qwen3-8B
```

**Annotate reasoning trace (per-token, custom decoder)**

```python
# Custom decoder that re-generates the same output, but allows tagging:
    # "Compute 2+3=5.\n" ==> "[problem_setup]Compute 2+3=5.[/problem_setup]"
python src/annotate_constrained.py # currently 6 TPS on 4o mini (40 minutes for 1 13K token trace)

# run in background
nohup python src/annotate_constrained.py > /tmp/out.out 2>&1 &
```

### Visualize

```python
python analysis/distribution.py
```

<img src="analysis/dist.png" alt="Distribution of thinking trace components" style="max-width: 500px;">
