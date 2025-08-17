```sh
uv venv --python 3.12 /venv
source /venv/bin/activate
```

```sh
pip install minieval
```

```sh
minieval -t minerva_500:cot -m deepseek-ai/DeepSeek-R1-0528-Qwen3-8B -b vllm --writer.save_path out
```

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

Idea: Decoding algorithm for sequence classification that, at each step, only allows the original sequence or a tag.

```python
python annotate_constrained.py # currently 6 TPS on 4o mini (40 minutes for 1 13K token trace)
```