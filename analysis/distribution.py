import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from collections import defaultdict
import pandas as pd
from pathlib import Path
import numpy as np

print('imports done')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

# Get all model directories
annotated_dir = Path("/root/ai2/traces/annotated")
model_dirs = [d for d in annotated_dir.iterdir() if d.is_dir()]

# Store results for all models
all_model_results = {}

for model_dir in model_dirs:
    model_name = model_dir.name
    print(f"Processing {model_name}...")
    
    # Find the annotated file (assuming there's one per model)
    annotated_files = list(model_dir.glob("*_annotated.jsonl"))
    if not annotated_files:
        print(f"No annotated files found for {model_name}")
        continue
    
    annotated_file = annotated_files[0]  # Take the first one if multiple exist
    
    # Load the annotated data
    annotated_data = []
    with open(annotated_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            annotated_data.append(entry)
    
    # Extract sentences and tags, tokenize each sentence
    all_sentences = []
    all_tags = []
    sentence_token_counts = []
    
    for entry in annotated_data:
        annotated_trace = entry["annotated_trace"]
        sentences = annotated_trace["sentences"]
        tags = annotated_trace["tags"]
        
        for sentence, tag in zip(sentences, tags):
            # Tokenize the sentence
            tokens = tokenizer.encode(sentence)
            token_count = len(tokens)
            
            all_sentences.append(sentence)
            all_tags.append(tag)
            sentence_token_counts.append(token_count)
    
    # Count tokens per tag and number of generations
    tag_token_counts = defaultdict(int)
    tag_generation_counts = defaultdict(int)
    for tag, token_count in zip(all_tags, sentence_token_counts):
        tag_token_counts[tag] += token_count
        tag_generation_counts[tag] += 1
    
    # Calculate average tokens per generation for each tag
    tag_avg_tokens = {}
    for tag in tag_token_counts:
        tag_avg_tokens[tag] = tag_token_counts[tag] / len(annotated_data)
    
    # Store results for this model
    all_model_results[model_name] = tag_avg_tokens

# Get all unique tags across all models
all_tags = set()
for model_results in all_model_results.values():
    all_tags.update(model_results.keys())

# Calculate total tokens across all models for each tag to determine sorting order
tag_totals = {}
for tag in all_tags:
    total = 0
    for model_results in all_model_results.values():
        total += model_results.get(tag, 0)
    tag_totals[tag] = total

# Sort tags by total tokens (largest to smallest)
all_tags = sorted(all_tags, key=lambda tag: tag_totals[tag], reverse=True)

# Create stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for stacked bar chart
models = list(all_model_results.keys())
bottom = np.zeros(len(models))

# Define colors for each tag
colors = plt.cm.Set3(np.linspace(0, 1, len(all_tags)))
tag_colors = dict(zip(all_tags, colors))

# Plot each tag as a layer in the stacked bar
# We need to reverse the order for plotting so the legend matches the visual stacking
for tag in reversed(all_tags):
    values = []
    for model in models:
        values.append(all_model_results[model].get(tag, 0))
    
    ax.bar(models, values, bottom=bottom, label=tag, color=tag_colors[tag])
    bottom += values

ax.set_title('Sentence-level Tagging of Thinking Traces (MATH 500, 0shot, temp=0)')
ax.set_ylabel('Avg. Tokens per Thought Trace')
# Reverse the legend order to match the visual stacking order
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('analysis/dist.png', dpi=300, bbox_inches='tight')
plt.show()