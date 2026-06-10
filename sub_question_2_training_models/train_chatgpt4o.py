"""
ChatGPT-4o - Multi-label Frame Classification
==============================================
Uses the OpenAI API (gpt-4o) with few-shot prompting to classify frames.
No fine-tuning: evaluates zero-shot / few-shot performance on the test set.

Requirements:
    pip install openai pandas scikit-learn tqdm

Usage:
    export OPENAI_API_KEY="sk-..."
    python train_chatgpt4o.py
"""

import ast
import json
import os
import time

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# Configuration
DATA_PATH = "golden_standard.csv"
MODEL_NAME = "gpt-4o"
RANDOM_STATE = 42
MAX_RETRIES = 3
RETRY_DELAY = 5          
RESULTS_PATH = "./chatgpt4o_results.json"

# All 15 possible frames 
ALL_FRAMES = [
    "Capacity and Resources",
    "Crime and Punishment",
    "Cultural Identity",
    "Economic",
    "External Regulation and Reputation",
    "Fairness and Equality",
    "Health and Safety",
    "Legality / Jurisprudence",
    "Morality",
    "Policy Prescription and Evaluation",
    "Political",
    "Public Opinion",
    "Quality of Life",
    "Security and Defense",
    "Other",
]

# Load and preprocess data
def parse_gold_frames(value: str) -> list[str]:
    """Parse gold_frames column to a Python list of label strings."""
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [item.strip() for item in value.split(",")]
    return value


print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Total rows: {len(df)}")

df["gold_frames"] = df["gold_frames"].apply(parse_gold_frames)
df = df.dropna(subset=["comment_body", "gold_frames"])

# Build label encoder from ALL frames to get a consistent binarization
label_encoder = MultiLabelBinarizer(classes=ALL_FRAMES)
label_encoder.fit(df["gold_frames"])

num_labels = len(label_encoder.classes_)
print(f"Number of unique labels: {num_labels}")

# 80/20 split — we only evaluate the test set for GPT-4o
train_df, test_df = train_test_split(df, test_size=0.20, random_state=RANDOM_STATE)
print(f"Training samples (reference, not used for fine-tuning): {len(train_df)}")
print(f"Testing  samples: {len(test_df)}")

# Few-shot examples drawn from training set (one per common frame)
few_shot_pool = train_df.sample(n=min(6, len(train_df)), random_state=RANDOM_STATE)

# Prompt builder
formatted_frames = "\n".join(f"  - {f}" for f in ALL_FRAMES)

SYSTEM_PROMPT = f"""You are an expert in framing analysis of online political discussions.
You are an expert annotator for framing analysis in argumentative discourse, using the Boydstun et al. (2014) media frame taxonomy.

Your task is to classify a Reddit comment into one or more of the following {num_labels} frames:
{formatted_frames}

You will be given:
1. The original post (OP) of an r/ChangeMyView thread — context only.
2. A single comment replying to the OP — the target for annotation.

For the comment (not the OP), identify which of the frames are present based on these definitions:

Frame Definitions:
- Economic: Costs, benefits, or other financial implications.
- Capacity and Resources: Availability of physical, human or financial resources, and capacity of current systems.
- Morality: Religious or ethical implications.
- Fairness and Equality: Balance or distribution of rights, responsibilities, and resources.
- Legality / Jurisprudence: Rights, freedoms, and authority of individuals, corporations, and government (constitutionality and jurisprudence).
- Policy Prescription and Evaluation: Discussion of specific policies aimed at addressing problems.
- Crime and Punishment: Effectiveness and implications of laws and their enforcement.
- Security and Defense: Threats to welfare of the individual, community, or nation.
- Health and Safety: Health care, sanitation, public safety.
- Quality of Life: Threats and opportunities for the individual's wealth, happiness, and well-being.
- Cultural Identity: Traditions, customs, or values of a social group in relation to a policy issue.
- Public Opinion: Attitudes and opinions of the general public, including polling and demographics.
- Political: Considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters.
- External Regulation and Reputation: International reputation or foreign policy of the U.S.
- Other: Any coherent group of frames not covered by the above categories.

Rules:
- A comment can have multiple frames (multi-label). Assign all frames that apply.
- Only assign a frame if the comment substantively engages with it — not just a passing mention.
- Use EXACTLY the frame names listed in the list above (case-sensitive and matching ALL_FRAMES).
- If no frames apply, use an empty list [].
- Return ONLY a JSON object with a single key "frames" containing a list of frame names.
- Do NOT include any explanation or extra text.

Example output:
{{"frames": ["Economic", "Cultural Identity"]}}
"""


def build_few_shot_messages(comment: str) -> list[dict]:
    """Build the messages list with few-shot examples followed by the target comment."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add few-shot examples
    for _, row in few_shot_pool.iterrows():
        frames_list = row["gold_frames"]
        messages.append({"role": "user", "content": row["comment_body"]})
        messages.append({
            "role": "assistant",
            "content": json.dumps({"frames": frames_list}),
        })

    # Add target comment
    messages.append({"role": "user", "content": comment})
    return messages

# OpenAI inference
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def classify_comment(comment: str) -> list[str]:
    """Call GPT-4o to classify a single comment. Returns list of frame strings."""
    messages = build_few_shot_messages(comment)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            frames = parsed.get("frames", [])
            # Keep only valid frame names
            frames = [f for f in frames if f in ALL_FRAMES]
            return frames

        except Exception as exc:
            print(f"  [Attempt {attempt + 1}/{MAX_RETRIES}] Error: {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    return []  # fallback: empty prediction


# Run inference on test set
print("\nRunning GPT-4o inference on test set...")
predictions = []

for comment in tqdm(test_df["comment_body"].tolist(), desc="Classifying"):
    pred_frames = classify_comment(comment)
    predictions.append(pred_frames)

# Evaluation
print("\nComputing evaluation metrics...")

# Binarize gold labels and predictions
y_true = label_encoder.transform(test_df["gold_frames"].tolist())
y_pred = label_encoder.transform(predictions)

metrics = {
    "accuracy":        float(accuracy_score(y_true, y_pred)),
    "f1_micro":        float(f1_score(y_true, y_pred, average="micro",  zero_division=0)),
    "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
    "recall_micro":    float(recall_score(y_true, y_pred, average="micro",    zero_division=0)),
    "f1_macro":        float(f1_score(y_true, y_pred, average="macro",  zero_division=0)),
}

# Per-class metrics
f1_per_class        = f1_score(y_true, y_pred, average=None, zero_division=0)
precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
recall_per_class    = recall_score(y_true, y_pred, average=None, zero_division=0)
for i, label in enumerate(label_encoder.classes_):
    metrics[f"accuracy_{label}"]  = float(accuracy_score(y_true[:, i], y_pred[:, i]))
    metrics[f"f1_{label}"]        = float(f1_per_class[i])
    metrics[f"precision_{label}"] = float(precision_per_class[i])
    metrics[f"recall_{label}"]    = float(recall_per_class[i])

print("\n=== GPT-4o Evaluation Results ===")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")

# Calculate the average number of frames per comment
avg_predicted_frames = float(np.mean([len(pred) for pred in predictions]))
avg_gold_frames      = float(np.mean([len(gold) for gold in test_df["gold_frames"].tolist()]))

# Save them in the metrics dictionary so they appear in your JSON file
metrics["avg_predicted_frames_per_comment"] = avg_predicted_frames
metrics["avg_gold_frames_per_comment"]      = avg_gold_frames
print(f"\nAverage predicted frames per comment: {avg_predicted_frames:.2f}")
print(f"Average gold frames per comment: {avg_gold_frames:.2f}")

# Save results to JSON
with open(RESULTS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nResults saved to '{RESULTS_PATH}'")

# Save raw predictions alongside ground truth for inspection
pred_df = test_df[["comment_id", "comment_body", "gold_frames"]].copy()
pred_df["predicted_frames"] = predictions
pred_df.to_csv("./chatgpt4o_predictions.csv", index=False)
print("Predictions saved to './chatgpt4o_predictions.csv'")
