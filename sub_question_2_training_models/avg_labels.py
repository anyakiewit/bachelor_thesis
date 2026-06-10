"""Average frames predicted per comment (vs gold), derived from per-frame CSVs + the test split."""
import ast, os, csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


def parse(v):
    if isinstance(v, str):
        try:
            p = ast.literal_eval(v)
            if isinstance(p, list):
                return p
        except (ValueError, SyntaxError):
            pass
        return [x.strip() for x in v.split(",")]
    return v


df = pd.read_csv("golden_standard.csv")
df["gold_frames"] = df["gold_frames"].apply(parse)
df = df.dropna(subset=["comment_body", "gold_frames"])
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["gold_frames"])
df["labels"] = list(Y.astype(float))
classes = list(mlb.classes_)
idx = {c: i for i, c in enumerate(classes)}

# reproduce the exact 80/20 test split the training scripts use
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
test_Y = np.array(test_df["labels"].tolist())
N = len(test_Y)
gold_pos = test_Y.sum(axis=0)
gold_avg = test_Y.sum() / N

print(f"Test comments: {N}")
print(f"GOLD (human) average frames/comment: {gold_avg:.2f}\n")
print(f"{'Model':12s} {'avg frames/comment predicted':>30s}")
print("-" * 44)
for m in ["bert", "modernbert", "qwen08b", "qwen2b"]:
    f = f"results/{m}_per_frame.csv"
    if not os.path.exists(f):
        print(f"{m:12s} {'(not ready yet)':>30s}")
        continue
    pred_total = 0.0
    for r in csv.DictReader(open(f)):
        fr = r["Frame"]
        if fr not in idx:
            continue
        i = idx[fr]
        acc = float(r["Accuracy"]); rec = float(r["Recall"]); gp = gold_pos[i]
        # acc*N = TP + TN ; TP = rec*gp ; TN = N - gp - FP ; FP = pred_pos - TP
        # => pred_pos = 2*TP + N - gp - acc*N
        pred_pos = 2 * rec * gp + N - gp - acc * N
        pred_total += max(0.0, pred_pos)
    print(f"{m:12s} {pred_total / N:>30.2f}")
