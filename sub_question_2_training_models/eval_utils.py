"""Shared post-training evaluation: tune threshold on validation, report on test,
and save raw probabilities so thresholds can be re-tuned later."""
import csv
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _predict_probs(trainer, ds):
    pred = trainer.predict(ds)
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs, pred.label_ids.astype(int)


def report(trainer, tokenized, classes, prefix, thresholds=None):
    classes = list(classes)
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(2, 15)]  # 0.10 .. 0.70

    # tune decision threshold on VALIDATION (max micro-F1) 
    val_probs, val_labels = _predict_probs(trainer, tokenized["validation"])
    print("\n=== Validation threshold sweep (micro-averaged) ===")
    print(f"{'thr':>5} {'F1':>8} {'Prec':>8} {'Recall':>8} {'avg_lab':>8}")
    best_t, best_f1 = 0.50, -1.0
    for t in thresholds:
        pr = (val_probs >= t).astype(int)
        f1 = f1_score(val_labels, pr, average="micro", zero_division=0)
        pp = precision_score(val_labels, pr, average="micro", zero_division=0)
        rr = recall_score(val_labels, pr, average="micro", zero_division=0)
        print(f"{t:>5.2f} {f1:>8.4f} {pp:>8.4f} {rr:>8.4f} {pr.sum() / len(pr):>8.2f}")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"BEST validation threshold = {best_t:.2f}  (val micro-F1 {best_f1:.4f})")

    # apply tuned threshold to TEST 
    test_probs, test_labels = _predict_probs(trainer, tokenized["test"])

    # save raw probabilities so future threshold tuning needs no retraining
    np.savez(
        f"{prefix}_probs.npz",
        val_probs=val_probs, val_labels=val_labels,
        test_probs=test_probs, test_labels=test_labels,
        threshold=np.array(best_t), classes=np.array(classes, dtype=object),
    )
    print(f"Saved raw probabilities to '{prefix}_probs.npz'")

    preds = (test_probs >= best_t).astype(int)
    p = precision_score(test_labels, preds, average=None, zero_division=0)
    r = recall_score(test_labels, preds, average=None, zero_division=0)
    f = f1_score(test_labels, preds, average=None, zero_division=0)
    a = [(preds[:, i] == test_labels[:, i]).mean() for i in range(test_labels.shape[1])]
    tot_acc = accuracy_score(test_labels, preds)
    tot_p = precision_score(test_labels, preds, average="micro", zero_division=0)
    tot_r = recall_score(test_labels, preds, average="micro", zero_division=0)
    tot_f = f1_score(test_labels, preds, average="micro", zero_division=0)
    avg_lab = preds.sum() / len(preds)

    header = f"{'Frame':45s} {'Accuracy':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}"
    lines = ["=" * 84, header, "-" * 84]
    for i, frame in enumerate(classes):
        lines.append(f"{frame:45s} {a[i]:8.3f} {p[i]:10.4f} {r[i]:8.4f} {f[i]:8.4f}")
    lines.append("-" * 84)
    lines.append(f"{'Total for all frames':45s} {tot_acc:8.3f} {tot_p:10.4f} {tot_r:8.4f} {tot_f:8.4f}")
    lines.append("=" * 84)
    print(f"\n=== TEST per-frame results (threshold {best_t:.2f}, tuned on validation) ===")
    print("\n".join(lines))
    print(f"Avg frames/comment predicted: {avg_lab:.2f}   (gold/human: 1.96)")

    with open(f"{prefix}_per_frame.csv", "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["Frame", "Accuracy", "Precision", "Recall", "F1"])
        for i, frame in enumerate(classes):
            w.writerow([frame, f"{a[i]:.3f}", f"{p[i]:.4f}", f"{r[i]:.4f}", f"{f[i]:.4f}"])
        w.writerow(["Total for all frames", f"{tot_acc:.3f}", f"{tot_p:.4f}", f"{tot_r:.4f}", f"{tot_f:.4f}"])

    with open(f"{prefix}_table.txt", "w") as tf:
        tf.write(f"Model: {prefix}   (threshold {best_t:.2f}, tuned on validation; seed 42)\n\n")
        tf.write("\n".join(lines) + "\n\n")
        tf.write(f"Avg frames/comment predicted: {avg_lab:.2f}   (gold/human: 1.96)\n")
    print(f"Wrote {prefix}_table.txt and {prefix}_per_frame.csv")
