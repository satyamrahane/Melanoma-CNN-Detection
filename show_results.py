import json, os

m = json.load(open("outputs/metrics.json"))
f = json.load(open("outputs/fairness_metrics.json"))

print("=" * 60)
print("  MelanomaAI - Full Model Evaluation Results")
print("  Threshold: %.2f" % m["threshold"])
print("=" * 60)
print()
print("  Model:         %s" % m["model_path"])
print("  Evaluated:     %s" % m["evaluated_at"])
print("  Total Samples: %d" % m["total_samples"])
print()
print("  --- CORE METRICS ---")
print("  Accuracy:      %.2f%%" % (m["accuracy"] * 100))
print("  AUC-ROC:       %.4f" % m["auc_roc"])
print("  Sensitivity:   %.2f%%" % (m["sensitivity"] * 100))
print("  Specificity:   %.2f%%" % (m["specificity"] * 100))
print("  F1-Score:      %.4f" % m["f1_score"])
print("  Precision:     %.2f%%" % (m["precision"] * 100))
print("  Avg Precision: %.4f" % m["average_precision"])
print()
print("  --- CONFUSION MATRIX ---")
print("  TN=%d  FP=%d" % (m["true_negatives"], m["false_positives"]))
print("  FN=%d  TP=%d" % (m["false_negatives"], m["true_positives"]))
print()
print("  --- RISK STRATIFICATION ---")
ra = m["risk_analysis"]
lc = ra["level_counts"]
print("  CRITICAL:  %d" % lc["CRITICAL"])
print("  HIGH:      %d" % lc["HIGH"])
print("  MODERATE:  %d" % lc["MODERATE"])
print("  LOW:       %d" % lc["LOW"])
print("  Mean Score:  %.2f" % ra["mean_score"])
print("  FP Caught:   %d (%.1f%%)" % (ra["fp_caught"], ra["fp_catch_rate"] * 100))
print()
print("  --- FAIRNESS METRICS ---")
print("  Light Skin N:    %d" % f["light_skin_n"])
print("  Dark Skin N:     %d" % f["dark_skin_n"])
print("  Light Mean Prob: %.4f" % f["light_mean_prob"])
print("  Dark Mean Prob:  %.4f" % f["dark_mean_prob"])
print("  Probability Gap: %.4f" % f["prob_gap"])
print()
print("  --- THRESHOLD ANALYSIS ---")
for row in m.get("threshold_analysis", []):
    marker = " <-- SELECTED" if abs(row["t"] - m["threshold"]) < 0.01 else ""
    f1_val = row["f1"]
    print("  t=%.2f | Acc=%.2f%% | Sens=%.2f%% | Spec=%.2f%% | F1=%.4f%s" % (
        row["t"], row["acc"]*100, row["sens"]*100, row["spec"]*100, f1_val, marker))
print()
print("  --- GRAPHS GENERATED ---")
for g in sorted(os.listdir("outputs/graphs")):
    size_kb = os.path.getsize(os.path.join("outputs/graphs", g)) / 1024
    print("    outputs/graphs/%s (%.1f KB)" % (g, size_kb))
print()
print("=" * 60)
