# === Patch DataLoader (macOS fix) ===
from torch.utils.data import DataLoader as TorchDataLoader
class PatchedDataLoader(TorchDataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['num_workers'] = 0
        super().__init__(*args, **kwargs)
import torch.utils.data
torch.utils.data.DataLoader = PatchedDataLoader


import os
import pandas as pd
from sqlalchemy import create_engine
from aerial import discretization, model, rule_extraction, rule_quality
from sklearn.metrics import classification_report, matthews_corrcoef

# ===== Config =====
DB_USER = "postgres"
DB_PASS = "password"
DB_NAME = "leakdb"
DB_HOST = "localhost"
DB_PORT = "5432"
TABLE_NAME = "leakage_wide"
INP_FILE = "Scenario-10/Hanoi_CMH_Scenario-10.inp"

FEATURE_PREFIX = "flow_"   # 可改成 "pressure_" 或 "demand_"
N_BINS = 3
EPOCHS = 6
LR = 1e-3

# 输出目录
os.makedirs("discretization", exist_ok=True)
os.makedirs("rules_learned", exist_ok=True)
os.makedirs("classification_reports", exist_ok=True)

# ===== Connect DB =====
print("📥 Connecting to TimescaleDB...")
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
df = pd.read_sql_table(TABLE_NAME, engine)

# ===== Feature selection =====
feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
print(f"🎯 Selected {len(feature_cols)} {FEATURE_PREFIX} columns.")

# ===== Parse INP =====
def parse_inp_sections(inp_path):
    sections = {}
    current = None
    with open(inp_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
                sections[current] = []
            else:
                if current:
                    sections[current].append(line)
    return sections

sections = parse_inp_sections(INP_FILE)

# PIPES: PipeID → (Node1, Node2)
pipes_map = {}
for line in sections.get("PIPES", []):
    parts = line.split()
    if len(parts) >= 3:
        pipes_map[parts[0]] = {"Node1": parts[1], "Node2": parts[2]}

# JUNCTIONS: NodeID → (Elevation, BaseDemand)
junctions_map = {}
for line in sections.get("JUNCTIONS", []):
    parts = line.split()
    if len(parts) >= 3:
        junctions_map[parts[0]] = {
            "elevation": float(parts[1]),
            "base_demand": float(parts[2])
        }

# ===== Merge semantics =====
semantic_df = pd.DataFrame(index=df.index)
semantic_added = 0

for col in feature_cols:
    if FEATURE_PREFIX == "flow_":
        link_id = col.replace("flow_", "")
        if link_id in pipes_map:
            node1 = pipes_map[link_id]["Node1"]
            if node1 in junctions_map:
                semantic_df[f"{col}_elev"] = junctions_map[node1]["elevation"]
                semantic_df[f"{col}_basedem"] = junctions_map[node1]["base_demand"]
                semantic_added += 2
    else:
        node_id = col.replace(FEATURE_PREFIX, "").replace("Node_", "")
        if node_id in junctions_map:
            semantic_df[f"{col}_elev"] = junctions_map[node_id]["elevation"]
            semantic_df[f"{col}_basedem"] = junctions_map[node_id]["base_demand"]
            semantic_added += 2


# 填充 NaN
semantic_df = semantic_df.fillna(semantic_df.mean())

# ===== Prepare data =====
X = pd.concat([df[feature_cols], semantic_df], axis=1)
y = df["leak_label"].astype(int).apply(lambda v: f"class_{v}")
data_df = pd.concat([X, y.rename("leak_label")], axis=1)

# ===== Discretization =====
print(f"🧮 Discretizing into {N_BINS} equal-frequency bins...")
disc_X = discretization.equal_frequency_discretization(X, n_bins=N_BINS)
disc_df = pd.concat([disc_X, y.rename("leak_label")], axis=1)
disc_df.to_csv(f"discretization/{FEATURE_PREFIX}semantic_discretized.csv", index=False)

# ===== Train model =====
print("🤖 Training autoencoder...")
trained_ae = model.train(disc_df, epochs=EPOCHS, lr=LR, num_workers=1)

# ===== Extract rules =====
print("🪄 Extracting rules...")
rules = rule_extraction.generate_rules(
    trained_ae,
    target_classes=[{"leak_label": "class_1"}, {"leak_label": "class_0"}],
    ant_similarity=0.4,
    cons_similarity=0.8
)

if not rules:
    raise RuntimeError("No rules generated — try lowering similarity thresholds.")

stats, rules = rule_quality.calculate_rule_stats(rules, trained_ae.input_vectors)
pd.DataFrame(rules).to_csv(f"rules_learned/{FEATURE_PREFIX}semantic_rules.csv", index=False)

# ===== Predict =====
print("🔍 Predicting on train set...")
X_features = disc_df.drop(columns=["leak_label"])
y_true = disc_df["leak_label"]

y_pred = []
for _, row in X_features.iterrows():
    matched = False
    for rule in rules:
        if all(row[cond.split("__")[0]] == cond.split("__")[1] for cond in rule['antecedents']):
            y_pred.append(rule['consequent'].split("__")[1])  # "class_0" → "0"
            matched = True
            break
    if not matched:
        y_pred.append("class_0")  # default

# 转换为数字
y_true_num = y_true.apply(lambda x: int(x.split("_")[1]))
y_pred_num = pd.Series(y_pred).apply(lambda x: int(x.split("_")[1]))

report = classification_report(y_true_num, y_pred_num, digits=3)
mcc = matthews_corrcoef(y_true_num, y_pred_num)

print(report)
print(f"MCC: {mcc:.4f}")

with open(f"classification_reports/{FEATURE_PREFIX}semantic_report.txt", "w") as f:
    f.write(report + f"\nMCC: {mcc:.4f}\n")
