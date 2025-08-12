# === Patch DataLoader (macOS fix) ===
from torch.utils.data import DataLoader as TorchDataLoader
class PatchedDataLoader(TorchDataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['num_workers'] = 0
        super().__init__(*args, **kwargs)
import torch.utils.data
torch.utils.data.DataLoader = PatchedDataLoader

import os
import ast
import pandas as pd
from sqlalchemy import create_engine
from aerial import discretization, model, rule_extraction, rule_quality
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, matthews_corrcoef

# === Config ===
DB_USER = "postgres"
DB_PASS = "password"
DB_NAME = "leakdb"
DB_HOST = "localhost"
DB_PORT = "5432"
TABLE_NAME = "leakage_wide"
N_BINS = 3
ANT_SIM = 0.1
CONS_SIM = 0.6

os.makedirs("discretization", exist_ok=True)
os.makedirs("rules_learned", exist_ok=True)
os.makedirs("rules_stats", exist_ok=True)
os.makedirs("classification_reports", exist_ok=True)

print("📥 Connecting to TimescaleDB...")
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
df = pd.read_sql_table(TABLE_NAME, engine)


# === Select demand features ===
demand_cols = [col for col in df.columns if col.startswith("demand_Node_")]
if not demand_cols:
    raise RuntimeError("No demand_Node_ columns found!")
print(f"🎯 Selected {len(demand_cols)} demand features.")

# === Add semantic columns from JUNCTIONS ===
print("🧠 Merging [JUNCTIONS] semantics (elevation, base demand)...")
inp_file = "Scenario-10/Hanoi_CMH_Scenario-10.inp"  # <-- change to your INP file path
junctions_data = {}
section = None
with open(inp_file) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.upper()
            continue
        if section == "[JUNCTIONS]":
            parts = line.split()
            node_id = parts[0].strip()
            elevation = float(parts[1])
            base_demand = float(parts[2]) if len(parts) > 2 else 0.0
            junctions_data[node_id] = {"elevation": elevation, "base_demand": base_demand}

semantic_cols_added = 0
for col in demand_cols:
    node_id = col.replace("demand_Node_", "")
    if node_id in junctions_data:
        df[f"{col}_elev"] = junctions_data[node_id]["elevation"]
        df[f"{col}_basedem"] = junctions_data[node_id]["base_demand"]
        semantic_cols_added += 2
print(f"✅ Semantic columns added: {semantic_cols_added}")

# Drop Timestamp before training
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

# === Discretization ===
X = df.drop(columns=["leak_label"])
y = df["leak_label"]
print(f"🧮 Discretizing into {N_BINS} equal-frequency bins...")
discretized_X = discretization.equal_width_discretization(X, n_bins=N_BINS)
discretized_df = pd.concat([discretized_X, y.rename("leak_label")], axis=1)

discretized_path = "discretization/demand_semantic_discretized.csv"
discretized_df.to_csv(discretized_path, index=False)
print(f"✅ Saved discretized table to {discretized_path}")

# === Train Autoencoder + Extract Rules ===
print("🤖 Training autoencoder...")
trained_ae = model.train(discretized_df, epochs=6, lr=1e-3, num_workers=1)

print("🪄 Extracting rules...")
rules = rule_extraction.generate_rules(
    trained_ae,
    target_classes=[{"leak_label": "1.0"}, {"leak_label": "0.0"}],
    ant_similarity=ANT_SIM,
    cons_similarity=CONS_SIM
)

if not rules:
    raise RuntimeError("No rules generated — try lowering similarity thresholds.")

stats, rules = rule_quality.calculate_rule_stats(rules, trained_ae.input_vectors)
rules_path = "rules_learned/demand_semantic_pyaerial_rules.csv"
pd.DataFrame(rules).to_csv(rules_path, index=False)
pd.DataFrame([stats]).to_csv("rules_stats/demand_semantic_rule_stats.csv", index=False)
print(f"✅ Rules saved: {len(rules)} to {rules_path}")

# === Rule-based Classification ===
print("🔍 Evaluating on test set...")
train_df, test_df = train_test_split(discretized_df, test_size=0.3, stratify=discretized_df["leak_label"], random_state=42)

rules_df = pd.read_csv(rules_path)

def parse_ants(ants):
    if isinstance(ants, str):
        try:
            ants = ast.literal_eval(ants)
        except:
            ants = [ants]
    return set(str(a) for a in ants)

def predict_with_rules(df, rules_df, default_class=0):
    preds = []
    for _, row in df.iterrows():
        row_items = set(f"{k}={v}" for k, v in row.items())
        matched = False
        for _, rule in rules_df.iterrows():
            ants = parse_ants(rule["antecedents"])
            if ants.issubset(row_items):
                preds.append(int(rule["consequent"].replace("class=", "")))
                matched = True
                break
        if not matched:
            preds.append(default_class)
    return preds

default_class = int(train_df["leak_label"].mode()[0])
y_true = test_df["leak_label"].astype(int)
y_pred = predict_with_rules(test_df.drop(columns=["leak_label"]), rules_df, default_class=default_class)

report = classification_report(y_true, y_pred, digits=3)
mcc = matthews_corrcoef(y_true, y_pred)

print(report)
print(f"MCC: {mcc:.3f}")

with open("classification_reports/demand_semantic_pyaerial_report.txt", "w") as f:
    f.write(report + f"\nMCC: {mcc:.3f}\n")

print("✅ Classification report saved.")
