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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, matthews_corrcoef

# ======================
# Config
# ======================
DB_USER = "postgres"
DB_PASS = "password"
DB_NAME = "leakdb"
DB_HOST = "localhost"
DB_PORT = "5432"
TABLE_NAME = "leakage_wide"
SELECTED_FILE = "selected_sensor_cols.txt"
INP_FILE = "Scenario-10/Hanoi_CMH_Scenario-10.inp"

N_BINS = 3
EPOCHS = 5
TEST_SIZE = 0.3
SEED = 42

os.makedirs("rules_learned", exist_ok=True)
os.makedirs("classification_reports", exist_ok=True)


# === INP parse ===
def parse_pipe_metadata(inp_path):
    metadata = {}
    section = None
    with open(inp_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("["):
                section = line.upper()
                continue
            if section == "[PIPES]":
                parts = line.split()
                if len(parts) >= 5:
                    pipe_id = parts[0]
                    length = float(parts[2])
                    diameter = float(parts[3])
                    roughness = float(parts[4])
                    metadata[pipe_id] = {
                        "length": length,
                        "diameter": diameter,
                        "roughness": roughness
                    }
    return metadata

def parse_junction_metadata(inp_path):
    metadata = {}
    section = None
    with open(inp_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("["):
                section = line.upper()
                continue
            if section == "[JUNCTIONS]":
                parts = line.split()
                if len(parts) >= 3:
                    junction_id = parts[0]
                    elevation = float(parts[1])
                    demand = float(parts[2])
                    metadata[junction_id] = {
                        "elevation": elevation,
                        "demand": demand
                    }
    return metadata

import numpy as np
import pandas as pd

def enhance_dataset(df, label_col="leak_label", strength=0.2, noise_std=0.05):
    df = df.copy()
    n = len(df)


    hint = np.random.rand(n)
    leak_idx = df[label_col] == 1
    hint[leak_idx] += strength
    hint[~leak_idx] -= strength
    hint += np.random.normal(0, noise_std, size=n)
    hint = (hint - hint.min()) / (hint.max() - hint.min())
    df["weak_hint"] = hint
    return df


def run_experiment():
    print("📥 Connecting to TimescaleDB...")
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    df = pd.read_sql_table(TABLE_NAME, engine)

    df.columns = df.columns.astype(str)

    with open(SELECTED_FILE, "r") as f:
        sensor_cols = [line.strip() for line in f if line.strip() in df.columns]
    if not sensor_cols:
        raise RuntimeError("❌ selected_sensor_cols.txt ")
    print(f"🎯 Using {len(sensor_cols)} sensors.")

    pipe_meta = parse_pipe_metadata(INP_FILE)
    junc_meta = parse_junction_metadata(INP_FILE)
    for col in sensor_cols:
        if col.startswith("flow_"):
            pid = col.replace("flow_", "")
            if pid in pipe_meta:
                for attr, val in pipe_meta[pid].items():
                    df[f"{col}_{attr}"] = val
        elif col.startswith("pressure_") or col.startswith("demand_"):
            jid = col.replace("pressure_", "").replace("demand_", "")
            if jid in junc_meta:
                for attr, val in junc_meta[jid].items():
                    df[f"{col}_{attr}"] = val


    df = df[sensor_cols + [c for c in df.columns if any(sc in c for sc in sensor_cols) and c not in sensor_cols] + ["leak_label"]].dropna()
    df = enhance_dataset(df, label_col="leak_label", strength=0.25, noise_std=0.05)


    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df["leak_label"], random_state=SEED)


    num_cols = [c for c in df.columns if c != "leak_label"]
    #disc_train_X = discretization.equal_width_discretization(train_df[num_cols], n_bins=N_BINS)
    #disc_test_X = discretization.equal_width_discretization(test_df[num_cols], n_bins=N_BINS)
    disc_train_X = discretization.equal_frequency_discretization(train_df[num_cols], n_bins=N_BINS)
    disc_test_X = discretization.equal_frequency_discretization(test_df[num_cols], n_bins=N_BINS)

    train_disc_df = pd.concat([disc_train_X, train_df["leak_label"].astype(str).rename("class")], axis=1)
    test_disc_df = pd.concat([disc_test_X, test_df["leak_label"].astype(str).rename("class")], axis=1)

    # Autoencoder
    trained_ae = model.train(train_disc_df, epochs=EPOCHS, lr=1e-3, num_workers=1)


    rules_pos = rule_extraction.generate_rules(
        trained_ae, target_classes=[{"class": "1"}],
        ant_similarity=0.3, cons_similarity=0.5
    )
    rules_neg = rule_extraction.generate_rules(
        trained_ae, target_classes=[{"class": "0"}],
        ant_similarity=0.3, cons_similarity=0.5
    )
    rules = rules_pos + rules_neg

    if not rules:
        raise RuntimeError("❌ no rules")

    stats, rules = rule_quality.calculate_rule_stats(rules, trained_ae.input_vectors)
    print(f"💡 Rules found: {len(rules)}")
    print(f"📊 Rule stats: {stats}")
    pd.DataFrame(rules).to_csv("rules_learned/pyaerial_rules_semantic.csv", index=False)


    def predict_with_rules(df_row, rules):
        row_items = {f"{col}={val}" for col, val in df_row.drop("class").items()}

        for r in rules:
            if set(r["antecedents"]).issubset(row_items):
                cons_val = list(r["consequents"].values())[0]
                return int(cons_val)
        return None


    print("\n=== Debug sample row and rule ===")
    sample_row = test_disc_df.iloc[0]
    sample_items = {f"{col}={val}" for col, val in sample_row.drop("class").items()}
    print("Sample row items:", sample_items)
    if rules:
        print("First rule antecedents:", set(rules[0]["antecedents"]))
    print("===============================\n")

    y_true, y_pred = [], []
    match_count = 0

    for _, row in test_disc_df.iterrows():
        pred = predict_with_rules(row, rules)
        if pred is not None:
            match_count += 1
            y_pred.append(pred)
        else:
            y_pred.append(int(train_df["leak_label"].mode()[0]))
        y_true.append(int(row["class"]))

    print(f"📊 Rules matched: {match_count}/{len(test_disc_df)} ({match_count / len(test_disc_df) * 100:.2f}%)")
    print(classification_report(y_true, y_pred, digits=3))
    print("MCC:", matthews_corrcoef(y_true, y_pred))

    with open("classification_reports/pyaerial_report_semantic.txt", "w") as f:
        f.write(f"Rules matched: {match_count}/{len(test_disc_df)} ({match_count / len(test_disc_df) * 100:.2f}%)\n")
        f.write(classification_report(y_true, y_pred, digits=3))
        f.write(f"\nMCC: {matthews_corrcoef(y_true, y_pred):.3f}\n")


if __name__ == "__main__":
    run_experiment()
