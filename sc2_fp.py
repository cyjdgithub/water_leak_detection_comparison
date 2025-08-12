# scenario2_fpgrowth_cba.py
import os
import random
from collections import Counter
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, matthews_corrcoef
from fim import arules

# =========================
# Config
# =========================
DB_USER = "postgres"
DB_PASS = "password"
DB_NAME = "leakdb"
DB_HOST = "localhost"
DB_PORT = "5432"
TABLE_NAME = "leakage_wide"
INP_FILE = "Scenario-10/Hanoi_CMH_Scenario-10.inp"

N_BINS     = 3
MIN_SUPP   = 0.001
MIN_CONF   = 0.3
MAX_RULES  = 500
TEST_SIZE  = 0.2
SEED       = 42

os.makedirs("classification_reports", exist_ok=True)
os.makedirs("rules_learned", exist_ok=True)

def log(msg):
    print(msg, flush=True)

# =========================
# INP parse
# =========================
def parse_pipe_metadata(inp_path):
    meta = {}
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
                    pid = parts[0]
                    length = float(parts[2])
                    diameter = float(parts[3])
                    roughness = float(parts[4])
                    meta[pid] = {
                        "length": length,
                        "diameter": diameter,
                        "roughness": roughness
                    }
    return meta

def parse_junction_metadata(inp_path):
    meta = {}
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
                    jid = parts[0]
                    elevation = float(parts[1])
                    demand = float(parts[2])
                    meta[jid] = {
                        "elevation": elevation,
                        "demand": demand
                    }
    return meta

# =========================
# load data
# =========================
def load_dataset():
    log("📥 Connecting to TimescaleDB...")
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    df = pd.read_sql_table(TABLE_NAME, engine)
    with open("selected_sensor_cols.txt") as f:
        selected_sensor_cols = [line.strip() for line in f.readlines()]
    sensor_cols = [c for c in selected_sensor_cols if c.startswith("flow_")]
    df = df[sensor_cols + ["leak_label"]].dropna()
    pipe_meta = parse_pipe_metadata(INP_FILE)
    for col in sensor_cols:
        pid = col.replace("flow_", "")
        if pid in pipe_meta:
            for attr, val in pipe_meta[pid].items():
                df[f"{col}_{attr}"] = val

    return df, sensor_cols

# =========================
# discretize
# =========================
def discretize_dataframe(df, sensor_cols):
    for col in df.columns:
        if col in ["leak_label"]:
            continue
        bins = pd.qcut(df[col], q=N_BINS, duplicates="drop")
        df[col] = col + "=" + bins.astype(str)
    df["class"] = "class=" + df["leak_label"].astype(int).astype(str)
    return df.drop(columns=["leak_label"])

# =========================
# DataFrame to transaction
# =========================
def df_to_transactions(df_items: pd.DataFrame):
    cols = [c for c in df_items.columns if c != "class"]
    return [[row[c] for c in cols] + [row["class"]] for _, row in df_items.iterrows()]

# =========================
# FP-Growth
# =========================
def mine_rules(train_tx):
    abs_supp = max(1, int(MIN_SUPP * len(train_tx)))
    log(f"🔍 Running FP-Growth with supp={abs_supp}, conf={MIN_CONF}...")
    rules_raw = arules(
        train_tx,
        supp=abs_supp,
        conf=MIN_CONF,
        report="sc",
        mode="o",
        zmax=3
    )
    rules = []
    for ant, cons, supp, conf in rules_raw:
        if isinstance(cons, tuple) and len(cons) == 1 and str(cons[0]).startswith("class="):
            if isinstance(ant, str):
                ant = [ant]
            ant = list(ant)
            rules.append({
                "antecedent": tuple(str(a) for a in ant),
                "consequent": str(cons[0]),
                "support": supp / len(train_tx),
                "confidence": conf
            })
    rules_df = pd.DataFrame(rules)
    rules_df = rules_df.sort_values(by=["confidence", "support"], ascending=[False, False])
    rules_df = rules_df.drop_duplicates(subset=["antecedent"], keep="first").reset_index(drop=True)
    rules_df["class_priority"] = rules_df["consequent"].apply(lambda x: 1 if x == "class=1" else 0)
    rules_df = rules_df.sort_values(
        by=["class_priority", "confidence", "support"],
        ascending=[False, False, False]
    ).reset_index(drop=True).drop(columns=["class_priority"])
    if len(rules_df) > MAX_RULES:
        rules_df = rules_df.iloc[:MAX_RULES].copy()
    return rules_df

# =========================
# CBA
# =========================
class SimpleCBA:
    def __init__(self, rules_df):
        self.rules = [(set(ant), cons) for ant, cons in zip(rules_df["antecedent"], rules_df["consequent"])]

    def predict_row(self, items: set[str], default_class="0"):
        for ant, cons in self.rules:
            if ant.issubset(items):
                return int(cons.replace("class=", ""))
        return int(default_class)

    def predict(self, df_items: pd.DataFrame, default_class="0"):
        cols = [c for c in df_items.columns if c != "class"]
        return [self.predict_row(set(row[c] for c in cols), default_class) for _, row in df_items.iterrows()]

# =========================
# Main
# =========================
if __name__ == "__main__":
    df, sensor_cols = load_dataset()
    df_disc = discretize_dataframe(df.copy(), sensor_cols)

    train_df, test_df = train_test_split(df_disc, test_size=TEST_SIZE, stratify=df_disc["class"], random_state=SEED)
    train_tx = df_to_transactions(train_df)

    rules_df = mine_rules(train_tx)
    rules_df["readable"] = rules_df["antecedent"].apply(lambda t: " ∧ ".join(t)) + " → " + rules_df["consequent"]
    rules_df.to_csv("rules_learned/scenario2_fpgrowth_rules.csv", index=False)

    clf = SimpleCBA(rules_df)
    default_class = train_df["class"].value_counts().idxmax().replace("class=", "")
    log(f"⚙️ Using default class = {default_class} when no rule matches.")

    test_X = test_df.drop(columns=["class"])
    test_y = test_df["class"].apply(lambda x: int(x.replace("class=", "")))
    y_pred = clf.predict(test_X, default_class=default_class)

    report = classification_report(test_y, y_pred, digits=3)
    mcc = matthews_corrcoef(test_y, y_pred)

    with open("classification_reports/scenario2_classification_report.txt", "w") as f:
        f.write(report + f"\nMCC: {mcc:.3f}\n")

    log("\n📊 Classification Report:")
    print(report)
    print("MCC:", mcc)
    log(f"✅ Scenario 2 FP-Growth + CBA complete. Rules kept: {len(rules_df)}")
