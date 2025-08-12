# fp_growth_cba_clean_balanced_random_default_flow.py
import os
import random
from collections import Counter
import numpy as np
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

N_BINS     = 3
MIN_SUPP   = 0.001
MIN_CONF   = 0.3
MAX_RULES  = 11000
TEST_SIZE  = 0.3


os.makedirs("classification_reports", exist_ok=True)
os.makedirs("rules_learned", exist_ok=True)

def log(msg):
    print(msg, flush=True)

# =========================
# 1) database
# =========================
log("📥 Connecting to TimescaleDB...")
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
df = pd.read_sql_table(TABLE_NAME, engine)

flow_cols = [c for c in df.columns if c.startswith("demand_")]
if not flow_cols:
    raise RuntimeError("no cols start with demand_ ")
log(f"📊 Using {len(flow_cols)} demand_ features.")

keep_cols = flow_cols + ["leak_label"]
df = df[keep_cols].dropna()

# =========================
# 2) Discretizing
# =========================
log(f"🧮 Discretizing {len(flow_cols)} features into {N_BINS} bins...")
for col in flow_cols:
    bins = pd.qcut(df[col], q=N_BINS, duplicates="drop")
    df[col] = col + "=" + bins.astype(str)

df["class"] = "class=" + df["leak_label"].astype(int).astype(str)
df = df.drop(columns=["leak_label"])

log("📊 Dataset class distribution:")
print(df["class"].value_counts())

# =========================
# 3) Train/Test split
# =========================
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, stratify=df["class"], random_state=42
)
log(f"🧪 Split: train={len(train_df)}, test={len(test_df)}")

# =========================
# 4) transaction
# =========================
def df_to_transactions(df_items: pd.DataFrame) -> list[list[str]]:
    cols = [c for c in df_items.columns if c != "class"]
    tx = []
    for _, row in df_items.iterrows():
        items = [row[c] for c in cols]
        items.append(row["class"])
        tx.append([str(v) for v in items])
    return tx

train_tx = df_to_transactions(train_df)

# =========================
# 5) FP-Growth
# =========================
abs_supp = max(1, int(MIN_SUPP * len(train_tx)))
log(f"🔍 Running FP-Growth with supp={abs_supp} ({MIN_SUPP:.3f} of {len(train_tx)}), conf={MIN_CONF:.2f}...")
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
        rules.append(
            {
                "antecedent": tuple(sorted(str(a) for a in ant)),
                "consequent": str(cons[0]),
                "support": supp / len(train_tx),
                "confidence": conf,
            }
        )
print(len(rules))
rules_df = pd.DataFrame(rules)
if rules_df.empty:
    raise RuntimeError("no class+...i right side")

rules_df["ant_len"] = rules_df["antecedent"].apply(len)

rules_df = rules_df.sort_values(by=["confidence", "support"], ascending=[False, False])
rules_df = rules_df.drop_duplicates(subset=["antecedent"], keep="first").reset_index(drop=True)
rules_df["class_priority"] = rules_df["consequent"].apply(lambda x: 1 if x == "class=1" else 0)
rules_df = rules_df.sort_values(
    by=["class_priority", "confidence", "support", "ant_len"],
    ascending=[False, False, False, True]
).reset_index(drop=True)
rules_df = rules_df.drop(columns=["class_priority"])

if len(rules_df) > MAX_RULES:
    rules_df = rules_df.iloc[:MAX_RULES].copy()

log(f"💡 Rules mined/kept: total={len(rules)}, kept={len(rules_df)}")
log("📊 Rule class distribution:")
print(rules_df["consequent"].value_counts())

rules_out = rules_df.copy()
rules_out["antecedent"] = rules_out["antecedent"].apply(lambda t: " ∧ ".join(t))
rules_out.to_csv("rules_learned/fp_rules_cba_ready.csv", index=False)

# =========================
# 6) CBA
# =========================
class SimpleCBA:
    def __init__(self, rules_df: pd.DataFrame):
        self.rules = [
            (set(ant), cons)
            for ant, cons in zip(rules_df["antecedent"], rules_df["consequent"])
        ]

    def predict_row(self, items: set[str]) -> int:
        for ant, cons in self.rules:
            if ant.issubset(items):
                return int(cons.replace("class=", ""))
        return random.choice([0, 1])

    def predict(self, df_items: pd.DataFrame) -> list[int]:
        cols = [c for c in df_items.columns if c != "class"]
        return [
            self.predict_row(set(str(row[c]) for c in cols))
            for _, row in df_items.iterrows()
        ]

clf = SimpleCBA(rules_df)

# =========================
# 7) evaluate
# =========================
y_test = test_df["class"].apply(lambda s: int(s.replace("class=", ""))).to_numpy()
y_pred = np.array(clf.predict(test_df))

log(f"📊 Predicted label distribution: {Counter(y_pred)}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=3))
print("MCC:", matthews_corrcoef(y_test, y_pred))

with open("classification_reports/fp_cba_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred, digits=3))
    f.write(f"\nMCC: {matthews_corrcoef(y_test, y_pred):.3f}\n")

log("✅ FP-Growth + CBA (random default, flow features) complete.")
