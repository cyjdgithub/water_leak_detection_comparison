# generate_rules_scenario6.py
# === Patch DataLoader (macOS fix) ===
from torch.utils.data import DataLoader as TorchDataLoader
class PatchedDataLoader(TorchDataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['num_workers'] = 0
        super().__init__(*args, **kwargs)
import torch.utils.data
torch.utils.data.DataLoader = PatchedDataLoader

import pandas as pd
from sqlalchemy import create_engine
from aerial import discretization, model, rule_extraction, rule_quality

# === DB setup ===
DB_USER = "postgres"
DB_PASS = "password"
DB_NAME = "leakdb"
DB_HOST = "localhost"
DB_PORT = "5432"
TABLE_NAME = "leakage_wide"

print("📥 Connecting to TimescaleDB...")
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
df = pd.read_sql_table(TABLE_NAME, engine)


# === Select demand features only ===
print("🎯 Selecting demand_Node_xx columns...")
demand_cols = [col for col in df.columns if col.startswith("demand_Node_")]
print(f"Selected demand columns: {demand_cols}")

# === Build training table ===
X = df[demand_cols]
y = df["leak_label"]

# === Discretization ===
print("🧮 Discretizing features...")
discretized_X = discretization.equal_width_discretization(X, n_bins=5)
discretized_df = pd.concat([discretized_X, y.rename("leak_label")], axis=1)

discretized_df.to_csv("discretization/scenario8_discretized.csv", index=False)
print("✅ Discretized data saved to 'scenario8_discretized.csv'.")

# === Train autoencoder & extract rules ===
print("🤖 Training autoencoder and extracting rules...")
trained_ae = model.train(discretized_df, epochs=6, lr=1e-3, num_workers=1)

rules = rule_extraction.generate_rules(
    trained_ae,
    target_classes=[{"leak_label": "1.0"}, {"leak_label": "0.0"}],
    ant_similarity=0.4,
    cons_similarity=0.8
)

if rules:
    stats, rules = rule_quality.calculate_rule_stats(rules, trained_ae.input_vectors)
    pd.DataFrame(rules).to_csv("rules_learned/scenario8_learned_rules.csv", index=False)
    pd.DataFrame([stats]).to_csv("rules_stats/scenario8_rule_stats.csv", index=False)
    print(f"✅ {len(rules)} rules extracted and saved.")
else:
    print("⚠️ No rules found.")
