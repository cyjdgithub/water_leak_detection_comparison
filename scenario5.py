# === pressure_timeframe_rule_learning.py ===
# === flow_timeframe_rule_learning.py ===

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

# === Select pressure features only ===
print("🎯 Selecting pressure_Node_xx columns...")
# === Select pressure features only ===
pressure_cols = [col for col in df.columns if col.startswith("pressure_Node_")]

print(f"Selected pressure columns: {pressure_cols}")

# === Build training table ===
X = df[pressure_cols]
y = df["leak_label"]

# === Discretization ===
print("🧮 Discretizing features...")
discretized_X = discretization.equal_width_discretization(X, n_bins=6)
discretized_df = pd.concat([discretized_X, y.rename("leak_label")], axis=1)

discretized_df.to_csv("discretization/scenario7_discretized.csv", index=False)
print("✅ Discretized data saved to 'scenario7_discretized.csv'.")

# === Train autoencoder & extract rules ===
print("🤖 Training autoencoder and extracting rules...")
trained_ae = model.train(discretized_df, epochs=5, lr=1e-3, num_workers=1)

"""rules = rule_extraction.generate_rules(
    trained_ae,
    target_classes=[{"leak_label": "1.0"}, {"leak_label": "0.0"}],
    ant_similarity=0.4,
    cons_similarity=0.8
)"""
# 学 leak=1 的规则
rules_pos = rule_extraction.generate_rules(
    trained_ae,
    target_classes=[{"leak_label": "1.0"}],
    ant_similarity=0.7,
    cons_similarity=0.9
)

# 学 leak=0 的规则
rules_neg = rule_extraction.generate_rules(
    trained_ae,
    target_classes=[{"leak_label": "0.0"}],
    ant_similarity=0.5,
    cons_similarity=0.7
)

# 合并 rules
rules = rules_pos + rules_neg


if rules:
    stats, rules = rule_quality.calculate_rule_stats(rules, trained_ae.input_vectors)
    pd.DataFrame(rules).to_csv("rules_learned/scenario7_learned_rules.csv", index=False)
    pd.DataFrame([stats]).to_csv("rules_stats/scenario7_rule_stats.csv", index=False)
    print(f"✅ {len(rules)} rules extracted and saved.")
else:
    print("⚠️ No rules found.")


# === Step 7: Rule-based classification report ===
print("\n🔍 Applying rules to classify samples...")

import pandas as pd
from rule_based_classifier import RuleBasedClassifier
from sklearn.metrics import classification_report

# 加载规则
rules_df = pd.read_csv("rules_learned/scenario7_learned_rules.csv")   # 改成 7/8 对应文件名

# 准备输入数据
X = discretized_X  # 已离散化的输入特征（DataFrame）
y = df["leak_label"]  # 原始标签

# 规则分类器预测
clf = RuleBasedClassifier()
clf.fit(rules_df)

y_pred = clf.predict(X)

# 评估
report = classification_report(y, y_pred, digits=3)
print(report)

# 保存分类报告
with open("classfication_reports/scenario7_classification_report.txt", "w") as f:
    f.write(report)

print("✅ Classification report saved.")