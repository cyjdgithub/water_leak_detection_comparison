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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, matthews_corrcoef
from aerial import discretization, model, rule_extraction, rule_quality

# =======================
# 配置
# =======================
DB_USER = "postgres"
DB_PASS = "password"
DB_NAME = "leakdb"
DB_HOST = "localhost"
DB_PORT = "5432"
TABLE_NAME = "leakage_wide"
SELECTED_FILE = "selected_sensor_cols.txt"

N_BINS = 5
EPOCHS = 5
TEST_SIZE = 0.3
SEED = 42

os.makedirs("rules_learned", exist_ok=True)
os.makedirs("classification_reports", exist_ok=True)

# =======================
# 主流程
# =======================
def run_experiment():
    print("📥 Connecting to TimescaleDB...")
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    df = pd.read_sql_table(TABLE_NAME, engine)

    # 读取传感器列
    with open(SELECTED_FILE, "r") as f:
        sensor_cols = [line.strip() for line in f if line.strip() in df.columns]
    if not sensor_cols:
        raise RuntimeError("❌ selected_sensor_cols.txt 中没有有效列")
    print(f"🎯 Using {len(sensor_cols)} features.")

    # 保留必要列
    df = df[sensor_cols + ["leak_label"]].dropna()

    # 划分训练 / 测试集
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df["leak_label"], random_state=SEED)

    # 离散化（保持分箱一致）
    disc_train_X = discretization.equal_width_discretization(train_df[sensor_cols], n_bins=N_BINS)
    disc_test_X = discretization.equal_width_discretization(test_df[sensor_cols], n_bins=N_BINS)
    train_disc_df = pd.concat([disc_train_X, train_df["leak_label"].astype(str).rename("class")], axis=1)
    test_disc_df = pd.concat([disc_test_X, test_df["leak_label"].astype(str).rename("class")], axis=1)

    # 训练 Autoencoder
    trained_ae = model.train(train_disc_df, epochs=EPOCHS, lr=1e-3, num_workers=1)

    # 挖掘规则（正负类分开）
    rules_pos = rule_extraction.generate_rules(
        trained_ae, target_classes=[{"class": "1"}],
        ant_similarity=0.5, cons_similarity=0.7
    )
    rules_neg = rule_extraction.generate_rules(
        trained_ae, target_classes=[{"class": "0"}],
        ant_similarity=0.5, cons_similarity=0.7
    )
    rules = rules_pos + rules_neg

    if not rules:
        raise RuntimeError("❌ 没有挖到规则")

    # 计算规则质量
    stats, rules = rule_quality.calculate_rule_stats(rules, trained_ae.input_vectors)
    print(f"💡 Rules found: {len(rules)}")
    print(f"📊 Rule stats: {stats}")

    # 保存规则
    pd.DataFrame(rules).to_csv("rules_learned/pyaerial_rules.csv", index=False)

    # 规则匹配预测
    def predict_with_rules(df_row, rules):
        row_items = set(df_row.drop("class"))
        for r in rules:
            if set(r["antecedents"]).issubset(row_items):
                cons_val = list(r["consequents"].values())[0]
                return int(cons_val)
        return None

    y_true, y_pred = [], []
    match_count = 0
    for _, row in test_disc_df.iterrows():
        pred = predict_with_rules(row, rules)
        if pred is not None:
            match_count += 1
            y_pred.append(pred)
        else:
            # 无匹配则默认预测为多数类
            y_pred.append(int(train_df["leak_label"].mode()[0]))
        y_true.append(int(row["class"]))

    print(f"📊 Rules matched: {match_count}/{len(test_disc_df)} ({match_count/len(test_disc_df)*100:.2f}%)")
    print(classification_report(y_true, y_pred, digits=3))
    print("MCC:", matthews_corrcoef(y_true, y_pred))

    # 保存报告
    with open("classification_reports/pyaerial_report.txt", "w") as f:
        f.write(f"Rules matched: {match_count}/{len(test_disc_df)} ({match_count/len(test_disc_df)*100:.2f}%)\n")
        f.write(classification_report(y_true, y_pred, digits=3))
        f.write(f"\nMCC: {matthews_corrcoef(y_true, y_pred):.3f}\n")

    print("✅ Done.")

if __name__ == "__main__":
    run_experiment()
