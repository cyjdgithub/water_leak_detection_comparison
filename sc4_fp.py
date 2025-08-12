# fp_growth_cba_flow_with_semantic.py
import os, random
from collections import Counter
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, matthews_corrcoef
from fim import arules

# ========= Config =========
DB_USER="postgres"; DB_PASS="password"; DB_NAME="leakdb"
DB_HOST="localhost"; DB_PORT="5432"; TABLE_NAME="leakage_wide"
INP_FILE="Scenario-10/Hanoi_CMH_Scenario-10.inp"

N_BINS=3; MIN_SUPP=0.001; MIN_CONF=0.3; MAX_RULES=11000; TEST_SIZE=0.3; SEED=42

os.makedirs("classification_reports", exist_ok=True)
os.makedirs("rules_learned", exist_ok=True)

def log(m): print(m, flush=True)

# ========= INP parse: [PIPES] =========
def parse_pipe_metadata(inp_path):
    meta, section = {}, None
    with open(inp_path,"r") as f:
        for line in f:
            s=line.strip()
            if not s or s.startswith(";"): continue
            if s.startswith("["): section=s.upper(); continue
            if section=="[PIPES]":
                parts=s.split()
                if len(parts)>=5:
                    pid=parts[0]; length=float(parts[2]); diameter=float(parts[3]); roughness=float(parts[4])
                    meta[pid]={"length":length,"diameter":diameter,"roughness":roughness}
    return meta

# ========= Load: flow + semantics =========
log("📥 Connecting to TimescaleDB...")
engine=create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
df=pd.read_sql_table(TABLE_NAME, engine)

flow_cols=[c for c in df.columns if c.startswith("flow_")]
if not flow_cols: raise RuntimeError("未找到 flow_ 列")
df=df[flow_cols+["leak_label"]].dropna()
log(f"📊 Using {len(flow_cols)} flow features.")

# merge [PIPES] diameter/length/roughness
log("🧠 Merging [PIPES] semantics...")
pmeta=parse_pipe_metadata(INP_FILE); added=0
for col in flow_cols:
    pid=col.replace("flow_","")
    if pid in pmeta:
        for k,v in pmeta[pid].items():
            df[f"{col}_{k}"]=v; added+=1
log(f"✅ Semantic columns added: {added}")

# ========= Discretization =========
item_cols=[c for c in df.columns if c!="leak_label"]
log(f"🧮 Discretizing {len(item_cols)} columns into {N_BINS} bins...")
for col in item_cols:
    bins=pd.qcut(df[col], q=N_BINS, duplicates="drop")
    df[col]=col+"="+bins.astype(str)

df["class"]="class="+df["leak_label"].astype(int).astype(str)
df=df.drop(columns=["leak_label"])
log("📊 Class dist:"); print(df["class"].value_counts())

# ========= Split =========
train_df, test_df=train_test_split(df, test_size=TEST_SIZE, stratify=df["class"], random_state=SEED)
log(f"🧪 Split: train={len(train_df)}, test={len(test_df)}")

# ========= To transactions =========
def df_to_tx(dfi):
    cols=[c for c in dfi.columns if c!="class"]
    return [[*(str(dfi.loc[i,c]) for c in cols), dfi.loc[i,"class"]] for i in dfi.index]
train_tx=df_to_tx(train_df)

# ========= FP-Growth =========
abs_supp=max(1, int(MIN_SUPP*len(train_tx)))
log(f"🔍 FP-Growth supp={abs_supp} ({MIN_SUPP:.3f}), conf={MIN_CONF:.2f}...")
rules_raw=arules(train_tx, supp=abs_supp, conf=MIN_CONF, report="sc", mode="o", zmax=3)

rules=[]
for ant,cons,supp,conf in rules_raw:
    if isinstance(cons,tuple) and len(cons)==1 and str(cons[0]).startswith("class="):
        if isinstance(ant,str): ant=[ant]
        ant=list(ant)
        rules.append({"antecedent":tuple(sorted(str(a) for a in ant)),
                      "consequent":str(cons[0]),
                      "support":supp/len(train_tx),
                      "confidence":conf})

rules_df=pd.DataFrame(rules)
if rules_df.empty: raise RuntimeError("no rule with class on the right side。")
rules_df["ant_len"]=rules_df["antecedent"].apply(len)
rules_df=rules_df.sort_values(["confidence","support"],ascending=[False,False]).drop_duplicates("antecedent")
rules_df["prio"]=rules_df["consequent"].eq("class=1").astype(int)
rules_df=rules_df.sort_values(["prio","confidence","support","ant_len"],ascending=[False,False,False,True]).drop(columns="prio")
if len(rules_df)>MAX_RULES: rules_df=rules_df.iloc[:MAX_RULES]
log(f"💡 Rules kept: {len(rules_df)}"); print(rules_df["consequent"].value_counts())

out=rules_df.copy(); out["readable"]=out["antecedent"].apply(lambda t:" ∧ ".join(t))+" → "+out["consequent"]
out.to_csv("rules_learned/fp_rules_flow_semantic.csv", index=False)

# ========= CBA (majority fallback) =========
class SimpleCBA:
    def __init__(self, rdf):
        self.rules=[(set(a),c) for a,c in zip(rdf["antecedent"], rdf["consequent"])]
    def predict_row(self, items, default_cls):
        for ant,cons in self.rules:
            if ant.issubset(items): return int(cons.replace("class=",""))
        return default_cls
    def predict(self, dfi, default_cls):
        cols=[c for c in dfi.columns if c!="class"]
        return [self.predict_row(set(str(row[c]) for c in cols), default_cls) for _,row in dfi.iterrows()]

clf=SimpleCBA(rules_df)
default_cls=int(train_df["class"].mode()[0].replace("class=",""))
y_test=test_df["class"].str.replace("class=","").astype(int).to_numpy()
y_pred=np.array(clf.predict(test_df, default_cls))

log(f"📊 Pred dist: {Counter(y_pred)}")
print("📊 Classification Report:"); print(classification_report(y_test,y_pred,digits=3))
print("MCC:", matthews_corrcoef(y_test,y_pred))
with open("classification_reports/fp_cba_report_flow_semantic.txt","w") as f:
    f.write(classification_report(y_test,y_pred,digits=3))
    f.write(f"\nMCC: {matthews_corrcoef(y_test,y_pred):.3f}\n")
log("✅ FP-Growth + CBA (flow + semantics) complete.")
