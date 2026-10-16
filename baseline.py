import pandas as pd
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("train.csv").dropna().head(2000)
model = SentenceTransformer("all-MiniLM-L6-v2")

e1 = model.encode(df["question1"].tolist(), convert_to_tensor=True)
e2 = model.encode(df["question2"].tolist(), convert_to_tensor=True)
df["sim"] = util.cos_sim(e1, e2).diagonal().cpu().numpy()

dup = df["is_duplicate"] == 1
rows = []
for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    hit = df["sim"] >= t
    tp = (hit & dup).sum()
    fp = (hit & ~dup).sum()
    rows.append({
        "threshold": t,
        "hit_rate": round(tp / dup.sum(), 3),
        "error_rate": round(fp / max(hit.sum(), 1), 3),
    })

res = pd.DataFrame(rows)
print(res)
res.to_csv("baseline_results.csv", index=False)