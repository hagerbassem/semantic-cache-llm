import pandas as pd
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("train.csv").dropna().head(1000)
model = SentenceTransformer("all-MiniLM-L6-v2")

e1 = model.encode(df["question1"].tolist(), convert_to_tensor=True)
e2 = model.encode(df["question2"].tolist(), convert_to_tensor=True)

df["sim"] = util.cos_sim(e1, e2).diagonal().cpu().numpy()
print(df.groupby("is_duplicate")["sim"].mean())