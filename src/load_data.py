import pandas as pd
from datasets import load_dataset

def load_scitldr_dataset(save_path="data/scitldr_clean.csv", limit=5000):
    dataset = load_dataset("scitldr", "AIC", trust_remote_code=True)["train"]

    rows = []
    for i, item in enumerate(dataset):
        source = item.get("source", [])
        tldr = item.get("target", [])
        if source and tldr:
            rows.append({
                "id": item["paper_id"],
                "title": item.get("title", "N/A"),
                "abstract": " ".join(source).strip(),
                "tldr": tldr[0].strip()
            })
        if len(rows) >= limit:
            break

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved {len(df)} entries to {save_path}")
