import fitz  # PyMuPDF
import joblib
import json
import numpy as np
import pandas as pd
import re
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Paths ---
PDF_DIR = Path("resources\Challenge - 1(a)\Datasets\Pdfs")
OUTPUT_DIR = Path("resources\Challenge - 1(a)\Datasets\Output.json")
MODEL_PATH = Path("app\models\head_clf_rf_tuned.joblib")

FONT_RE = re.compile(r'(?i)bold')

# --- Feature Extraction (same as process_pdfs.py) ---
def extract_lines(pdf_path: Path) -> pd.DataFrame:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  - Error opening PDF {pdf_path.name}: {e}")
        return pd.DataFrame()
    rows = []
    for page_no, page in enumerate(doc, start=1):
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
        page_lines_data = []
        for b in blocks:
            for l in b.get("lines", []):
                text = " ".join(s["text"] for s in l["spans"]).strip()
                if not text: continue
                text = re.sub(r'\s+', ' ', text).strip()
                span = l["spans"][0]
                page_lines_data.append({
                    "text": text, "font_size": span["size"], "font_name": span["font"],
                    "bold": bool(FONT_RE.search(span["font"])), "x0": l["bbox"][0],
                    "y0": l["bbox"][1], "page": page_no
                })
        page_df = pd.DataFrame(page_lines_data)
        if not page_df.empty:
            min_x0 = page_df['x0'].min()
            page_df['indentation'] = page_df['x0'] - min_x0
            page_df['v_spacing'] = page_df['y0'].diff().fillna(0)
            rows.extend(page_df.to_dict('records'))
    doc.close()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["len"] = df.text.str.len()
    df["upper_ratio"] = df.text.str.fullmatch(r"[A-Z0-9\W]+").fillna(False).astype(int)
    df["ends_colon"] = df.text.str.endswith(":").astype(int)
    df["font_rank"] = df.groupby("page")["font_size"].rank(method="dense", ascending=False)
    df["centered"] = (df['indentation'] < 10).astype(int)
    df = df.drop_duplicates(subset=['text', 'page']).reset_index(drop=True)
    return df

# --- Labeling Function ---
def label_lines(df, outline):
    df['is_heading'] = 0
    df['heading_level'] = None
    outline_set = {(o['text'].strip(), int(o['page'])): o['level'] for o in outline}
    for idx, row in df.iterrows():
        key = (row['text'].strip(), int(row['page'])-1)  # page-1 for zero-indexed
        if key in outline_set:
            df.at[idx, 'is_heading'] = 1
            df.at[idx, 'heading_level'] = outline_set[key]
    return df

# --- Main Training Script ---
all_dfs = []
pdf_files = list(PDF_DIR.glob("*.pdf"))
for pdf_path in pdf_files:
    json_path = OUTPUT_DIR / (pdf_path.stem + ".json")
    if not json_path.exists():
        print(f"No ground truth for {pdf_path.name}")
        continue
    with open(json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    outline = gt.get("outline", [])
    df = extract_lines(pdf_path)
    if df.empty:
        continue
    df = label_lines(df, outline)
    all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index=True)
features = ["font_size", "bold", "len", "upper_ratio", "ends_colon", "font_rank", "centered", "indentation", "v_spacing"]
X = full_df[features].values
y_bin = full_df['is_heading'].values
y_multi = full_df.loc[full_df['is_heading'] == 1, 'heading_level'].values
X_multi = full_df.loc[full_df['is_heading'] == 1, features].values

# --- Train Models ---

# --- Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_val_scaled = sc.transform(X_val)
clf_bin = RandomForestClassifier(n_estimators=100, random_state=42)
clf_bin.fit(X_train_scaled, y_train)

# --- Binary Classification Metrics ---
y_pred = clf_bin.predict(X_val_scaled)
print("\n--- Heading Detection Metrics (Binary) ---")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_pred):.4f}")
print(f"F1-score: {f1_score(y_val, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# --- Multi-class Classification (on headings only) ---
X_multi = full_df.loc[full_df['is_heading'] == 1, features].values
y_multi = full_df.loc[full_df['is_heading'] == 1, 'heading_level'].values
X_multi_train, X_multi_val, y_multi_train, y_multi_val = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi)
X_multi_train_scaled = sc.transform(X_multi_train)
X_multi_val_scaled = sc.transform(X_multi_val)
clf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
clf_multi.fit(X_multi_train_scaled, y_multi_train)
y_multi_pred = clf_multi.predict(X_multi_val_scaled)
print("\n--- Heading Level Metrics (Multi-class) ---")
print(classification_report(y_multi_val, y_multi_pred))

# --- Feature Importance ---
print("\n--- Feature Importances (Binary Classifier) ---")
for feat, imp in zip(features, clf_bin.feature_importances_):
    print(f"{feat}: {imp:.4f}")
print("\n--- Feature Importances (Multi-class Classifier) ---")
for feat, imp in zip(features, clf_multi.feature_importances_):
    print(f"{feat}: {imp:.4f}")

# --- Save Model Bundle ---
joblib.dump((sc, clf_bin, clf_multi), MODEL_PATH)
print(f"✅ Model trained and saved to {MODEL_PATH}")

# --- Save Model Bundle ---
joblib.dump((sc, clf_bin, clf_multi), MODEL_PATH)
print(f"✅ Model trained and saved to {MODEL_PATH}")
