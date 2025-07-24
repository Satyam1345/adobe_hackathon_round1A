import fitz  # PyMuPDF
import joblib
import json
import numpy as np
import pandas as pd
import re
import os
from pathlib import Path

# --- Configuration ---
PDF_DIR = Path("/app/input")
OUT_DIR = Path("/app/output")
MODEL_PATH = Path("/app/models/head_clf_rf_tuned.joblib")

# Regular expression to detect bold fonts
FONT_RE = re.compile(r'(?i)bold')

# --- Feature Extraction and Prediction Functions ---

def extract_lines(pdf_path: Path) -> pd.DataFrame:
    """
    Extracts text lines and their properties (font size, name, position, etc.)
    from each page of a PDF file.
    """
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

        # Create a DataFrame for the current page to calculate relative features
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

    # --- Feature Engineering ---
    df["len"] = df.text.str.len()
    df["upper_ratio"] = df.text.str.fullmatch(r"[A-Z0-9\W]+").fillna(False).astype(int)
    df["ends_colon"] = df.text.str.endswith(":").astype(int)
    df["font_rank"] = df.groupby("page")["font_size"].rank(method="dense", ascending=False)
    df["centered"] = (df['indentation'] < 10).astype(int)
    df = df.drop_duplicates(subset=['text', 'page']).reset_index(drop=True)
    return df

def prepare_matrix(df: pd.DataFrame):
    """Prepares the feature matrix for the machine learning model."""
    feature_columns = ["font_size", "bold", "len", "upper_ratio", "ends_colon",
                       "font_rank", "centered", "indentation", "v_spacing"]
    return df[feature_columns].values

def predict_outline(df: pd.DataFrame, model_bundle):
    """Uses the pre-trained model to predict which lines are headings and their levels."""
    sc, clf_bin, clf_multi = model_bundle
    
    X = prepare_matrix(df)
    X = sc.transform(X)

    # Predict which lines are headings
    is_heading = clf_bin.predict(X)
    headings_df = df[is_heading == 1].copy()

    if not headings_df.empty:
        # Predict the level (H1, H2, etc.) for the identified headings
        heading_levels = clf_multi.predict(X[is_heading == 1])
        headings_df['level'] = heading_levels
        headings_df = headings_df.drop_duplicates(subset=['text', 'page'])
        headings_df['level_num'] = headings_df['level'].str.replace('H', '').astype(int)

        # --- Post-processing to fix heading hierarchy ---
        headings_df = headings_df.sort_values(['page', 'y0']).reset_index(drop=True)
        previous_level = 0
        for i, row in headings_df.iterrows():
            current_level = row['level_num']
            if current_level > previous_level + 1:
                new_level = previous_level + 1
                headings_df.loc[i, 'level'] = f'H{new_level}'
                headings_df.loc[i, 'level_num'] = new_level
            previous_level = headings_df.loc[i, 'level_num']

        return headings_df[['text', 'page', 'level']]
    return pd.DataFrame(columns=['text', 'page', 'level'])

# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Starting PDF Outline Extraction ---")
    OUT_DIR.mkdir(exist_ok=True)

    # 1. Load the pre-trained model bundle once
    try:
        model_bundle = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: Model file not found at {MODEL_PATH}. Exiting.")
        exit(1)

    # 2. Process each PDF found in the input directory
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDF files found in /app/input directory.")
    else:
        print(f"Found {len(pdf_files)} PDF(s) to process...")

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        try:
            # 3. Extract features and predict the outline
            df_full = extract_lines(pdf_path)
            if df_full.empty:
                print(f"  - ⚠️ Could not extract any text from {pdf_path.name}. Skipping.")
                continue

            predicted_headings = predict_outline(df_full, model_bundle)

            # 4. Extract the document title (heuristic: largest font on the first page)
            title = "No Title Found"
            page_1_lines = df_full[df_full.page == 1]
            if not page_1_lines.empty:
                title = page_1_lines.sort_values("font_size", ascending=False).iloc[0].text.strip()

            # 5. Format the output JSON according to the required schema
            outline = []
            if not predicted_headings.empty:
                for row in predicted_headings.itertuples():
                    outline.append({
                        "level": row.level,
                        "text": row.text.strip(),
                        "page": int(row.page)-1
                    })

            output_json = {"title": title, "outline": outline}
            
            # 6. Write the final JSON file to the output directory
            out_path = OUT_DIR / (pdf_path.stem + ".json")
            with open(out_path, "w") as f:
                json.dump(output_json, f, indent=2)
            print(f"  - ✅ Successfully wrote output to {out_path.name}")

        except Exception as e:
            print(f"  - ❌ An unexpected error occurred while processing {pdf_path.name}: {e}")

    print("\n--- PDF processing complete. ---")
