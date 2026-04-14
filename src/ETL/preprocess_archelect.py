import os
import pandas as pd
from helper_function.print import *

TEXT_ROOT = "data/raw/text_files"  
CSV_PATH = "data/raw/archelect_search.csv"
OUTPUT_PATH = "data/clean/archelect_clean.parquet"
OUTPUT_DIR = "data/clean"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure the folder exists

COLUMNS_TO_KEEP = [
    'id', 'date', 'subject', 'title',
    'contexte-election', 'contexte-tour',
    'departement',
    'titulaire-nom', 'titulaire-prenom', 'titulaire-sexe',
    'titulaire-age',
    'titulaire-mandat-en-cours', 'titulaire-mandat-passe',
    'titulaire-liste'
]

rows = []

print("="*20 + " [Start extraction] " + "="*20)
for root, dirs, files in os.walk(TEXT_ROOT):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            file_id = os.path.splitext(file)[0]
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    text = f.read()
            rows.append({
                            "id": file_id,
                            "raw_text": text
                        })

df_txt = pd.DataFrame(rows)
print(blue(f"Number of txt loaded : {len(df_txt)}"))

df_meta = pd.read_csv(CSV_PATH)
df_meta = df_meta[COLUMNS_TO_KEEP]
print(f"Number of rows in archelect : {len(df_meta)}")

df_final = df_txt.merge(df_meta, on="id", how="left")
# Remove useless spaces
df_final["raw_text"] = df_final["raw_text"].str.strip()

print(blue(f"Number of rows after : {len(df_final)}"))

missing_meta = df_final["date"].isna().sum()
print(f"TXT without associate metadata : {missing_meta}")

df_final.to_parquet(OUTPUT_PATH, index=False)

print(green("Preprocessing is over"))