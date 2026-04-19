import os
import pandas as pd
from helper_function.print import *

TEXT_ROOT = "data/raw/text_files"
CSV_PATH = "data/raw/archelect_search.csv"
OUTPUT_PATH = "data/clean/archelect_clean.parquet"
OUTPUT_DIR = "data/clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure the folder exists

COLUMNS_TO_KEEP = [
    "id",
    "date",
    "subject",
    "title",
    "contexte-election",
    "contexte-tour",
    "departement",
    "titulaire-nom",
    "titulaire-prenom",
    "titulaire-sexe",
    "titulaire-age",
    "titulaire-mandat-en-cours",
    "titulaire-mandat-passe",
    "titulaire-liste",
]

rows = []

print("=" * 20 + " [Start extraction] " + "=" * 20)
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
            rows.append({"id": file_id, "raw_text": text})

df_txt = pd.DataFrame(rows)
print(blue(f"Number of txt loaded : {len(df_txt)}"))

df_meta = pd.read_csv(CSV_PATH)
df_meta = df_meta[COLUMNS_TO_KEEP]
print(f"Number of rows in archelect : {len(df_meta)}")

df_final = df_txt.merge(df_meta, on="id", how="left")

# Remove useless spaces and upper cases
df_final["raw_text"] = df_final["raw_text"].str.strip()
df_final["raw_text"] = df_final["raw_text"].str.lower()

print(blue(f"Number of rows after : {len(df_final)}"))


# --- Political Party Classification ---
def map_to_party(row):
    liste = str(row["titulaire-liste"]).lower()
    text = str(row["raw_text"]).lower()

    # 1. Front National
    if any(
        x in liste
        for x in [
            "front national",
            "français d'abord",
            "france française",
            "liste entente populaire et nationale",
        ]
    ):
        return "Front National"
    if "front national" in text or " fn " in text:
        return "Front National"

    # 2. PCF
    if any(
        x in liste
        for x in ["communiste", "forces de gauche", "transformation à grande vitesse"]
    ):
        return "PCF"
    if "parti communiste" in text or " pcf " in text:
        return "PCF"

    # 3. PS
    if any(
        x in liste
        for x in [
            "majorité présidentielle",
            "france unie",
            "mitterrand",
            "union de la gauche",
            "socialiste",
        ]
    ):
        return "PS"
    if "parti socialiste" in text or " ps " in text:
        return "PS"

    # 4. RPR / LR (Main Right)
    if any(
        x in liste
        for x in [
            "rassemblement pour la république",
            "union du rassemblement et du centre",
            "union pour la france",
            "majorité de la france",
            "union pour une nouvelle majorité",
        ]
    ):
        return "RPR / LR"
    if " rpr " in text or "rassemblement pour la république" in text:
        return "RPR / LR"

    # 5. UDF / Centre
    if any(
        x in liste
        for x in [
            "démocratie française",
            "udf",
            "union centriste",
            "rassemblement démocrate",
            "ouverture au centre",
        ]
    ):
        return "UDF / Centre"
    if " udf " in text or "union pour la démocratie française" in text:
        return "UDF / Centre"

    # 6. Ecologistes
    if any(
        x in liste
        for x in ["ecologie", "verts", "environnement", "génération ecologie"]
    ):
        return "Ecologistes"
    if "écologiste" in text or " les verts " in text:
        return "Ecologistes"

    # 7. Extreme Gauche
    if any(
        x in liste
        for x in [
            "lutte ouvrière",
            "lcr",
            "parti des travailleurs",
            "autogestion",
            "révolutionnaire",
        ]
    ):
        return "Extreme Gauche"

    # 8. Extreme Droite / Nationalistes (Non-FN)
    if any(
        x in liste for x in ["nationaliste", "opposition nationale", "force nationale"]
    ):
        return "Extreme Droite"

    return "Independent"


print("Mapping political parties...")
df_final["affiliate political party"] = df_final.apply(map_to_party, axis=1)

missing_meta = df_final["date"].isna().sum()
print(f"TXT without associate metadata : {missing_meta}")

df_final.to_parquet(OUTPUT_PATH, index=False)

print(green("Preprocessing is over"))
