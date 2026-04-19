# mc cp data/raw/archelect_search.csv s3/aelbaz/Linguistic_Analysis/data/raw/archelect_search.csv
import os
import re
import zipfile
import tempfile
from pathlib import Path
from dotenv import load_dotenv

import duckdb
import pandas as pd
import requests
from tqdm import tqdm

from arkindex_export import open_database, Element, Transcription
from arkindex_export.queries import list_children

load_dotenv()
EXPORT_URL = (
    "https://demo.arkindex.org/api/v1/export/d1295972-0f7d-400e-b9d8-c6ee07bdb68b/"
)
ARKINDEX_TOKEN = os.getenv("ARKINDEX_TOKEN")  # optionnel

# MinIO S3
S3_BUCKET = os.getenv("S3_BUCKET", "arkindex-export")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "s3")
# Output parquet
OUTPUT_PATH = "Linguistic_Analysis/data/raw/transcriptions_1993_legislatives.parquet"

LEGISLATIVES_1993_ID = "2d71d778-ce90-424e-9313-8b208113e512"
YEAR = "1993"
ELECTION_TYPE = "legislatives"


def sanitize(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-. ]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    return name or "untitled"


def download_export_zip(url: str, out_path: Path):
    headers = {}
    print("Token:", ARKINDEX_TOKEN)
    if ARKINDEX_TOKEN:
        headers["Authorization"] = f"Bearer {ARKINDEX_TOKEN}"

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=10_485_760):  # 10 MB
                if chunk:
                    f.write(chunk)


def extract_sqlite(zip_path: Path, extract_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as z:
        sqlite_files = [n for n in z.namelist() if n.endswith(".sqlite")]
        if not sqlite_files:
            raise RuntimeError("Pas de fichier .sqlite dans le zip.")
        sqlite_name = sqlite_files[0]
        z.extract(sqlite_name, path=extract_dir)
        return extract_dir / sqlite_name


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "export.zip"
        extract_dir = tmp / "unzipped"

        print("Téléchargement export Arkindex (zip) en local temporaire...")
        download_export_zip(EXPORT_URL, zip_path)

        print("Extraction sqlite...")
        sqlite_path = extract_sqlite(zip_path, extract_dir)

        print("Ouverture sqlite...")
        open_database(sqlite_path)

        print("Lecture documents...")
        documents = list_children(LEGISLATIVES_1993_ID).where(
            Element.type == "document"
        )

        rows = []
        for document in tqdm(documents, desc="Documents"):
            pages = list_children(document.id).where(Element.type == "page")

            chunks = []
            for page in pages:
                t = (
                    Transcription.select()
                    .where(Transcription.element == page.id)
                    .first()
                )
                if t and t.text:
                    chunks.append(t.text)

            if not chunks:
                continue

            rows.append(
                {
                    "year": YEAR,
                    "type": ELECTION_TYPE,
                    "document_id": document.id,
                    "document_name": sanitize(document.name),
                    "text": "\n".join(chunks),
                }
            )

        print(f"{len(rows)} documents avec transcription")

        df = pd.DataFrame(rows)

        print("Ecriture parquet vers MinIO via DuckDB...")
        con = duckdb.connect(database=":memory:")

        # Active httpfs + config s3
        con.sql("INSTALL httpfs;")
        con.sql("LOAD httpfs;")

        con.sql(f"""
            SET s3_endpoint='{S3_ENDPOINT}';
            SET s3_access_key_id='{S3_ACCESS_KEY}';
            SET s3_secret_access_key='{S3_SECRET_KEY}';
            SET s3_url_style='path';
            SET s3_use_ssl=false;
        """)

        con.register("transcriptions_df", df)

        con.sql(f"""
            COPY transcriptions_df
            TO 's3://{S3_BUCKET}/{OUTPUT_PATH}'
            (FORMAT PARQUET);
        """)

        print(f"OK -> s3://{S3_BUCKET}/{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
