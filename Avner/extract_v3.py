from arkindex_export import open_database, Element, Transcription
from arkindex_export.queries import list_children
from pathlib import Path
from tqdm import tqdm
import duckdb
import re
from dotenv import load_dotenv
import os

load_dotenv()

SQLITE_PATH = os.getenv("SQLITE_PATH")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX")


def export_arkindex_to_minio(
    sqlite_path: str, bucket: str, s3_prefix: str = "arkindex_exports"
) -> None:
    """
    Exporte toutes les transcriptions Arkindex vers MinIO.
    1 document = 1 fichier texte sur S3.
    """

    def clean_name(name: str) -> str:
        return re.sub(r"[^\w\-\.]", "_", name)

    open_database(Path(sqlite_path))
    con = duckdb.connect(":memory:")

    root_folders = Element.select().where(Element.type == "folder")

    for folder in tqdm(root_folders):
        documents = list_children(folder.id).where(Element.type == "document")

        for document in documents:
            pages = list_children(document.id).where(Element.type == "page")

            full_text = ""

            for page in pages:
                page_transcription = (
                    Transcription.select()
                    .where(Transcription.element == page.id)
                    .first()
                )
                if page_transcription:
                    full_text += page_transcription.text

            if not full_text.strip():
                continue

            safe_folder = clean_name(folder.name)
            safe_doc = clean_name(document.name)

            s3_path = f"s3://{bucket}/{s3_prefix}/{safe_folder}/{safe_doc}.txt"

            # table temporaire
            con.execute("CREATE OR REPLACE TABLE tmp(text VARCHAR)")
            con.execute("DELETE FROM tmp")
            con.execute("INSERT INTO tmp VALUES (?)", [full_text])

            # écriture directe sur MinIO
            con.sql(f"""
                COPY (SELECT text FROM tmp)
                TO '{s3_path}'
                (FORMAT CSV, HEADER FALSE, DELIMITER '\n', QUOTE '');
            """)

    con.close()
    print("Export terminé.")


if __name__ == "__main__":
    export_arkindex_to_minio(
        sqlite_path=SQLITE_PATH, bucket=S3_BUCKET, s3_prefix=S3_PREFIX
    )
