import logging
from pathlib import Path
import duckdb


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


def make_parquet(bucket: str, input_path: str, output_path: str = None) -> str:
    """Convert CSV to Parquet on S3 using DuckDB."""

    if output_path is None:
        output_path = input_path.replace(".csv", ".parquet")

    con = duckdb.connect(database=":memory:")

    logger.info(f"Converting s3://{bucket}/{input_path} → s3://{bucket}/{output_path}")
    con.sql(f"""
        COPY (
            SELECT * 
            FROM read_csv_auto('s3://{bucket}/{input_path}')
        )
        TO 's3://{bucket}/{output_path}'
        (FORMAT PARQUET)
    """)

    logger.info(f"Parquet saved to s3://{bucket}/{output_path}")
    con.close()
    return output_path


def load_data(path: str) -> duckdb.DuckDBPyRelation:
    """Load dataset from Parquet entirely in DuckDB."""
    path = Path(path)

    logger.info(f"Loading data from {path}")
    con = duckdb.connect(":memory:")
    rel = con.sql(f"SELECT * FROM read_parquet('{path}')")

    # Validation SQL
    row_count = rel.aggregate("count(*)").fetchone()[0]
    if row_count == 0:
        logger.error("Parquet file is empty")
        raise ValueError("Parquet is empty")

    cols = rel.columns
    expected_cols = ["country", "year", "region", "costhealthydietpppusd"]
    missing_cols = [col for col in expected_cols if col not in cols]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    valid_rows = (
        rel.filter("costhealthydietpppusd IS NOT NULL")
        .aggregate("count(*)")
        .fetchone()[0]
    )
    if valid_rows == 0:
        logger.error("No valid cost data found")
        raise ValueError("All cost values are NaN")

    logger.info(f"Loaded {row_count:,} rows, {len(cols)} columns")
    return rel
