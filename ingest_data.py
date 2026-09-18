import sqlite3
import urllib.request
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
ZIP_PATH = "data/ai4i2020.zip"
CSV_NAME = "ai4i2020.csv"
DB_PATH = "data/maintenance.db"

# 1. Download and extract dataset
print("[*] Downloading dataset from UCI...")
import io
import zipfile

req = urllib.request.Request(
    DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
)
with urllib.request.urlopen(req) as resp:
    zip_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extract(CSV_NAME, path="data/")

# 2. Read and standardize column names
# Replacing spaces, brackets, and special characters makes SQL querying straightforward
df = pd.read_csv(f"data/{CSV_NAME}")
df.columns = [
    col.strip()
    .replace(" ", "_")
    .replace("[", "")
    .replace("]", "")
    .replace("/", "_")
    for col in df.columns
]

print(f"[*] Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

# 3. Store into SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Store as the 'equipment_telemetry' table
df.to_sql("equipment_telemetry", conn, if_exists="replace", index=False)

# Add an index on Product_ID and Type for efficient query execution
cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON equipment_telemetry(Type);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_failure ON equipment_telemetry(Machine_failure);")
conn.commit()
conn.close()

print(f"[+] Data loaded and indexed in {DB_PATH}")