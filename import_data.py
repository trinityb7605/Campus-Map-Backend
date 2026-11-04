"""
import_data.py - batch import / update campus data in MongoDB from CSV files.

Usage (run from the backend repo root, with MONGODB_URI set or in a .env file):

    python import_data.py                     # import every CSV found in ./data
    python import_data.py buildings           # import only one collection
    python import_data.py --dry-run           # validate + show what WOULD change
    python import_data.py --data-dir other/   # use a different folder
    python import_data.py --export            # DB -> CSV files (to bootstrap/refresh the sheet)

CSV files (one per collection, in ./data):
    buildings.csv, parkinglots.csv, departments.csv, university_services.csv

Rows are UPSERTED by their ID column, so re-running the script is safe:
existing records are updated, new ones are inserted, nothing is deleted.
If any row is invalid, NOTHING is written and every problem is listed.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# --------------------------------------------------------------------------
# Collection definitions - column names match the MongoDB / Pydantic schema
# --------------------------------------------------------------------------
# key:       the field used to match existing documents (upsert)
# key_type:  "int" or "str"
# required:  columns that must not be blank
# text:      optional plain-text columns (blank -> null in MongoDB)
# int_opt:   optional integer columns (blank -> null)
# coords:    row has latitude/longitude columns -> stored as [lng, lat]
# images:    row has a comma-separated `images` column -> stored as a list
# hours:     row has hours_monday ... hours_sunday columns -> stored as dict
COLLECTIONS = {
    "buildings": {
        "file": "buildings.csv",
        "key": "buildingId",
        "key_type": "int",
        "required": ["buildingId", "name", "category", "latitude", "longitude"],
        "text": ["demo_description"],
        "int_opt": [],
        "coords": True,
        "images": True,
        "hours": False,
        "columns": ["buildingId", "name", "category", "latitude", "longitude",
                    "demo_description", "images"],
    },
    "parkinglots": {
        "file": "parkinglots.csv",
        "key": "lotId",
        "key_type": "int",
        "required": ["lotId", "name", "category", "latitude", "longitude"],
        "text": [],
        "int_opt": [],
        "coords": True,
        "images": False,
        "hours": False,
        "columns": ["lotId", "name", "category", "latitude", "longitude"],
    },
    "departments": {
        "file": "departments.csv",
        "key": "_id",
        "key_type": "str",
        "required": ["_id", "name", "category"],
        "text": ["room", "description", "phone", "email", "hours_note",
                 "official_website", "catalog_website"],
        "int_opt": ["buildingId"],
        "coords": False,
        "images": False,
        "hours": True,
        "columns": ["_id", "name", "category", "buildingId", "room", "description",
                    "phone", "email"] + [f"hours_{d}" for d in DAYS]
                   + ["hours_note", "official_website", "catalog_website"],
    },
    "university_services": {
        "file": "university_services.csv",
        "key": "_id",
        "key_type": "str",
        "required": ["_id", "name", "category"],
        "text": ["room", "description", "phone", "email", "hours_note",
                 "official_website", "catalog_website"],
        "int_opt": ["buildingId"],
        "coords": False,
        "images": False,
        "hours": True,
        "columns": ["_id", "name", "category", "buildingId", "room", "description",
                    "phone", "email"] + [f"hours_{d}" for d in DAYS]
                   + ["hours_note", "official_website", "catalog_website"],
    },
}


# --------------------------------------------------------------------------
# CSV row  ->  MongoDB document
# --------------------------------------------------------------------------
def split_images(cell: str) -> list[str]:
    """'url1, url2 ; url3' -> ['url1', 'url2', 'url3'] (commas, semicolons or newlines)."""
    parts = cell.replace(";", ",").replace("\n", ",").split(",")
    return [p.strip() for p in parts if p.strip()]


def row_to_doc(row: dict, spec: dict) -> dict:
    """Convert one CSV row to a MongoDB document. Raises ValueError with a clear message."""
    row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
    problems = [f"'{c}' is blank" for c in spec["required"] if not row.get(c)]
    if problems:
        raise ValueError("; ".join(problems))

    doc: dict = {}
    key = spec["key"]
    try:
        key_value = int(row[key]) if spec["key_type"] == "int" else row[key]
    except ValueError:
        raise ValueError(f"'{key}' must be a whole number, got '{row[key]}'")
    if spec["key_type"] == "int" and key_value <= 0:
        raise ValueError(f"'{key}' must be greater than 0")

    doc["name"] = row["name"]
    doc["category"] = row["category"]

    if spec["coords"]:
        try:
            lat, lng = float(row["latitude"]), float(row["longitude"])
        except ValueError:
            raise ValueError("latitude/longitude must be numbers")
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            raise ValueError(f"latitude/longitude out of range ({lat}, {lng})")
        # The API stores coordinates as [longitude, latitude]
        doc["coordinates"] = [lng, lat]

    for col in spec["int_opt"]:
        if row.get(col):
            try:
                doc[col] = int(row[col])
            except ValueError:
                raise ValueError(f"'{col}' must be a whole number, got '{row[col]}'")
        else:
            doc[col] = None

    for col in spec["text"]:
        doc[col] = row.get(col) or None

    if spec["images"]:
        doc["images"] = split_images(row.get("images", ""))

    if spec["hours"]:
        hours = {d: row[f"hours_{d}"] for d in DAYS if row.get(f"hours_{d}")}
        doc["hours"] = hours or None

    return {"_key": key_value, **doc}


def read_csv(path: Path, spec: dict):
    """Return (docs, errors). Never writes anything."""
    docs, errors, seen = [], [], set()
    with open(path, newline="", encoding="utf-8-sig") as f:  # utf-8-sig handles Excel's BOM
        reader = csv.DictReader(f)
        headers = {h.strip() for h in (reader.fieldnames or [])}
        missing = [c for c in spec["required"] if c not in headers]
        if missing:
            return [], [f"{path.name}: missing required column(s): {', '.join(missing)}"]
        for line_no, row in enumerate(reader, start=2):  # line 1 is the header
            if not any((v or "").strip() for v in row.values()):
                continue  # skip fully blank rows (common in exported sheets)
            try:
                doc = row_to_doc(row, spec)
            except ValueError as e:
                errors.append(f"{path.name} line {line_no}: {e}")
                continue
            if doc["_key"] in seen:
                errors.append(f"{path.name} line {line_no}: duplicate {spec['key']} {doc['_key']!r}")
                continue
            seen.add(doc["_key"])
            docs.append(doc)
    return docs, errors


# --------------------------------------------------------------------------
# MongoDB helpers
# --------------------------------------------------------------------------
def get_database():
    try:
        from dotenv import load_dotenv  # optional; used for local .env files
        load_dotenv()
    except ImportError:
        pass
    uri = os.getenv("MONGODB_URI")
    if not uri:
        sys.exit("MONGODB_URI is not set (put it in your environment or a .env file).")
    from pymongo import MongoClient
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    return client[os.getenv("MONGODB_DATABASE", "UWPmap")]


def upsert(db, name: str, spec: dict, docs: list[dict]) -> tuple[int, int, int]:
    """Upsert docs by key. Returns (inserted, modified, unchanged)."""
    from pymongo import UpdateOne
    ops = []
    for d in docs:
        d = dict(d)
        key_value = d.pop("_key")
        if spec["key"] != "_id":
            d[spec["key"]] = key_value  # store the ID field itself (e.g. buildingId)
        ops.append(UpdateOne({spec["key"]: key_value}, {"$set": d}, upsert=True))
    result = db[name].bulk_write(ops, ordered=False)
    inserted = result.upserted_count
    modified = result.modified_count
    return inserted, modified, len(ops) - inserted - modified


# --------------------------------------------------------------------------
# MongoDB document  ->  CSV row  (used by --export)
# --------------------------------------------------------------------------
def doc_to_row(doc: dict, spec: dict) -> dict:
    row = {}
    for col in spec["columns"]:
        if col == "latitude":
            row[col] = doc["coordinates"][1]
        elif col == "longitude":
            row[col] = doc["coordinates"][0]
        elif col == "images":
            row[col] = ", ".join(doc.get("images") or [])
        elif col.startswith("hours_") and col[6:] in DAYS:
            row[col] = (doc.get("hours") or {}).get(col[6:], "")
        else:
            v = doc.get(col)
            row[col] = "" if v is None else v
    return row


def write_csv(path: Path, spec: dict, docs: list[dict]) -> None:
    sort_key = spec["key"]
    docs = sorted(docs, key=lambda d: (str(d[sort_key]) if spec["key_type"] == "str" else d[sort_key]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=spec["columns"])
        writer.writeheader()
        for d in docs:
            writer.writerow(doc_to_row(d, spec))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Import campus CSV data into MongoDB.")
    parser.add_argument("collections", nargs="*", choices=list(COLLECTIONS),
                        help="Which collection(s) to import (default: all CSVs found)")
    parser.add_argument("--data-dir", default="data", help="Folder holding the CSV files (default: data)")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write to MongoDB")
    parser.add_argument("--export", action="store_true", help="Export current MongoDB data to CSV files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    names = args.collections or list(COLLECTIONS)

    if args.export:
        db = get_database()
        for name in names:
            spec = COLLECTIONS[name]
            docs = list(db[name].find())
            write_csv(data_dir / spec["file"], spec, docs)
            print(f"exported {len(docs):>3} -> {data_dir / spec['file']}")
        return

    # 1) Read + validate everything first, so a bad row never causes a half-finished import
    loaded, all_errors = {}, []
    for name in names:
        path = data_dir / COLLECTIONS[name]["file"]
        if not path.exists():
            if args.collections:
                all_errors.append(f"{path} not found")
            else:
                print(f"skip {name}: {path} not found")
            continue
        docs, errors = read_csv(path, COLLECTIONS[name])
        loaded[name] = docs
        all_errors += errors

    if all_errors:
        print("Import aborted - fix these rows and run again (nothing was written):")
        for e in all_errors:
            print("  -", e)
        sys.exit(1)
    if not loaded:
        sys.exit("No CSV files found to import.")

    # 2) Write
    if args.dry_run:
        for name, docs in loaded.items():
            print(f"[dry run] {name}: {len(docs)} valid rows would be upserted")
        return

    db = get_database()
    for name, docs in loaded.items():
        ins, mod, same = upsert(db, name, COLLECTIONS[name], docs)
        print(f"{name}: {ins} inserted, {mod} updated, {same} unchanged")
    print("Database successfully updated from CSV!")


if __name__ == "__main__":
    main()
