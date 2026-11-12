"""
import_data.py - batch import/update campus data in MongoDB from CSV files.

Usage (run from backend repo root, with MONGODB_URI set or in .env file):

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
PARKING_ACCESS_FIELDS = {
    "facultyStaff": "Faculty/Staff",
    "commuter": "Commuter",
    "visitor": "Visitor",
    "residenceHall": "Residence Hall",
    "hourly": "Hourly Parking",
}


# --------------------------------------------------------------------------
# Collection definitions - column names match MongoDB/Pydantic schema
# --------------------------------------------------------------------------
# key:       field used to match existing documents (upsert)
# key_type:  "int" or "str"
# required:  columns that can't be blank
# text:      optional plain-text columns (blank -> null in MongoDB)
# int_opt:   optional integer columns (blank -> null)
# coords:    row has latitude/longitude columns -> stored as [lng, lat]
# images:    row has comma-separated `images` column -> stored as list
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
        "required": ["lotId", "name", *PARKING_ACCESS_FIELDS, "latitude", "longitude"],
        "text": [],
        "int_opt": [],
        "bool": list(PARKING_ACCESS_FIELDS),
        "category_from_bool": True,
        "coords": True,
        "images": False,
        "hours": False,
        "columns": ["lotId", "name", *PARKING_ACCESS_FIELDS, "latitude", "longitude"],
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

# split cell of image links into list
def split_images(cell: str) -> list[str]:
    """'url1, url2 ; url3' -> ['url1', 'url2', 'url3'] (commas, semicolons or newlines)."""
    # treat commas, semicolons, and new lines as separators
    parts = cell.replace(";", ",").replace("\n", ",").split(",")
    # trim spaces & skip empty entries
    return [p.strip() for p in parts if p.strip()]


def parking_category(doc: dict) -> str:
    """Build the existing display label from explicit parking access flags."""
    selected = [
        label for field, label in PARKING_ACCESS_FIELDS.items() if doc[field]
    ]
    if not selected:
        raise ValueError("at least one parking category must be true")
    if selected == ["Faculty/Staff", "Commuter", "Visitor"]:
        return "Faculty/Staff, Commuter, and Visitor"
    if len(selected) == 1:
        return selected[0]
    if len(selected) == 2:
        return f"{selected[0]} and {selected[1]}"
    return ", ".join(selected[:-1]) + f", and {selected[-1]}"


# turn 1 CSV row into checked MongoDB document
def row_to_doc(row: dict, spec: dict) -> dict:
    """Convert one CSV row to a MongoDB document. Raises ValueError with a clear message."""
    # remove extra spaces from CSV column names & values
    row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
    # stop if any required field is missing
    problems = [f"'{c}' is blank" for c in spec["required"] if not row.get(c)]
    if problems:
        raise ValueError("; ".join(problems))

    doc: dict = {}
    key = spec["key"]
    try:
        # convert # IDs to integers; leave string IDs as text
        key_value = int(row[key]) if spec["key_type"] == "int" else row[key]
    except ValueError:
        raise ValueError(f"'{key}' must be a whole number, got '{row[key]}'")
    if spec["key_type"] == "int" and key_value <= 0:
        raise ValueError(f"'{key}' must be greater than 0")

    # shared fields by all campus record types
    doc["name"] = row["name"]

    for col in spec.get("bool", []):
        value = row.get(col, "").casefold()
        if value in {"true", "1", "yes", "y"}:
            doc[col] = True
        elif value in {"false", "0", "no", "n"}:
            doc[col] = False
        else:
            raise ValueError(f"'{col}' must be true or false, got '{row.get(col, '')}'")

    if spec.get("category_from_bool"):
        doc["category"] = parking_category(doc)
    else:
        doc["category"] = row["category"]

    if spec["coords"]:
        try:
            # read latitude & longitude separately before storing map coordinates
            lat, lng = float(row["latitude"]), float(row["longitude"])
        except ValueError:
            raise ValueError("latitude/longitude must be numbers")
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            raise ValueError(f"latitude/longitude out of range ({lat}, {lng})")
        # API stores coordinates as [longitude, latitude]
        doc["coordinates"] = [lng, lat]

    for col in spec["int_opt"]:
        if row.get(col):
            try:
                # optional # fields are still stored as numbers
                doc[col] = int(row[col])
            except ValueError:
                raise ValueError(f"'{col}' must be a whole number, got '{row[col]}'")
        else:
            # store missing optional field as null
            doc[col] = None

    for col in spec["text"]:
        # turn blank optional text into null instead of empty string
        doc[col] = row.get(col) or None

    if spec["images"]:
        # store image links as list
        doc["images"] = split_images(row.get("images", ""))

    if spec["hours"]:
        # include only days that have hours entered
        hours = {d: row[f"hours_{d}"] for d in DAYS if row.get(f"hours_{d}")}
        doc["hours"] = hours or None

    # keep match key separate till database write is built
    return {"_key": key_value, **doc}

# read CSV, convert rows, & collect errors
def read_csv(path: Path, spec: dict):
    """Return (docs, errors). Never writes anything."""
    # track valid rows, problems, and IDs already used in file
    docs, errors, seen = [], [], set()
    with open(path, newline="", encoding="utf-8-sig") as f:  # utf-8-sig handles Excel's BOM
        # read each CSV row as dictionary using header names
        reader = csv.DictReader(f)
        headers = {h.strip() for h in (reader.fieldnames or [])}
        # missing required header makes every row in file unusable
        missing = [c for c in spec["required"] if c not in headers]
        if missing:
            return [], [f"{path.name}: missing required column(s): {', '.join(missing)}"]
        for line_no, row in enumerate(reader, start=2):  # line 1 is the header
            if not any((v or "").strip() for v in row.values()):
                continue  # skip fully blank rows
            try:
                # convert & validate row w/o writing to MongoDB
                doc = row_to_doc(row, spec)
            except ValueError as e:
                errors.append(f"{path.name} line {line_no}: {e}")
                continue
            if doc["_key"] in seen:
                errors.append(f"{path.name} line {line_no}: duplicate {spec['key']} {doc['_key']!r}")
                continue
            # only add row after it passes checks & has unique ID
            seen.add(doc["_key"])
            docs.append(doc)
    return docs, errors


# --------------------------------------------------------------------------
# MongoDB helpers
# --------------------------------------------------------------------------

# connect to configured MongoDB database
def get_database():
    try:
        from dotenv import load_dotenv  # optional; used for local .env files
        # load local .env settings when python-dotenv is installed
        load_dotenv()
    except ImportError:
        pass
    uri = os.getenv("MONGODB_URI")
    if not uri:
        sys.exit("MONGODB_URI is not set (put it in your environment or a .env file).")
    from pymongo import MongoClient
    # stop waiting if MongoDB can't be reached
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    # use configured database/project default
    return client[os.getenv("MONGODB_DATABASE", "UWPmap")]

# insert new documents/update existing by IDs
def upsert(db, name: str, spec: dict, docs: list[dict]) -> tuple[int, int, int]:
    """Upsert docs by key. Returns (inserted, modified, unchanged)."""
    from pymongo import UpdateOne
    ops = []
    for d in docs:
        # make copy so removing helper key doesn't change input
        d = dict(d)
        key_value = d.pop("_key")
        if spec["key"] != "_id":
            d[spec["key"]] = key_value  # store ID field (e.g. buildingId)
        # update existing ID/insert new document when not found
        ops.append(UpdateOne({spec["key"]: key_value}, {"$set": d}, upsert=True))
    # send writes together (unordered mode lets independent rows continue)
    result = db[name].bulk_write(ops, ordered=False)
    inserted = result.upserted_count
    modified = result.modified_count
    return inserted, modified, len(ops) - inserted - modified


# --------------------------------------------------------------------------
# MongoDB document  ->  CSV row  (used by --export)
# --------------------------------------------------------------------------

# turn 1 MongoDB document into CSV row
def doc_to_row(doc: dict, spec: dict) -> dict:
    row = {}
    for col in spec["columns"]:
        if col == "latitude":
            # MongoDB coordinates = [longitude, latitude]
            row[col] = doc["coordinates"][1]
        elif col == "longitude":
            row[col] = doc["coordinates"][0]
        elif col == "images":
            # put image list in 1 CSV cell
            row[col] = ", ".join(doc.get("images") or [])
        elif col.startswith("hours_") and col[6:] in DAYS:
            # turn hours dictionary in 1 column per day
            row[col] = (doc.get("hours") or {}).get(col[6:], "")
        else:
            # use blank CSV cells for null values in MongoDB
            v = doc.get(col)
            row[col] = "" if v is None else v
    return row

# write documents to CSV file with expected columns
def write_csv(path: Path, spec: dict, docs: list[dict]) -> None:
    sort_key = spec["key"]
    # keep exports in predictable order
    docs = sorted(docs, key=lambda d: (str(d[sort_key]) if spec["key_type"] == "str" else d[sort_key]))
    # create output folder if doesn't exist yet
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        # write expected headers, then 1 row per document
        writer = csv.DictWriter(f, fieldnames=spec["columns"])
        writer.writeheader()
        for d in docs:
            writer.writerow(doc_to_row(d, spec))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

# parse command options & run import, dry run, or export
def main() -> None:
    parser = argparse.ArgumentParser(description="Import campus CSV data into MongoDB.")
    # accept optional collection names & import/export settings
    parser.add_argument("collections", nargs="*", choices=list(COLLECTIONS),
                        help="Which collection(s) to import (default: all CSVs found)")
    parser.add_argument("--data-dir", default="data", help="Folder holding the CSV files (default: data)")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write to MongoDB")
    parser.add_argument("--export", action="store_true", help="Export current MongoDB data to CSV files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    # no collection names means use every collection definition
    names = args.collections or list(COLLECTIONS)

    if args.export:
        # export current database records to CSV instead of importing
        db = get_database()
        for name in names:
            spec = COLLECTIONS[name]
            docs = list(db[name].find())
            write_csv(data_dir / spec["file"], spec, docs)
            print(f"exported {len(docs):>3} -> {data_dir / spec['file']}")
        return

    # 1) Read + validate everything first, so bad row never causes half-finished import
    loaded, all_errors = {}, []
    for name in names:
        # find CSV that matches collection
        path = data_dir / COLLECTIONS[name]["file"]
        if not path.exists():
            if args.collections:
                all_errors.append(f"{path} not found")
            else:
                print(f"skip {name}: {path} not found")
            continue
        docs, errors = read_csv(path, COLLECTIONS[name])
        loaded[name] = docs
        # keep checking files so all errors can be fixed at once
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
        # report what would be written without opening a database connection
        for name, docs in loaded.items():
            print(f"[dry run] {name}: {len(docs)} valid rows would be upserted")
        return

    # only connect after all CSV data has passed validation
    db = get_database()
    for name, docs in loaded.items():
        ins, mod, same = upsert(db, name, COLLECTIONS[name], docs)
        print(f"{name}: {ins} inserted, {mod} updated, {same} unchanged")
    print("Database successfully updated from CSV!")

if __name__ == "__main__":
    main()
