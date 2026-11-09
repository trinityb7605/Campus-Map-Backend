import os
from functools import lru_cache
from pymongo import MongoClient
from pymongo.database import Database

# show clear error when database connection settings missing
class DatabaseConfigurationError(RuntimeError):
    pass

# connect to MongoDB & set up indexes API uses for lookups & chat search
@lru_cache(maxsize=1)
def _connect(uri: str, database_name: str) -> Database:
    # wait up to 5 seconds when can't reach MongoDB
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    database = client[database_name]

    # keep campus IDs unique & make building lookups faster
    database["buildings"].create_index("buildingId", unique=True)
    database["parkinglots"].create_index("lotId", unique=True)
    database["university_services"].create_index("buildingId")
    database["departments"].create_index("buildingId")

    # let chat assistant search fields, ranking names above descriptions (MongoDB allows 1 text index per collection)
    database["buildings"].create_index(
        [("name", "text"), ("category", "text"), ("demo_description", "text")],
        weights={"name": 10, "category": 3, "demo_description": 1},
        name="buildings_text",
    )
    database["departments"].create_index(
        [("name", "text"), ("category", "text"), ("description", "text")],
        weights={"name": 10, "category": 3, "description": 2},
        name="departments_text",
    )
    database["university_services"].create_index(
        [("name", "text"), ("category", "text"), ("description", "text")],
        weights={"name": 10, "category": 3, "description": 2},
        name="university_services_text",
    )
    database["parkinglots"].create_index(
        [("name", "text"), ("category", "text")],
        weights={"name": 10, "category": 3},
        name="parkinglots_text",
    )
    return database

# check connection settings & return configured database
def get_database() -> Database:
    # read connection URI from environment (not source code)
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise DatabaseConfigurationError(
            "MONGODB_URI must be set to connect to the database."
        )

    # use configured name/project's default database
    database_name = os.getenv("MONGODB_DATABASE", "UWPmap")
    # reuse connection when same URI & database name are requested again
    return _connect(uri, database_name)
