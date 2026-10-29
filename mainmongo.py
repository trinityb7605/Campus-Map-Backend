import os
from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database


class DatabaseConfigurationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _connect(uri: str, database_name: str) -> Database:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    database = client[database_name]
    database["buildings"].create_index("buildingId", unique=True)
    database["parkinglots"].create_index("lotId", unique=True)
    database["university_services"].create_index("buildingId")
    database["departments"].create_index("buildingId")
    return database


def get_database() -> Database:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise DatabaseConfigurationError(
            "MONGODB_URI must be set to connect to the database."
        )

    database_name = os.getenv("MONGODB_DATABASE", "UWPmap")
    return _connect(uri, database_name)
