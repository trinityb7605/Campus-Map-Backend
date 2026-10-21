import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

from mainmongo import DatabaseConfigurationError, get_database
from schema import Building, BuildingCreate, ParkingLot, ParkingLotCreate

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(DatabaseConfigurationError)
async def database_configuration_error(
    _request: Request, exc: DatabaseConfigurationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


@app.exception_handler(PyMongoError)
async def database_operation_error(
    _request: Request, exc: PyMongoError
) -> JSONResponse:
    logger.error("MongoDB operation failed (%s)", type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "The database is temporarily unavailable."},
    )


def get_buildings_collection(
    database: Annotated[Database, Depends(get_database)],
) -> Collection:
    return database["buildings"]


def get_parking_collection(
    database: Annotated[Database, Depends(get_database)],
) -> Collection:
    return database["parkinglots"]


BuildingsCollection = Annotated[Collection, Depends(get_buildings_collection)]
ParkingCollection = Annotated[Collection, Depends(get_parking_collection)]


@app.get("/buildings", response_model=list[Building])
def get_buildings(collection: BuildingsCollection) -> list[dict]:
    return list(collection.find())


@app.post(
    "/buildings",
    response_model=Building,
    status_code=status.HTTP_201_CREATED,
)
def create_building(
    building: BuildingCreate,
    collection: BuildingsCollection,
) -> dict:
    document = building.model_dump()
    collection.insert_one(document)
    return document


@app.get("/buildings/{building_id}", response_model=Building)
def get_building(
    building_id: int,
    collection: BuildingsCollection,
) -> dict:
    building = collection.find_one({"buildingId": building_id})
    if building is None:
        raise HTTPException(status_code=404, detail="Building not found")
    return building


@app.get("/parking", response_model=list[ParkingLot])
def get_parking(collection: ParkingCollection) -> list[dict]:
    return list(collection.find())


@app.post(
    "/parking",
    response_model=ParkingLot,
    status_code=status.HTTP_201_CREATED,
)
def create_parking_lot(
    parking_lot: ParkingLotCreate,
    collection: ParkingCollection,
) -> dict:
    document = parking_lot.model_dump()
    collection.insert_one(document)
    return document


@app.get("/parking/{lot_id}", response_model=ParkingLot)
def get_parking_lot(
    lot_id: int,
    collection: ParkingCollection,
) -> dict:
    parking_lot = collection.find_one({"lotId": lot_id})
    if parking_lot is None:
        raise HTTPException(status_code=404, detail="Parking lot not found")
    return parking_lot


@app.get("/search", response_model=list[Building])
def search_buildings(
    collection: BuildingsCollection,
    q: Annotated[str, Query()] = "",
) -> list[dict]:
    query = q.casefold()
    return [
        building
        for building in collection.find()
        if query in building["name"].casefold()
    ]


@app.exception_handler(DuplicateKeyError)
async def duplicate_key_error(
    _request: Request, _exc: DuplicateKeyError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": "A record with that ID already exists."},
    )
