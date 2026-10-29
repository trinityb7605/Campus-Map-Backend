import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

from mainmongo import DatabaseConfigurationError, get_database
from schema import Building, BuildingCreate, BuildingDetails, CampusUnit, ParkingLot, ParkingLotCreate

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


def get_services_collection(
    database: Annotated[Database, Depends(get_database)],
) -> Collection:
    return database["university_services"]


def get_departments_collection(
    database: Annotated[Database, Depends(get_database)],
) -> Collection:
    return database["departments"]


BuildingsCollection = Annotated[Collection, Depends(get_buildings_collection)]
ParkingCollection = Annotated[Collection, Depends(get_parking_collection)]
ServicesCollection = Annotated[Collection, Depends(get_services_collection)]
DepartmentsCollection = Annotated[Collection, Depends(get_departments_collection)]


def to_unit(document: dict) -> dict:
    # Shape university_services/departments doc for the API
    return {
        "id": document["_id"],
        "name": document["name"],
        "category": document.get("category"),
        "buildingId": document.get("buildingId"),
        "room": document.get("room"),
        "description": document.get("description"),
        "phone": document.get("phone"),
        "email": document.get("email"),
        "hours": document.get("hours"),
        "hours_note": document.get("hours_note"),
        "webpage": document.get("official_website"),
        "catalog_webpage": document.get("catalog_website"),
    }


def to_units(documents) -> list[dict]:
    return [
        to_unit(document)
        for document in sorted(documents, key=lambda d: d["name"].casefold())
    ]


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


@app.get("/buildings/{building_id}/details", response_model=BuildingDetails)
def get_building_details(
    building_id: int,
    buildings: BuildingsCollection,
    services: ServicesCollection,
    departments: DepartmentsCollection,
) -> dict:
    # Building + services & departments located in it
    building = buildings.find_one({"buildingId": building_id})
    if building is None:
        raise HTTPException(status_code=404, detail="Building not found")

    query = {"buildingId": building_id}
    return {
        "building": {
            "id": building["buildingId"],
            "name": building["name"],
            "category": building["category"],
            "coordinates": building["coordinates"],
            "description": building.get("demo_description"),
        },
        "services": to_units(services.find(query)),
        "departments": to_units(departments.find(query)),
    }


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


@app.get("/services", response_model=list[CampusUnit])
def get_services(collection: ServicesCollection) -> list[dict]:
    return [to_unit(document) for document in collection.find()]


@app.get("/services/{service_id}", response_model=CampusUnit)
def get_service(service_id: str, collection: ServicesCollection) -> dict:
    service = collection.find_one({"_id": service_id})
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")
    return to_unit(service)


@app.get("/departments", response_model=list[CampusUnit])
def get_departments(collection: DepartmentsCollection) -> list[dict]:
    return [to_unit(document) for document in collection.find()]


@app.get("/departments/{department_id}", response_model=CampusUnit)
def get_department(department_id: str, collection: DepartmentsCollection) -> dict:
    department = collection.find_one({"_id": department_id})
    if department is None:
        raise HTTPException(status_code=404, detail="Department not found")
    return to_unit(department)


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
