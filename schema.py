from pydantic import BaseModel, ConfigDict, Field


class _CampusRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    coordinates: tuple[float, float]


class BuildingCreate(_CampusRecord):
    buildingId: int = Field(gt=0)
    demo_description: str | None = None


class Building(_CampusRecord):
    model_config = ConfigDict(extra="ignore")
    
    buildingId: int
    demo_description: str | None = None


class ParkingLotCreate(_CampusRecord):
    lotId: int = Field(gt=0)


class ParkingLot(_CampusRecord):
    model_config = ConfigDict(extra="ignore")
    
    lotId: int


class CampusUnit(BaseModel):
    """ University service or Academic department/school
        Don't get own map marker; only shown inside designated 
        building `buildingId` points """
    id: str
    name: str
    category: str | None = None
    buildingId: int | None = None
    room: str | None = None
    description: str | None = None
    phone: str | None = None
    email: str | None = None
    hours: dict[str, str] | None = None
    hours_note: str | None = None
    webpage: str | None = None
    catalog_webpage: str | None = None


# Used for storing university services & departments in buildings
class BuildingSummary(BaseModel):
    # Compact version of Buildings information
    id: int  # same as buildingId
    name: str
    category: str
    coordinates: tuple[float, float]
    description: str | None = None


class BuildingDetails(BaseModel):
    # Expanded version of Buildings information
    building: BuildingSummary
    services: list[CampusUnit]
    departments: list[CampusUnit]