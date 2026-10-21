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
