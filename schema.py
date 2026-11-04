from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated, Literal


class _CampusRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    coordinates: tuple[float, float]


class BuildingCreate(_CampusRecord):
    buildingId: int = Field(gt=0)
    demo_description: str | None = None
    images: list[str] = Field(default_factory=list)


class Building(_CampusRecord):
    model_config = ConfigDict(extra="ignore")
    
    buildingId: int
    demo_description: str | None = None
    images: list[str] = Field(default_factory=list)


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
    images: list[str] = Field(default_factory=list)


class BuildingDetails(BaseModel):
    # Expanded version of Buildings information
    building: BuildingSummary
    services: list[CampusUnit]
    departments: list[CampusUnit]

# ---- AI assistant (POST /chat) ----
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(min_length=1, max_length=1000)]


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    # Previous turns of this conversation (oldest first) so follow-up questions work
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class ChatSource(BaseModel):
    type: Literal["building", "department", "service", "parking"]
    id: str
    name: str
    buildingId: int | None = None  # lets the app open/zoom to this building


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
