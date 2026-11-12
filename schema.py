from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated, Literal

# share common fields & reject fields API doesn't expect
class _CampusRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    # require name & category; store map coordinates as 2 numbers
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    coordinates: tuple[float, float]

# validate new building before saving
class BuildingCreate(_CampusRecord):
    buildingId: int = Field(gt=0)
    demo_description: str | None = None
    # create separate empty list for each building
    images: list[str] = Field(default_factory=list)

# describe building returned by API, allowing extra database fields
class Building(_CampusRecord):
    model_config = ConfigDict(extra="ignore")
    
    buildingId: int
    demo_description: str | None = None
    images: list[str] = Field(default_factory=list)

# validate new parking lot & require positive ID
class ParkingLotCreate(_CampusRecord):
    lotId: int = Field(gt=0)
    facultyStaff: bool = False
    commuter: bool = False
    visitor: bool = False
    residenceHall: bool = False
    hourly: bool = False

# describe parking lot returned by API
class ParkingLot(_CampusRecord):
    model_config = ConfigDict(extra="ignore")
    
    lotId: int
    facultyStaff: bool = False
    commuter: bool = False
    visitor: bool = False
    residenceHall: bool = False
    hourly: bool = False

# describe department/service shown inside building
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

# hold compact building information returned with its units
class BuildingSummary(BaseModel):
    # use buildingId as ID in details response
    id: int  # same as buildingId
    name: str
    category: str
    coordinates: tuple[float, float]
    description: str | None = None
    images: list[str] = Field(default_factory=list)

# combine building with departments/services
class BuildingDetails(BaseModel):
    building: BuildingSummary
    services: list[CampusUnit]
    departments: list[CampusUnit]

# store 1 user or assistant message in chat history
class ChatMessage(BaseModel):
    # only these two roles are allowed, and message text can't be blank
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(min_length=1, max_length=1000)]

# validate chat question & optional earlier messages
class ChatRequest(BaseModel):
    # trim whitespace and limit the question size
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    # Previous turns of this conversation (oldest first) so follow-up questions work
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

# describe campus record chat answer used
class ChatSource(BaseModel):
    # limit sources to record types the app knows how to display
    type: Literal["building", "department", "service", "parking"]
    id: str
    name: str
    buildingId: int | None = None  # lets the app open/zoom to this building

# shape answer & optional source list for API response
class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)