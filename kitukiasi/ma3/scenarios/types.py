from __future__ import annotations

from pydantic import BaseModel, Field


class Position(BaseModel):
    longitude: float
    latitude: float


class Trip(BaseModel):
    departure: int
    origin: dict = Field(..., alias="origin")
    destination: dict = Field(..., alias="destination")
    mode: str
    purpose: str


class Person(BaseModel):
    trips: list[Trip]


class Scenario(BaseModel):
    scenario_name: str
    map_name: str
    people: list[Person]


def position_obj(lon: float, lat: float) -> dict:
    return {"Position": {"longitude": lon, "latitude": lat}}
