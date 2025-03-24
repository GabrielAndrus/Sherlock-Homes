"""Defining House"""
from dataclasses import dataclass


@dataclass
class House:
    """House data class"""
    id: int
    beds: int
    baths: int
    DEN: bool
    size: tuple[int, int]
    parking: bool
    building_age: int
    maint: int
    price: int
    location: tuple[int, int]

    def __init__(self, id: int, beds: int, baths: int, size: tuple[int, int], building_age: int,
                 maint: int, price: int, location: tuple[int, int], den=False, parking=False,) -> None:
        """Initialize a new house.
        """
        self.int = int
        self.beds = beds
        self.baths = baths
        self.DEN = den
        self.size = size
        self.parking = parking
        self.maint = maint
        self.building_age = building_age
        self.price = price
        self.location = location

