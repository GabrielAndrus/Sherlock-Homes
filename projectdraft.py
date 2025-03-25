"""Defining House"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from geopy.distance import geodesic
import pandas as pd
import numpy as np

house_data = pd.read_csv("real-estate-data.csv")

house_data.replace("N/A", pd.NA, inplace=True)
house_data_cleaned = house_data.dropna()
house_data_cleaned = house_data_cleaned.copy()

house_data_cleaned['DEN_new'] = house_data_cleaned['DEN'].map({'YES': True, 'no': False})
house_data_cleaned['parking_new'] = house_data_cleaned['parking'].map({'Yes': True, 'N': False})
house_data_cleaned['location'] = list(zip(house_data_cleaned['lt'], house_data_cleaned['lg']))
house_data_cleaned['size_range'] = house_data_cleaned['size'].apply(
    lambda x: tuple(map(int, x.replace(" sqft", "").split('-'))) if isinstance(x, str) and '-' in x else None
)
del house_data_cleaned['size']
del house_data_cleaned['parking']
del house_data_cleaned['lt']
del house_data_cleaned['lg']
del house_data_cleaned['DEN']
del house_data_cleaned['exposure']
del house_data_cleaned['D_mkt']
del house_data_cleaned['ward']


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

    def __init__(self, h_id: int, beds: int, baths: int, size: tuple[int, int], building_age: int,
                 maint: int, price: int, location: tuple[int, int], den=False, parking=False,) -> None:
        """Initialize a new house.
        """
        self.id = h_id
        self.beds = beds
        self.baths = baths
        self.DEN = den
        self.size = size
        self.parking = parking
        self.maint = maint
        self.building_age = building_age
        self.price = price
        self.location = location

    def generate_edge_weight(self, h2: House) -> float:
        """Generates a value based on the similarity of two houses."""

        # Feature similarity (beds, baths, sqft)
        features1 = [self.beds, self.baths, self.size[0], self.size[1]]
        features2 = [h2.beds, h2.baths, h2.size[0], h2.size[1]]

        features1 = np.array(features1, dtype=float)
        features2 = np.array(features2, dtype=float)

        feature_diff = np.linalg.norm(np.array(features1) - np.array(features2))
        feature_score = 1 / (1 + feature_diff)

        # Calculates location similarity
        location_dist = geodesic(self.location, h2.location).km
        location_score = 1 / (1 + location_dist)

        # Calculates maintenance cost similarity
        maint_diff = abs(self.maint - h2.maint)
        maint_score = 1 / (1 + maint_diff)

        # Calculates building age similarity
        age_diff = abs(self.building_age - h2.building_age)
        age_score = 1 / (1 + age_diff)

        total_weight = .4 * feature_score + .3 * location_score + .2 * maint_score + .1 * age_score

        return total_weight


class _Vertex:
    """A vertex in a graph.

    Instance Attributes:
        - item: The data stored in this vertex.
        - neighbours: The vertices that are adjacent to this vertex.

    Representation Invariants:
        - self not in self.neighbours
        - all(self in u.neighbours for u in self.neighbours)
    """
    item: int
    house_data: House
    neighbours: dict[_Vertex, float]

    def __init__(self, item: int, house: House, neighbours: {}) -> None:
        """Initialize a new vertex with the given item and neighbours."""
        self.item = item
        self.house_data = house
        self.neighbours = neighbours


class Graph:
    """A graph.

    Representation Invariants:
        - all(item == self._vertices[item].item for item in self._vertices)
    """
    # Private Instance Attributes:
    #     - _vertices:
    #         A collection of the vertices contained in this graph.
    #         Maps item to _Vertex object.
    _vertices: dict[int, _Vertex]

    def __init__(self) -> None:
        """Initialize an empty graph (no vertices or edges)."""
        self._vertices = {}

    def add_vertex(self, house: House, item: Any) -> None:
        """Add a vertex with the given item to this graph.

        The new vertex is not adjacent to any other vertices.

        Preconditions:
            - item not in self._vertices
        """
        if item not in self._vertices:
            self._vertices[item] = _Vertex(item, house, set())

    def add_edge(self, weight: float, item1: Any, item2: Any) -> None:
        """Add an edge between the two vertices with the given items in this graph.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - item1 != item2
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            v2 = self._vertices[item2]

            # Add the new edge
            v1.neighbours[v2] = weight
            v2.neighbours[v1] = weight
        else:
            # We didn't find an existing vertex for both items.
            raise ValueError


def load_houses(houses: pd.DataFrame) -> Graph:
    """Load graph"""

    houses_graph = Graph()

    house_list = [House(**row) for row in houses.to_dict(orient="records")]

    for v in house_list:
        houses_graph.add_vertex(v, v.id)

    for house1 in house_list:
        for house2 in house_list:
            weight = house1.generate_edge_weight(house2)
            if house1.id != house2.id and weight > 0.5:
                houses_graph.add_edge(weight, house1, house2)

    return houses_graph
