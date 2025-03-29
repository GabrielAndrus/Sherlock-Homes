"""Defining House"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from geopy.distance import geodesic
import pandas as pd
import folium
import webbrowser
import numpy as np


def clean_houses_data(file: str) -> pd.DataFrame:
    """Cleans the house data file"""
    houses_data = pd.read_csv(file)

    houses_data.replace("N/A", pd.NA, inplace=True)
    houses_data.replace("NA", pd.NA, inplace=True)
    house_data_cleaned = houses_data.dropna()
    house_data_cleaned = house_data_cleaned.copy()

    house_data_cleaned['DEN'] = house_data_cleaned['DEN'].map({'YES': True, 'no': False})
    house_data_cleaned['parking'] = house_data_cleaned['parking'].map({'Yes': True, 'N': False})
    house_data_cleaned['location'] = list(zip(house_data_cleaned['lt'], house_data_cleaned['lg']))
    house_data_cleaned['size'] = house_data_cleaned['size'].apply(
        lambda x: tuple(map(int, x.replace(" sqft", "").split('-'))) if isinstance(x, str) and '-' in x else None
    )

    del house_data_cleaned['lt']
    del house_data_cleaned['lg']
    del house_data_cleaned['exposure']
    del house_data_cleaned['D_mkt']
    del house_data_cleaned['ward']

    return house_data_cleaned.dropna()[:501]


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
    location: tuple[float, float]

    def __init__(self, h_id: int, beds: int, baths: int, size: tuple[int, int], building_age: int,
                 maint: int, price: int, location: tuple[float, float], DEN=False, parking=False,) -> None:
        """Initialize a new house.
        """
        self.id = h_id
        self.beds = beds
        self.baths = baths
        self.DEN = DEN
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
        alpha = 0.1  # Adjust scaling
        location_score = np.exp(-alpha * location_dist)

        # Calculates maintenance cost similarity
        maint_diff = abs(self.maint - h2.maint)
        maint_score = 1 / (1 + maint_diff)

        # Calculates building age similarity
        age_diff = abs(self.building_age - h2.building_age)
        age_score = 1 / (1 + age_diff)

        total_weight = .35 * feature_score + .35 * location_score + .2 * maint_score + .1 * age_score

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
    neighbours: dict[int, float]

    def __init__(self, item: int, house: House, neighbours: {}) -> None:
        """Initialize a new vertex with the given item and neighbours."""
        self.item = item
        self.house_data = house
        self.neighbours = neighbours

    def return_neighbors(self) -> set[tuple[int, int]]:
        """Returns the pairs of neighbours of this vertex."""

        pairs = set()
        neighbors = list(self.neighbours.keys())

        # Connect to the vertex itself (if you want to include self as "pair")
        # pairs.add((min(self.item, self.item), max(self.item, self.item)))  # (a,a)

        # Create all neighbor pairs
        for neighbor in neighbors:
            pairs.add((min(self.item, neighbor), max(self.item, neighbor)))

        return pairs


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
            self._vertices[item] = _Vertex(item, house, {})

    def add_edge(self, weight: float, item1: Any, item2: Any) -> None:
        """Add an edge between the two vertices with the given items in this graph.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - item1 != item2
        """
        all_ids = set(self._vertices.keys())

        if item1 in all_ids and item2 in all_ids:
            v1 = self._vertices[item1]
            v2 = self._vertices[item2]

            # Add the new edge
            v1.neighbours[v2.item] = weight
            v2.neighbours[v1.item] = weight
        else:
            # We didn't find an existing vertex for both items.
            raise ValueError

    def list_vertices(self) -> list[_Vertex]:
        """Returns a list of all vertices in the graph"""

        return list(self._vertices.values())

    def return_adjacent_pairs(self) -> set[tuple[int, int]]:
        """Returns all adjacent pairs in this graph"""
        things = set()

        for vertex in self._vertices:
            things = things.union(self._vertices[vertex].return_neighbors())

        return things

    def get_vertex(self, item: int) -> _Vertex:
        """Return the vertex associated with the given item.

        Raise a KeyError if the item does not exist in the graph.
        """
        if item in self._vertices:
            return self._vertices[item]
        else:
            raise KeyError(f"Vertex with item {item} not found in graph.")

    def get_all_vertices(self) -> list[_Vertex]:
        """Return a list of all vertices in the graph."""
        return list(self._vertices.values())

    def get_all_ids(self) -> list[int]:
        """Return a list of all vertices in the graph."""
        return list(self._vertices.keys())


def load_houses(houses: pd.DataFrame) -> Graph:
    """Load graph"""

    houses_graph = Graph()

    house_list = [House(**row) for row in houses.to_dict(orient="records")]

    for v in house_list:
        houses_graph.add_vertex(v, v.id)

    for house1 in house_list:
        for house2 in house_list:
            weight = house1.generate_edge_weight(house2)
            if house1.id != house2.id and weight > 0.69:
                houses_graph.add_edge(weight, house1.id, house2.id)

    return houses_graph


def knn_model(users_house: House, houses_graph: Graph) -> _Vertex:
    """Turns user_input into a node and finds its nearest neighbours from the graph"""
    user_house_vertex = _Vertex(users_house.id, users_house, {})

    for vertex in houses_graph.list_vertices():
        weight = vertex.house_data.generate_edge_weight(users_house)
        if weight > .69:
            user_house_vertex.neighbours[vertex.item] = weight

    return user_house_vertex


def load_user_graph(users_vertex: _Vertex, houses_graph: Graph) -> Graph:
    """loads graph of houses nearest to users requests"""
    graph = Graph()

    # Add user vertex first
    graph.add_vertex(users_vertex.house_data, users_vertex.item)

    # First pass: add all vertices (user + neighbors)
    for neighbor, _ in users_vertex.neighbours.items():
        neighbor_vertex = houses_graph.get_vertex(neighbor)
        graph.add_vertex(neighbor_vertex.house_data, neighbor)

    # Second pass: connect user to neighbors
    for neighbor, weight in users_vertex.neighbours.items():
        graph.add_edge(weight, users_vertex.item, neighbor)

        # Connect neighbors to each other if they were originally connected
        neighbor_vertex = houses_graph.get_vertex(neighbor)
        for second_neighbor, second_weight in neighbor_vertex.neighbours.items():
            if second_neighbor in users_vertex.neighbours:
                try:
                    graph.add_edge(second_weight, neighbor, second_neighbor)
                except ValueError:
                    # Skip if edge already exists or vertices missing
                    continue

    return graph


def find_average_price(users_graph: Graph) -> float:
    """Returns the average house price"""
    houses = users_graph.get_all_vertices()
    prices_so_far = 0

    if houses == 1:
        print("No houses found matching user input")
        return 0

    for house in houses:
        if house.item != 1:
            prices_so_far += house.house_data.price

    return prices_so_far/(len(houses)-1)


def lat_lng_map(houses: pd.DataFrame) -> dict[int: tuple[float, float]]:
    """generates a mapping of house to location"""
    return {row['h_id']: row['location'] for _, row in houses.iterrows()}


def load_map(location_map: dict[int: tuple[float, float]], houses_graph: Graph):
    """loads map"""

    my_map1 = folium.Map(location=[43.66579167224076, -79.38951447651665],
                         zoom_start=12)

    for _id in location_map:
        loc = location_map[_id]

        folium.Marker([loc[0], loc[1]],
                      popup=loc, icon=folium.Icon(color="blue", icon="home", prefix="fa")).add_to(my_map1)

    '''all_pairs = houses_graph.return_adjacent_pairs()

    for pair in all_pairs:
        folium.PolyLine(locations=[location_map[pair[0]], location_map[pair[1]]], weight=1,
                        color="#2E8B57", line_opacity=0.15).add_to(my_map1)'''

    my_map1.save("my_map1.html")
    webbrowser.open("my_map1.html")


def load_recommended_map(location_map: dict[int: tuple[float, float]], houses_graph: Graph):
    """loads map"""

    my_map2 = folium.Map(location=[43.66579167224076, -79.38951447651665],
                         zoom_start=12)

    vertices = houses_graph.get_all_vertices()

    user_loc = location_map[1]

    folium.Marker(
        location=[user_loc[0], user_loc[1]],
        popup="User",
        icon=folium.Icon(color="red", icon="home", prefix="fa")  # Change "home" to other icons if needed
    ).add_to(my_map2)

    for vertex in vertices:
        if vertex.house_data.id != 1:
            loc = vertex.house_data.location

            popup_text = f"""
            Price: ${vertex.house_data.price}<br>
            Beds: {int(vertex.house_data.beds)}<br>
            Baths: {vertex.house_data.baths}<br>
            Size: {vertex.house_data.size}<br>
            Parking: {vertex.house_data.parking}<br>
            Location: {vertex.house_data.location}
            """

            folium.Marker([loc[0], loc[1]], popup=popup_text,
                          icon=folium.Icon(color="blue", icon="home", prefix="fa")).add_to(my_map2)

    all_pairs = houses_graph.return_adjacent_pairs()

    # iterate through vertices to plot all markers with price and id

    for pair in all_pairs:

        folium.PolyLine(locations=[location_map[pair[0]], location_map[pair[1]]], weight=1,
                        color="#2E8B57", line_opacity=0.15).add_to(my_map2)

    my_map2.save("my_map2.html")
    webbrowser.open("my_map2.html")


house_data = clean_houses_data("real-estate-data.csv")
loc_map = lat_lng_map(house_data)
house_graph = load_houses(house_data)

user_house = House(1, 2, 2, (500, 999), 1, 767, 838000,
                   (43.634466596335, -79.42543575581886), True, True)
user_vertex = knn_model(user_house, house_graph)
user_graph = load_user_graph(user_vertex, house_graph)

loc_map[1] = user_house.location
load_map(loc_map, house_graph)
load_recommended_map(loc_map, user_graph)

loc_map[1] = user_house.location
load_map(loc_map, house_graph)
load_recommended_map(loc_map, user_graph)

