"""Defining House Daniel please update docstring/description for this"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from geopy.distance import geodesic
import pandas as pd
import folium
import webbrowser
import numpy as np
import os

import ner_model
from SherlockChatbot import *
from SherlockChatbot import user_preferences


def clean_houses_data(file: str) -> pd.DataFrame:
    """Cleans the house data file"""
    houses_data = pd.read_csv(file)

    # Handle NA

    houses_data.replace(["N/A", "NA", "NaN", "null", ""], pd.NA, inplace=True)

    # Handle parking and den

    houses_data['DEN'] = houses_data['DEN'].map({'YES': True, 'no': False}).fillna(False)
    houses_data['parking'] = houses_data['parking'].map({'Yes': True, 'N': False}).fillna(False)

    # Handle location

    houses_data['lt'] = pd.to_numeric(houses_data['lt'], errors='coerce').fillna(43.7)
    houses_data['lg'] = pd.to_numeric(houses_data['lg'], errors='coerce').fillna(-79.4)
    houses_data['location'] = list(zip(houses_data['lt'], houses_data['lg']))

    # Handle size

    def parse_size(size_str):
        """Parses size from data set"""
        try:
            if isinstance(size_str, str):
                nums = re.findall(r'\d+', size_str.replace(",", ""))
                if len(nums) >= 2:
                    return (int(nums[0]), int(nums[1]))
                elif len(nums) == 1:
                    val = int(nums[0])
                    return (val, val)
        except (ValueError, TypeError):
            pass
        return (800, 1200)

    houses_data['size'] = houses_data['size'].apply(parse_size)

    # Handle numeric fields
    numeric_cols = ['maint', 'building_age', 'price', 'beds', 'baths']
    for col in numeric_cols:
        houses_data[col] = pd.to_numeric(houses_data[col], errors='coerce')

    cols_to_drop = ['lt', 'lg', 'exposure', 'D_mkt', 'ward']
    house_data_cleaned = houses_data.drop(columns=[c for c in cols_to_drop if c in houses_data.columns])
    house_data_cleaned = house_data_cleaned.dropna(subset=['beds', 'baths', 'price', 'location'])[:501]

    return house_data_cleaned


@dataclass
class House:
    """House data class"""
    id: 1
    beds: int | None
    baths: int | None
    DEN: bool | None
    size: tuple[int, int] | None
    parking: bool | None
    building_age: int | None
    maint: int | None
    price: int | None
    location: tuple[float, float] | None

    def __init__(self, h_id=1, beds: int | None = None, baths: int | None = None, size: tuple[int, int] | None = None,
                 building_age: int | None = None,
                 maint: int | None = None, price: int | None = None, location: tuple[float, float] | None = None,
                 DEN: bool | None = None,
                 parking: bool | None = None, **kwargs) -> None:
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
        self.location = location if location is not None else (43.66579167224076, -79.38951447651665)

    def generate_edge_weight(self, h2: House) -> float:
        """Generates a value based on the similarity of two houses."""

        def safe_compare(a, b, default=0):
            """Safely compare None/missing values"""
            if a is None and b is None:
                return default

            if a is None or b is None:
                return 0

            # If comparing two numeric none values

            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return abs(a - b)

            # If comparing two string none values

            if isinstance(a, str) and isinstance(b, str):
                return 0 if a.lower == b.lower else default

            # If comparing multiple types of none values

            try:
                num_a = float(a) if not isinstance(a, (int, float)) else a
                num_b = float(b) if not isinstance(b, (int, float)) else b
                return abs(num_a - num_b)
            except (ValueError, TypeError):
                # Fallback to string comparison
                str_a = str(a).lower()
                str_b = str(b).lower()
                return 0 if str_a == str_b else default

        # Calculate difference between sizes
        size_diff = 0
        if self.size is not None and h2.size is not None:
            size_diff = (abs(self.size[0] - h2.size[0]) + (abs(self.size[1] - h2.size[1])))
        elif self.size is not None or h2.size is not None:
            size_diff = 1000

        beds_diff = safe_compare(self.beds, h2.beds, default=0)
        baths_diff = safe_compare(self.baths, h2.baths, default=0)
        feature_diff = beds_diff + baths_diff + (size_diff / 1000)

        max_possible_diff = 5 + 5 + 2
        feature_score = 1 - (feature_diff / max_possible_diff)

        if self.location is None or h2.location is None:
            location_score = 0.5  # if argument missing
        else:
            try:
                location_dist = geodesic(self.location, h2.location).kilometers
                alpha = 0.1  # Adjust scaling
                location_score = np.exp(-alpha * float(location_dist))
            except (ValueError, TypeError):
                location_score = 0.5

        maint_diff = safe_compare(self.maint, h2.maint, default=1000)
        maint_score = 1 / (1 + (maint_diff / 1000))

        age_diff = safe_compare(self.building_age, h2.building_age, default=20)
        age_score = 1/ (1 + (age_diff / 20))

        den_score = 1.0 if (self.DEN == h2.DEN) or (self.DEN is None or h2.DEN is None) else 0.7
        parking_score = 1.0 if (self.parking == h2.parking) or (self.parking is None or h2.parking is None) else 0.7

        total_weight = (
                0.4 * feature_score +
                0.4 * location_score +
                0.1 * maint_score +
                0.05 * age_score +
                0.025 * den_score +
                0.025 * parking_score
        )

        return max(0, min(1, total_weight))


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
    """Load graph with more inclusive threshold"""
    houses_graph = Graph()
    house_list = [House(**row) for row in houses.to_dict(orient="records")]

    for v in house_list:
        houses_graph.add_vertex(v, v.id)

    for house1 in house_list:
        for house2 in house_list:
            if house1.id != house2.id:
                weight = house1.generate_edge_weight(house2)
                if weight > 0.5:  # Lowered from 0.69 to 0.5
                    houses_graph.add_edge(weight, house1.id, house2.id)

    print(f"Loaded {len(house_list)} houses with {len(houses_graph.return_adjacent_pairs())} connections")
    return houses_graph


def knn_model(users_house: House, houses_graph: Graph) -> _Vertex:
    user_house_vertex = _Vertex(users_house.id, users_house, {})
    similarities = []

    for vertex in houses_graph.list_vertices():
        weight = vertex.house_data.generate_edge_weight(users_house)
        similarities.append((vertex.item, weight))

    # Sort by similarity and return y most similar

    similarities.sort(key=lambda x: x[1], reverse=True)
    for item, weight in similarities[:num_results]:
        user_house_vertex.neighbours[item] = weight

    print(f"Found {len(user_house_vertex.neighbours)} most similar houses")
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

    return prices_so_far / (len(houses) - 1)


def lat_lng_map(houses: pd.DataFrame) -> dict[int: tuple[float, float]]:
    """generates a mapping of house to location"""
    return {row['h_id']: row['location'] for _, row in houses.iterrows()}


def load_map(location_map: dict[int: tuple[float, float]], houses_graph: Graph):
    """loads map"""

    my_map1 = folium.Map(location=[43.66579167224076, -79.38951447651665],
                         zoom_start=12)

    for _id in location_map:
        loc = location_map[_id]
        if loc is None:
            continue

        folium.Marker([loc[0], loc[1]],
                      popup=loc, icon=folium.Icon(color="blue", icon="home", prefix="fa")).add_to(my_map1)

    all_pairs = houses_graph.return_adjacent_pairs()

    for pair in all_pairs:
        folium.PolyLine(locations=[location_map[pair[0]], location_map[pair[1]]], weight=1,
                        color="#2E8B57", line_opacity=0.15).add_to(my_map1)

    my_map1.save("my_map1.html")
    map_path = os.path.abspath("my_map1.html")
    webbrowser.open(f'file://{map_path}')


def load_recommended_map(location_map: dict[int, tuple[float, float]], houses_graph: Graph):
    """Loads map with better debugging"""
    try:
        # Create map centered on user location
        user_loc = location_map.get(1, (43.66579167224076, -79.38951447651665))
        my_map = folium.Map(location=[user_loc[0], user_loc[1]], zoom_start=13)

        # Add user marker
        folium.Marker(
            [user_loc[0], user_loc[1]],
            popup="Your Preferences",
            icon=folium.Icon(color='red', icon='home')
        ).add_to(my_map)

        # Add recommended houses
        vertices = houses_graph.get_all_vertices()
        houses_added = 0

        for vertex in vertices:
            if vertex.house_data.id == 1:  # Skip user vertex
                continue

            loc = vertex.house_data.location
            if not loc or len(loc) < 2:
                continue

            try:
                popup_text = f"""
                <b>House #{vertex.house_data.id}</b><br>
                Price: ${vertex.house_data.price:,}<br>
                Beds: {vertex.house_data.beds}<br>
                Baths: {vertex.house_data.baths}<br>
                Size: {vertex.house_data.size[0]}-{vertex.house_data.size[1]} sqft
                """

                folium.Marker(
                    [loc[0], loc[1]],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color='blue', icon='home')
                ).add_to(my_map)
                houses_added += 1
            except Exception as e:
                print(f"Skipping house {vertex.house_data.id}: {str(e)}")

        print(f"Added {houses_added} houses to the map")

        # Save and open
        map_path = "recommendations_map.html"
        my_map.save(map_path)
        webbrowser.open(f'file://{os.path.abspath(map_path)}')

    except Exception as e:
        print(f"Failed to create map: {str(e)}")

house_data = clean_houses_data("real-estate-data.csv")
print('Done cleaning data.')
loc_map = lat_lng_map(house_data)
print('Done making map.')
house_graph = load_houses(house_data)
print('Done loading house graph.')

user_house = House(1, beds=user_preferences.get('bedrooms'),
                   baths=user_preferences.get('bathrooms'),
                   size=user_preferences.get('size'),
                   building_age=user_preferences.get('building_age'),
                   maint=user_preferences.get('maint'),
                   price=user_preferences.get('price'),
                   location=user_preferences.get('location'),
                   DEN=user_preferences.get('den'),
                   parking=user_preferences.get('parking'))
# user_house = House(1, 2, 2, (500, 999), 1, 767, 838000,
#                  (43.634466596335, -79.42543575581886), True, True)

user_vertex = knn_model(user_house, house_graph)
user_graph = load_user_graph(user_vertex, house_graph)

loc_map[1] = user_house.location
load_map(loc_map, house_graph)
load_recommended_map(loc_map, user_graph)

loc_map[1] = user_house.location
load_map(loc_map, house_graph)
load_recommended_map(loc_map, user_graph)
