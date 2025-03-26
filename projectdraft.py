"""Defining House"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from geopy.distance import geodesic
import pandas as pd
import networkx as nx
from networkx.classes import neighbors
from plotly.graph_objs import Scatter, Figure
import numpy as np


def clean_houses_data(file: str) -> pd.DataFrame:
    """Cleans the house data file"""
    house_data = pd.read_csv(file)

    house_data.replace("N/A", pd.NA, inplace=True)
    house_data.replace("NA", pd.NA, inplace=True)
    house_data_cleaned = house_data.dropna()
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
        MAX_TORONTO_DISTANCE = 40  # Approximate max distance in km
        location_score = 1 - (location_dist / MAX_TORONTO_DISTANCE)

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
            v1.neighbours[v2] = weight
            v2.neighbours[v1] = weight
        else:
            # We didn't find an existing vertex for both items.
            raise ValueError

    def list_vertices(self) -> list[_Vertex]:
        """Returns a list of all vertices in the graph"""

        return list(self._vertices.values())

    def to_networkx(self, max_vertices: int = 5000) -> nx.Graph:
        """Convert this graph into a networkx Graph.

        max_vertices specifies the maximum number of vertices that can appear in the graph.
        (This is necessary to limit the visualization output for large graphs.)

        Note that this method is provided for you, and you shouldn't change it.
        """
        graph_nx = nx.Graph()

        for v in self._vertices.values():
            graph_nx.add_node(v.item, kind=getattr(v, 'kind', 'normal'))  # Default to 'normal'

            for u in v.neighbours:
                if graph_nx.number_of_nodes() < max_vertices:
                    graph_nx.add_node(u.item, kind=getattr(u, 'kind', 'normal'))

                if u.item in graph_nx.nodes:
                    graph_nx.add_edge(v.item, u.item)

            if graph_nx.number_of_nodes() >= max_vertices:
                break

        return graph_nx


def load_houses(houses: pd.DataFrame) -> Graph:
    """Load graph"""

    houses_graph = Graph()

    house_list = [House(**row) for row in houses.to_dict(orient="records")]

    for v in house_list:
        houses_graph.add_vertex(v, v.id)

    for house1 in house_list:
        for house2 in house_list:
            weight = house1.generate_edge_weight(house2)
            if house1.id != house2.id and weight > 0.7:
                houses_graph.add_edge(weight, house1.id, house2.id)

    return houses_graph

# Colours to use when visualizing different clusters.
COLOUR_SCHEME = [
    '#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A', '#B68100',
    '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1', '#FC0080', '#B2828D',
    '#6C7C32', '#778AAE', '#862A16', '#A777F1', '#620042', '#1616A7', '#DA60CA',
    '#6C4516', '#0D2A63', '#AF0038'
]

LINE_COLOUR = 'rgb(210,210,210)'
VERTEX_BORDER_COLOUR = 'rgb(50, 50, 50)'
HOUSE_COLOUR = 'rgb(89, 205, 105)'  # Green for all nodes
USER_COLOUR = 'rgb(105, 89, 205)'


def knn_model(user_house: House, house_graph: Graph) -> _Vertex:
    """Turns user_input into a node and finds its nearest neighbours from the graph"""
    user_house_vertex = _Vertex(user_house.id, user_house, {})

    for vertex in house_graph.list_vertices():
        weight = vertex.house_data.generate_edge_weight(user_house)
        if weight > .75:
            user_house_vertex.neighbours[vertex] = weight

    return user_house_vertex


def load_user_graph(user_vertex: _Vertex) -> Graph:
    """loads graph of houses nearest to users requests"""
    graph = Graph()
    graph.add_vertex(user_vertex.house_data, user_vertex.item)

    # Add all nearest neighbors
    for neighbor, weight in user_vertex.neighbours.items():
        graph.add_vertex(neighbor.house_data, neighbor.item)

    # Connect only the user vertex to its nearest neighbors
    for neighbor, weight in user_vertex.neighbours.items():
        graph.add_edge(weight, user_vertex.item, neighbor.item)

        # Also connect neighbors to each other if they were originally connected
        for second_neighbor, second_weight in neighbor.neighbours.items():
            if second_neighbor in user_vertex.neighbours:
                graph.add_edge(second_weight, neighbor.item, second_neighbor.item)

    return graph


def lat_lng_map(houses: pd.DataFrame) -> dict[int: tuple[float, float]]:
    """generates a mapping of house to location"""
    return {row['h_id']: row['location'] for _, row in houses.iterrows()}


def visualize_graph(graph: Graph,
                    lat_long_map: dict[str, tuple[float, float]],
                    user_vertex: int = 1,  # Add user_vertex as a parameter
                    max_vertices: int = 5000,
                    output_file: str = '') -> None:
    """Visualize the given graph using lat-long coordinates for node placement.

    - lat_long_map: Dictionary mapping vertex names to (latitude, longitude).
    - user_vertex: The vertex representing the user's house (highlighted).
    - max_vertices: The maximum number of vertices to visualize.
    - output_file: Filename to save the output image (optional).
    """
    graph_nx = graph.to_networkx(max_vertices)

    # Extract node positions and labels
    x_values = []
    y_values = []
    labels = []
    colors = []  # To store colors for nodes

    for node in graph_nx.nodes:
        if node in lat_long_map:
            lat, lon = lat_long_map[node]
            x_values.append(lon)  # Longitude as x-axis
            y_values.append(lat)  # Latitude as y-axis
            labels.append(node)

            # Set color: red for user vertex, blue for others
            if node == user_vertex:
                colors.append("red")  # User vertex is red
            else:
                colors.append("blue")  # All others are blue

    # Prepare edge data
    x_edges = []
    y_edges = []
    for edge in graph_nx.edges:
        if edge[0] in lat_long_map and edge[1] in lat_long_map:
            x0, y0 = lat_long_map[edge[0]][1], lat_long_map[edge[0]][0]  # (lon, lat)
            x1, y1 = lat_long_map[edge[1]][1], lat_long_map[edge[1]][0]  # (lon, lat)

            x_edges += [x0, x1, None]  # None creates line breaks
            y_edges += [y0, y1, None]

    # Edge trace
    trace_edges = Scatter(
        x=x_edges,
        y=y_edges,
        mode='lines',
        line=dict(color='lightblue', width=0.5),
        hoverinfo='none',
    )

    # Node trace
    trace_nodes = Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(
            size=5,
            color=colors,  # Apply the color array
            line=dict(color='black', width=1)
        ),
        text=labels,
        hovertemplate='%{text}',
    )

    # Create figure
    fig = Figure(data=[trace_edges, trace_nodes])

    # Set layout
    fig.update_layout(
        title="Graph Visualization with Lat-Long",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        width=1000,
        height=800,
        showlegend=False,
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgrey"),  # Faint grid
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")   # Faint grid
    )

    # Display or save
    if output_file == '':
        fig.show()
    else:
        fig.write_image(output_file)


house_data = clean_houses_data("real-estate-data.csv")
loc_map = lat_lng_map(house_data)
house_graph = load_houses(house_data)
user_house = House(1, 2, 2, (500,999), 1, 767, 838000,
                   (43.6204466596335, -79.37543575581886), True, True)
user_vertex = knn_model(user_house, house_graph)
user_graph = load_user_graph(user_vertex)

# visualize_graph(house_graph, loc_map)
loc_map[1] = user_house.location
visualize_graph(user_graph, loc_map)
