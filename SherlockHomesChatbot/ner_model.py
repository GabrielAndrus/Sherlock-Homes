"""This file contains the Named Entity Recognition model (NER) for Sherlock Homes' chatbot.
It uses SpaCy's built-in natural language processing to chunk user_input into different "entities," from which
the model extracts information and assigns it to a dictionary containing important housing information.
This feature has been hard-coded to ensure fine-tuning; instead of using some sort of API or ChatGPT wrapper,
we wanted the model to be limited only to relevant information about housing, as per our project.

Implemented by: Gabriel Andrus, 3/25/25.
"""

import spacy
import re

nlp = spacy.load('en_core_web_sm')

# Decimal pattern for latitude and longitude
DECIMAL_PATTERN = r"""
        (-?\d{1,3}\.\d+)\s*,\s*  # Latitude (e.g., 40.7128)
        (-?\d{1,3}\.\d+)          # Longitude (e.g., -74.0060)
    """

# Important entities
extracted = {
    'ward': None,
    'bedrooms': None,
    'bathrooms': None,
    'size': None,
    'parking': None,
    'building_age': None,
    'maint': None,
    'price_min': None,
    'price_max': None,
    'Lt': None,
    'Lg': None
}


def extract_entities(user_input):
    """Extract pimportant information from user input.

    Important information: Ward, Number of beds, Baths, Size, Parking,
    Building_age, Maint, Price, Lt, Lg
    """
    doc = nlp(user_input)  # Create document from user_input

    user_entities = doc.ents

    valid_wards = []
    for ent in user_entities:
        if ent.label_ == 'WARD' and ent.text.upper().startswith('W') and ent.text[1:].isdigit():
            valid_wards.append(ent.text)
    extracted['ward'] = valid_wards[0]  # First ward mention

    # Get bedroom matches
    valid_beds = []
    bed_entries = {'bedrooms', 'bed', 'room', 'br,' 'beds'}
    for ent in user_entities:
        if ent.label in bed_entries:
            valid_beds.append(ent.text)
    if valid_beds:
        extracted['bedrooms'] = valid_beds[0]  # First instance of bed terminology

    valid_bathrooms = []
    bed_entries = {'bathrooms', 'bath', 'room', 'br,' 'baths'}
    for ent in user_entities:
        if ent.label in bed_entries:
            valid_bathrooms.append(ent.text)
    extracted['bathrooms'] = valid_bathrooms[0]

    extract_size(user_entities)  # Extract size

    extract_parking(user_entities)  # Extract parking

    extract_building_age(user_entities)  # Extract information about building_age

    extract_maint(user_entities)  # Extract information about maintenance cost

    coordinates = extract_location(user_input)  # Extract location
    extracted['Lt'] = coordinates[0]
    extracted['Lg'] = coordinates[1]


def extract_size(user_entities, min_size=0, max_size=2000):
    """Extract size information from user input."""

    for ent in user_entities:
        if ent.label == 'SIZE':
            size_match = re.search(r'\b(\d+)\b', ent.text)
            if size_match:
                size = int(size_match.group(1))
                if min_size <= size <= max_size:
                    extracted['size'] = size


def extract_parking(user_entities):
    """Extract parking information from user input, based on whether a positive, parking-related term is included."""
    # Positive phrases

    positive_terms = {r"\bhas parking\b",
                      r"\bwith parking\b",
                      r"\bparking (?:available|included|provided)\b",
                      r"\b(?:free|private|dedicated) parking\b"
                      }

    negative_terms = {r"\bno parking\b",
                      r"\bwithout garage\b"}

    for ent in user_entities:
        if ent.label == 'PARKING':
            if ent.text in positive_terms:
                extracted['parking'] = False
            elif ent.text in negative_terms:
                extracted['parking'] = True
            else:
                extracted['parking'] = False


def extract_building_age(user_entities):
    """Extract building age from user input."""
    for ent in user_entities:
        if ent.label_ == 'BUILDING_AGE':
            age_match = re.search(r"\b\d+\b", ent.text)
            if age_match:
                extracted['building_age'] = int(age_match.group())
        else:
            return None


def extract_maint(user_entities, min_maint=0, max_maint=3000):
    """Extract maint from user input."""
    for ent in user_entities:
        if ent.label_ == 'MAINT':
            maint_match = re.search(r"\b\d+\b", ent.text)
            if maint_match:
                maintenance_price = int(maint_match.group())
                if min_maint <= maintenance_price <= max_maint:
                    extracted['maint'] = maintenance_price


def extract_location(text):
    """Extract location information from user input."""

    decimal_match = re.search(DECIMAL_PATTERN, text, re.VERBOSE)

    if decimal_match:
        Lt, Lg = map(float, decimal_match.groups())
        return (Lt, Lg)

    return None
