"""Tests the functionality of the NER model using pre-determined input."""

from ner_model import *

user_input = "I want a house in ward 13 with bedrooms 8, baths 3 and has parking."

print(extract_housing_info(user_input))
