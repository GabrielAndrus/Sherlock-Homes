"""This file contains the Named Entity Recognition model (NER) for Sherlock Homes' chatbot.
It uses SpaCy's built-in natural language processing to chunk user_input into different "entities," from which
the model extracts information and assigns it to a dictionary containing important housing information.
This feature has been hard-coded to ensure fine-tuning; instead of using some sort of API or ChatGPT wrapper,
we wanted the model to be limited only to relevant information about housing, as per our project.

Using this chatbot requires that the user downloads spacy locally:
Find the tutorial here: https://spacy.io/
For mac users, simply type the following into terminal:
pip install spacy
python -m spacy download en_core_web_sm
OR import spacy via python settings.

Citations: DeepSeek AI generated the patterns to add to the matcher (lines 25-36), but the rest of the code is mine.

Implemented by: Gabriel Andrus, Deepankar Garg. 3/25/25.
"""

import spacy
from spacy.matcher import Matcher
import re

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Add all relevant housing language patterns to the matcher.
matcher.add("WARD", [[{"LOWER": "ward"}, {"IS_DIGIT": True}]])
matcher.add("WARD", [[{"TEXT": {"REGEX": "^W\d+$"}}]])
matcher.add("BEDROOMS", [[{"IS_DIGIT": True}, {"LOWER": {"IN": ["bedrooms", "bedroom", "beds", "bed"]}}]])
matcher.add("BEDROOMS", [[{"LOWER": {"IN": ["bedroom", "br"]}}, {"IS_DIGIT": True}]])
matcher.add("BATHROOMS", [[{"IS_DIGIT": True}, {"LOWER": {"IN": ["bathrooms", "bathroom", "baths", "bath"]}}]])
matcher.add("BATHROOMS", [[{"LOWER": {"IN": ["bathroom", "ba"]}}, {"IS_DIGIT": True}]])
matcher.add("SIZE", [[{"IS_DIGIT": True}, {"LOWER": {"IN": ["sqft", "sq"]}}]])
matcher.add("PARKING", [[{"LOWER": "parking"}]])
matcher.add("PARKING", [[{"LOWER": "no"}, {"LOWER": "parking"}]])
matcher.add("DEN", [[{"LOWER": "den"}]])
matcher.add("DEN", [[{"LOWER": "no"}, {"LOWER": "den"}]])
matcher.add("PRICE", [[{"TEXT": "$"}, {"IS_DIGIT": True}]])
matcher.add("PRICE", [[{"LOWER": "price"}, {"IS_DIGIT": True}]])
matcher.add("BUILDING AGE", [[{"LOWER": "age"}, {"IS_DIGIT": True}]])
matcher.add("BUILDING AGE", [[{"LOWER": "years old"}, {"IS_DIGIT": True}]])
matcher.add("MAINT", [
    [{"LOWER": "maintenance"}, {"LOWER": "fee"}, {"TEXT": {"REGEX": r"\$?[\d,]+"}}],
    [{"LOWER": "maintenance"}, {"LOWER": "fees"}, {"TEXT": {"REGEX": r"\$?[\d,]+"}}],
    [{"LOWER": "maintenance"}, {"TEXT": {"REGEX": r"\$?[\d,]+"}}],
    [{"LOWER": "maint"}, {"TEXT": {"REGEX": r"\$?[\d,]+"}}],
    [{"LOWER": "maintenance"}, {"IS_CURRENCY": True}, {"LIKE_NUM": True}]])
matcher.add("LOCATION", [[{"IS_DIGIT": True}, {"LOWER": "city"}]])
matcher.add("LOCATION", [[{"IS_DIGIT": True}, {"LOWER": "state"}]])
matcher.add("LATITUDE", [
    [{"LOWER": {"IN": ["latitude", "lat"]}}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}],
    [{"TEXT": {"REGEX": r"^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?)(°| degrees)?[ ]?[NSns]?$"}}]
])

matcher.add("LONGITUDE", [
    [{"LOWER": {"IN": ["longitude", "long", "lon"]}}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}],
    [{"TEXT": {"REGEX": r"^[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)(°| degrees)?[ ]?[EWew]?$"}}]
])

matcher.add("COORDINATES", [
    [{"TEXT": {"REGEX": r"^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?)$"}},
     {"IS_PUNCT": True},
     {"TEXT": {"REGEX": r"^[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)$"}}],
    [{"TEXT": {"REGEX": r"^(\d{1,3}\.\d+)°?[ ]?[NSns]?$"}},
     {"IS_PUNCT": True},
     {"TEXT": {"REGEX": r"^(\d{1,3}\.\d+)°?[ ]?[EWew]?$"}}]
])


def extract_housing_info(user_input):
    """Extracts housing information from user input."""
    doc = nlp(user_input)
    matches = matcher(doc)

    results = {
        'bedrooms': None,
        'bathrooms': None,
        'size': None,
        'parking': None,
        'building_age': None,
        'maint': None,
        'price': None,
        'location': None,
        'latitude': None,
        'longitude': None,
        'coordinates': None,
        'den': None

    }

    for match_id, start, end in matches:
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]

        # Get number of beds

        if label == "BEDROOMS":
            num = next((t.text for t in span if t.is_digit), None)
            if num:
                results['bedrooms'] = num

        # Get number of bathrooms

        elif label == "BATHROOMS":
            num = next((t.text for t in span if t.is_digit), None)
            if num:
                results['bathrooms'] = num

        # Get a range of sizes

        elif label == "SIZE":

            size_text = user_input.strip().lower()
            range_match = re.search(r'(\d+)\s*(?:-|to)\s*(\d+)\s*(?:sqft|sq\.?|square\s*feet)?\b', size_text)
            single_match = re.search(r'(\d+)\s*(?:sqft|sq\.?|square\s*feet)?\b', size_text)

            if range_match:
                num1 = int(range_match.group(1))  # First number (200)
                num2 = int(range_match.group(2))  # Second number (500)
                results['size'] = (min(num1, num2), max(num1, num2))  # Ensures (200, 500)
            else:
                size_value = int(single_match.group(1))
                results['size'] = (size_value, size_value)

        # Get parking

        elif label == "PARKING":
            if "no" in span.text.lower():
                results['parking'] = False
            else:
                results['parking'] = True

        # Get building age

        elif label == "BUILDING AGE":
            num = next(t.text for t in span if t.is_digit)
            if num:
                results['building_age'] = f"Building age {num}"

        elif label == "MAINT":
            text = span.text
            numbers = re.findall(r'[\d,]+\.?\d*', text)
            if numbers:
                num_str = numbers[0].replace(',', '')
                try:
                    results['maint'] = int(float(num_str))
                except ValueError:
                    pass

        # Get price

        elif label == "PRICE":
            num = next((t.text for t in span if t.is_digit), None)
            if num:
                results['price'] = num

        # Get den

        elif label == "DEN":
            if 'no' in span.text.lower():
                results['den'] = False
            else:
                results['den'] = True

        # Get location

        elif label == "LOCATION":
            num = next((t.text for t in span if t.is_digit), None)
            if num:
                results['location'] = num

        # Get latitude

        elif label == "LATITUDE":
            num = next((t.text for t in span if t.like_num), None)
            if num:
                results['latitude'] = num

        # Get longitude

        elif label == "LONGITUDE":
            num = next((t.text for t in span if t.like_num), None)
            if num:
                results['longitude'] = num

        # Get coordinates

        elif label == "COORDINATES":
            nums = [t.text for t in span if t.like_num]
            if len(nums) == 2:
                results['coordinates'] = f"{nums[0]},{nums[1]}"

        if results['longitude'] and results['latitude'] and not results['coordinates']:
            results['coordinates'] = f"{results['longitude']},{results['latitude']}"

        # In case no location is specified (so code doesn't break)
        if not results['location']:
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC", "FAC"):
                    results['Location'] = ent.text
                    break

    return results
