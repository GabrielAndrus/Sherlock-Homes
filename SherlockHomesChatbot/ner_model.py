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

Implemented by: Gabriel Andrus, 3/25/25.
"""

import spacy
from spacy.matcher import Matcher

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
matcher.add("PRICE", [[{"TEXT": "$"}, {"IS_DIGIT": True}]])
matcher.add("PRICE", [[{"LOWER": "price"}, {"IS_DIGIT": True}]])

def extract_housing_info(user_input):
    """Extract housing information from user input."""
    doc = nlp(user_input)
    matches = matcher(doc)

    results = {
        'ward': None,
        'bedrooms': None,
        'bathrooms': None,
        'size': None,
        'parking': None,
        'price': None
    }

    for match_id, start, end in matches:
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]

        if label == "WARD":
            num = ''.join([t.text for t in span if t.is_digit])
            if num:
                results['ward'] = f"Ward {num}"

        elif label == "BEDROOMS":
            num = next((t.text for t in span if t.is_digit), None)
            if num:
                results['bedrooms'] = num

        elif label == "BATHROOMS":
            num = next((t.text for t in span if t.is_digit), None)
            if num:
                results['bathrooms'] = num

        elif label == "SIZE":
            num = next((t.text for t in span if t.is_digit), None)
            if num:
                results['size'] = num

        elif label == "PARKING":
            if "no" in span.text.lower():
                results['parking'] = False
            else:
                results['parking'] = True

        elif label == "PRICE":
            num = next((t.text for t in span if t.is_digit), None)
            if num:
                results['price'] = num

    return results
