"""Test to determine whether the
SpaCyLabel module, as defined in ner_model, works as expected."""

import spacy

# Load nlp
nlp = spacy.load("en_core_web_sm")

config = {
    "overwrite_ents": True,
    "patterns": [
        {"label": "WARD", "pattern": [{"LOWER": "ward"}, {"IS_DIGIT": True}]},
        {"label": "BATHROOMS", "pattern": [{"LOWER": "bath"}, {"IS_DIGIT": True}]}
    ]
}

nlp.add_pipe("entity_ruler", config=config, before="ner")

doc = nlp("Apartment in ward 5 with 2 bath")
print([(ent.text, ent.label_) for ent in doc.ents])

# expected output = [('ward 5', 'WARD'), ('bath 2', 'BATHROOMS')]
