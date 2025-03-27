""""Test whether SpaCy has been installed correctly."""
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("A complex-example,!")
print([token.text for token in doc])
