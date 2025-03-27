"""Sherlock Homes' main functionality.
Uses the NER model to extract relevant information from the user's response.

Implemented by: Gabriel Andrus, 3/26/2025
"""

from ner_model import *
import time


# Main ChatBot Interaction

print("Hello there! I'm Sherlock Homes, your friendly Toronto real-estate sleuth! It's nice to meet you")
time.sleep(3)

print("So, soon-to-be home-owner, tell me your preferences!")
print("How many beds do you want? Parking or no parking? Does location matter? How about maintenance price? "
      "Let me solve the mystery for you.")
time.sleep(2)
print("Oh, and please use DIGITS for numbers. I have a stigmatism in my right eye that prevents me from "
      "seeing words like 'two'")

user_response2 = input("What are you looking for in a house?")
time.sleep(2)
user_preferences = extract_housing_info(user_response2)
print(user_preferences)
