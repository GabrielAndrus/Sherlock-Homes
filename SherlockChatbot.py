"""Sherlock Homes' main functionality.
Uses the NER model to extract relevant information from the user's response.

Implemented by: Gabriel Andrus, 3/26/2025
"""

from ner_model import *
import time

def handle_vague_input(user_preferences: dict):
    """If the user gives little to no information, prompt then to give more."""
    missing = []
    if not user_preferences.get('beds'):
        missing.append('beds')
    if not user_preferences.get('price'):
        missing.append('price')
    if not user_preferences.get('location'):
        missing.append('location')
    if not user_preferences.get('baths'):
        missing.append('baths')

    if missing:
       return (f"Could you please specify your preferred: {', '.join(missing)} ? Say something like '3 beds under $1 mil")

    else:
        return ("Thanks for all your information! It helps me a lot.")

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
print("Thank you for your input! I'll generate some houses you'd like based on your preferences!")
time.sleep(2)
user_preferences = extract_housing_info(user_response2) # Dictionary containing user preferences
handle_vague_input(user_preferences) # Handle vague and absent user input
time.sleep(3)
print("Got it! I'm closer to solving this mystery. Now...")
num_results = int(input("How many results do you want to see?"))


time.sleep(2)
print("Generating...")
time.sleep(3)
print("Generating...")

if __name__ == '__main__':
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'max-line-length': 120,
        'extra-imports': ['time']
    })
