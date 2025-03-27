from flask import Blueprint
chatbot_call = Blueprint(__name__, 'chatbot')


@chatbot_call.route("/")
def chatbot():
    return "This is chatbot"
