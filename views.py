from flask import Blueprint
home_call = Blueprint(__name__, 'home')


@home_call.route("/")
def home():
    return "This is home page"
