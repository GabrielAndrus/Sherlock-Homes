from flask import Flask
from views import home_call
from chatbot import chatbot, chatbot_call

app = Flask(__name__)

app.register_blueprint(home_call, url_prefix="/home")
app.register_blueprint(chatbot_call, url_prefix="/chatbot")

if __name__ == '__main__':
    app.run(debug=True, port=9000)
