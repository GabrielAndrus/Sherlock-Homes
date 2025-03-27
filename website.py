from flask import Flask
from views import chatbot, home
app = Flask(__name__)

app.register_blueprint(home, url_prefix="/home")
app.register_blueprint(chatbot, url_prefix="/chatbot")

if __name__ == '__main__':
    app.run(debug=True, port=9000)
