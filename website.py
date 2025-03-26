from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return "This is the home page"


if __name__ == '__main__':
    app.run(debug=True, port=8000)
