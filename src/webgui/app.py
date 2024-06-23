from flask import Flask, render_template, url_for
from src.stats.stats import readStats

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")
    # return "<h1>Hacker News Picker</h1>"

@app.route('/model')
def model():
    data = readStats()
    return f"{data}"

with app.test_request_context():
    print(url_for('model'))
