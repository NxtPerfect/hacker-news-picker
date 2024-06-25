from flask import Flask, render_template, url_for
from src.database.db import DB_URL, loadData
from src.stats.stats import readStats

app = Flask(__name__)

@app.route('/')
def home():
    articles = loadData(DB_URL)[:10]
    for index, article in articles.iterrows():
        print(article["Title"])
    return render_template("home.html", articles=articles)
    # return "<h1>Hacker News Picker</h1>"

@app.route('/info')
def model():
    data = readStats()
    return f"{data}"

with app.test_request_context():
    print(url_for('model'))
