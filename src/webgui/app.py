from flask import Flask, render_template, url_for, request
from src.database.db import DB_URL, loadData
from src.stats.stats import readStats
from src.webgui.utils.compiler import TailwindCompiler

app = Flask(__name__)
TailwindCompiler(app, npm_script_name="watch", debugmode_only=True)

@app.route('/')
def home():
    category = request.args.get('category')
    interest_rating = 0
    interest_rating = request.args.get('interest_rating')
    articles = loadData(DB_URL)
    categories = articles["Category"].unique()
    articles.fillna({"Category": "TODO"}, inplace=True)
    articles.fillna({"Interest_Rating" :-1}, inplace=True)
    if category != "" and category:
        articles = articles[(articles["Category"] == category)]
    if interest_rating != 0 and interest_rating:
        articles = articles[(articles["Interest_Rating"] >= interest_rating)]
    return render_template("home.html", articles=articles, categories=categories, current_category=category, current_rating=interest_rating)

@app.route('/info')
def model():
    data = readStats()
    return f"{data}"

with app.test_request_context():
    print(url_for('model'))
