# Data preparing
## Step 1
- [x] script to go onto hacker news and fetch titles with links of articles
    - [x] if article in db, skip
    - [x] for pagination when all articles done, link is [https://news.ycombinator.com/?p=2] for 2 = page
    - [x] script should be seperate from the model, so that if at any point i want to change website to get news from
        it should only require the script to be changed, not the model
- [x] when adding article, update stats

### Optional
- [/] add at least 5000 articles
    - Maybe it's not needed
    - At least for now it's not needed
- [x] Ensure categories are consistent, don't create categories for a singular article

## Step 2
- [x] One-hot-encode the titles or some other way to turn text into numbers
    - ensure all titles are of same length
    - preprocess before splitting data
    - actually LabelEncoder is better for categories
- [x] Encode titles into numbers, probably tensor
    - [x] csr_matrix has no attribute 'to' when trying to move tfidf vector to cuda
    - [x] Create custom dataset in src/categorize/dataset.py
    - Uses bert

# Model
- [ ] Ideally, it should be able to both categorize the article, and judge if it's interesting and on what scale

## Step 1
- [x] Model to categorize article based on title
    - Use already existing categories from file
    - At least 90% accuracy
    - RNN, especially LSTM or GRU might be good,
    but computionally expensive
    - CNN can be faster but less effective
    - Uses GRU with 98% accuracy

## Step 2
- [x] Model to predict the interesting rate of an article based on title
    - Needs at least 80% accuracy
    - MLP easy to implement but worse than RNN
    - RNN, better than MLP but more computionally expensive
    - [x] How to embed title + category

# Generalization
- [/] main.py should run scraper
    - [x] categorize articles that don't have label
        - almost getting labels, just need to each
        each value from the tensor
    - [x] predict interest rating
        - bad size when loading model
        - input size is equal to how many articles there are
    - [/] once all data is labeled and has good accuracy
        load models and only retrain on feedback

# Feedback
- [ ] Feedback should change data in the .csv
then there's an option to retrain the model

## Optional
- [ ] Create one model to do it all or somehow join them together

# UI
## Step 1
- [ ] What frontend will I use?
    - supposedly it needs python backend,
    but if i only need .csv file of data the model spits out,
    I could get away with using nextjs or htmx

## Step 2
- [ ] Create UI in Figma
    - minimalistic
    - page for newest articles
        - detailed page for articles
            - feedback button,
            interesting or not,
            change rating,
            change category,
            remove article
            - link to article
    - sort by category
    - sort by interesting_rating
    - ?page to control the model
        - ?retrain model not needed if feedback works realtime
    - statistics
        - how many articles fetched
        - how many articles discarded
        - how many articles accepted
        - accuracy based on user feedback

- [ ] Implement it
    - maybe using streamlit?
    or mesop
    - or use flask and htmx?
