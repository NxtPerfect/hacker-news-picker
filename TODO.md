# Data preparing
## Step 1
- [ ] script to go onto hacker news and fetch titles with links of articles
    - if article in db, skip
    - if not, give a choice, either interesting or not interesting and discard
        - if interesting, what rating and category
    - if article discarded once, don't show it again
    - for pagination when all articles done, link is [https://news.ycombinator.com/?p=2] for 2 = page
- [ ] add at least 5000 articles

## Step 2
- [ ] One-hot-encode the titles or some other way to turn text into numbers
    - ensure all titles are of same length
- [ ] Ensure categories are consistent, don't create categories for a singular article

# Model
- [ ] Ideally, it should be able to both categorize the article, and judge if it's interesting and on what scale

## Step 1
- [ ] Model to predict the interesting rate of an article based on title
    - Needs at least 80% accuracy

## Step 2
- [ ] Model to categorize article based on title
    - Use already existing categories from file
    - At least 90% accuracy

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
