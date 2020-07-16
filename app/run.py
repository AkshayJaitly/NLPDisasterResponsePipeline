import json
import re
import plotly
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    """
    Cleans, tokenizes, removes stopwords and stems the input text.
    Parameters
    text(string) : input text.
    Returns
    word_tokens : list of cleaned and stemmed tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    word_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        word_tokens.append(clean_tok)

    return word_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')

# Read sql lite table
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # group messages by genre
    genre_counts = df.groupby('genre').count()['message']

    # List of genre names
    genre_names = list(genre_counts.index)

    # get categories
    categories = df.drop(['id','message','original','genre'], axis=1)

    # category counts
    category_counts = categories.sum().sort_values(ascending=False) # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html

    # Category Names
    category_names = list(category_counts.index)

    # Social media categories
    social = df[df['genre'] == 'social']

    # get news categories
    social_categories = social.drop(['id','message','original','genre'], axis=1)

    # top 5 news
    social_counts = social_categories.sum().sort_values(ascending=False)[:5]

    # names of top 5 news
    social_category_names = list(social_counts.index)

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
    words = pd.Series(' '.join(df['message'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)).lower().split())

    # top 5 words
    top_5_counts = words[~words.isin(stopwords.words("english"))].value_counts()[:5]

    # top 5 word names
    top_5_words = list(top_5_counts.index)


    # group by categories
    counts = categories.groupby(categories.sum(axis=1)).count()['related']



    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color='MediumPurple'), #https://plotly.com/python/marker-style/
                    opacity=0.6
                )
            ],

            'layout': {
                'autosize': True,
                'title': 'Distribution of message genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                    marker=dict(color='DarkSlateGrey'),
                    opacity=0.8
                )
            ],

            'layout': {
                'title': 'Distribution of message categories',
                'yaxis': {
                    'title': "Number of values"
                },
                'xaxis': {
                    'title': "Disaster Categories",
                    'tickangle' : 20,
                },
                'automargin':True
            }
        },

        {
            'data': [
                Bar(
                    x=counts.index,
                    y=counts.values
                )
            ],

            'layout': {
                'title': 'Distribution of marked categories per message',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Number of marked categories"
                },
            }
        },
        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names[::-1],
                    z=categories.corr().values,
                    colorscale='Productivity' # https://plotly.com/python/heatmaps/
                )
            ],

            'layout': {
                'title': 'Heatmap of categories',
                'xaxis': {
                    'tickangle' : 45
                },
                'autosize':True,
                'automargin':True
            }
        },
        {
            'data': [
                Bar(
                    x=top_5_words,
                    y=top_5_counts
                )
            ],

            'layout': {
                'title': 'Top 5 most frequent words in messages',
                'yaxis': {
                    'title': "Frequency",

                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=social_category_names,
                    values=social_counts
                )
            ],

            'layout': {
                'title': 'Social - Pie diagram of top 5 Message Categories'
            }
        },

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]

    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
