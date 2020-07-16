# Disaster Response Pipeline Project

### Table of Contents

1.  [Instructions](#instructions)
2.  [Installations](#installations)
3.  [Demo Gif](#gif)
4.  [Project Motivation](#motivation)
5.  [File Descriptions](#files)
6.  [Acknowledgements](#acknowledged)

### Instructions:

1.  Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2.  Run the following command in the app's directory to run your web app.
    `python run.py`

3.  Go to http://0.0.0.0:3001/
    (https://view6914b2f4-3001.udacity-student-workspaces.com/)

The main page

![messageDistribution](images/messageDistribution.png)

![socialPie](images/socialPie.png)

The model output

![classification1](images/classification1.png)

![classification2](images/classification2.png)

## Installations<a name="installations"></a>

> 1.  Python 3.5+
> 2.  ML Libraries: NumPy, Pandas, SkLearn
> 3.  NLP Libraries: NLTK
> 4.  SQLlite Libraries: SQLalchemy
> 5.  Model Loading and Saving Library: joblib
> 6.  Web App and Visualization: Flask, Plotly

## Demo Gif<a name="gif"></a>

![Nlp Demo](gif/nlpdisaster.gif)

## Project Motivation<a name="motivation"></a>

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing and Machine learning model to categorize messages on a real time basis.

The disaster response pipeline is broken into 3 main subproblems:

1.  Building an ETL pipeline to extract data, clean the data and save the data in a SQLite Database
2.  Building a NLP & ML pipeline to train our model
3.  Run a Web App to show our model results using Flask and Plotly.

## File Descriptions <a name="files"></a>

- `app/`

  - `templates/`
    - `master.html` : Main page of web application
    - `go.html` : Classification results page of web application
  - `run.py` : Flask applications python code

- `data/`

  - `disaster_categories.csv` : Disaster categories dataset
  - `disaster_messages.csv` : Disaster Messages dataset
  - `process_data.py` : The ETL data processing pipeline script
  - `DisasterResponse.db` : The database with the merged and cleaned data

- `models/`

  - `train_classifier.py` : The NLP and Machine learning pipeline script which trains and improves the model

- `gif/` - `nlpdisaster.gif` - A small demo gif of the appliction.

## Acknowledgements<a name="acknowledged"></a>

Thanks to Figure Eight and AirBnb for the dataset and Udacity for the course.
