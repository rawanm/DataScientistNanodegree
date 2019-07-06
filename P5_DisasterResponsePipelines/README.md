# Disaster Response Pipeline Project
> This project (Disaster Response Pipeline) is part of Udacity's Data Scientists Nanodegree Program. 

### Introduction: 
This project builds a Disaster Response Pipeline as a web app. Users can enter new message and the system will analyze the text and display the relevent message category. This project analyzes disaster data and uses data from [Figure Eight](https://www.figure-eight.com) to build machine learning model that classifies messages. 


### Project Components:

1. ETL Pipeline: 
    Loads the data and perform ETL pipline to save it as SQLite database. 

2. ML Pipeline: 
    Builds and trains a machine learning classification model and save it as .pkl file to be used in web app.

3. Flask Web App: 
    A simple web app to classify new messages, code is mostly provided by [Udacity](https://www.udacity.com)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves, specify to run gridsearch (true, false)
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl true/fasle`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

