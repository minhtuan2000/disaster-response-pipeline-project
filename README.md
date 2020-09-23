# Disaster Response Pipeline Project

This project aims to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Usages
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
    ```bash
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```
    - To run ML pipeline that trains classifier and saves
    ```bash
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```

2. Run the following command in the app's directory (`/app`) to run the web app.
    ```bash
    python run.py
    ```

3. Go to http://0.0.0.0:3001/

### Project components

#### Dataset
- `/data/disaster_messages.csv`: A dataset contains more than 30,000 disaster messages
- `/data/disaster_categories.csv`: A dataset contains pre-classified categories of disaster messages for training purpose

#### Scripts
- `/data/process_data.py`: An ETL pipeline that cleans and stores data in a database
- `/models/train_classifier.py`: An ML pipeline that trains classifier and saves
- `/app/run.py`: A web app that displays visualizations of the data and classifies messages
