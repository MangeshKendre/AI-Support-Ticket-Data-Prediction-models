# Ticket System Prediction App

## Overview
This Streamlit app is designed to predict response times for support tickets based on their categorization.
Additionally, it includes a ticket classification model and sentiment analysis for ticket descriptions.

## Features
- **Response Time Prediction:**
  - Utilizes a Random Forest Regressor model to predict response times for support tickets.
  - Users can input ticket categorization details to receive predicted response times.

- **Ticket Classification:**
  - Uses a Naive Bayes classifier to categorize tickets based on their descriptions.
  - Users can input ticket descriptions to receive predicted categories.

- **Sentiment Analysis:**
  - Performs sentiment analysis on ticket descriptions.
  - Users can input ticket descriptions to analyze sentiment.


-Set up a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
Install dependencies: pip install -r requirements.txt

-Run the Streamlit app: streamlit run app.py

-Visit http://localhost:8501 in your browser to interact with the app.
