
################## Ticket classification model ######################################################



import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import joblib
# Load the data into a pandas DataFrame
df = pd.read_csv('digitalxc.csv', encoding='latin1')
# Assume 'Description', 'Categorization Tier 1', and other relevant columns

# Data Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    return cleaned_text

df['CleanedDescription'] = df['Description'].apply(preprocess_text)

# Ticket Classification Model
X_text = df['CleanedDescription']
y_category = df['Categorization Tier 1']

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X_text)

loaded_classifier = joblib.load('model.joblib')


# Streamlit App
st.title('Ticket Classification Model')
user_input_description = st.text_area('Enter the ticket description:')

if st.button('Predict Category'):
    cleaned_user_input = preprocess_text(user_input_description)
    user_input_tfidf = vectorizer.transform([cleaned_user_input])
    predicted_category = loaded_classifier.predict(user_input_tfidf)
    
    st.write(f'Predicted Category: {predicted_category[0]}')












######## Response time prediction ############################################################################################

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

# Assuming df_encoded is your encoded DataFrame
df_encoded = pd.read_csv('time_pred.csv')

# Ensure 'Categorization Tier 1', 'Categorization Tier 2', and 'Categorization Tier 3' are categorical
df_encoded[['Categorization Tier 1', 'Categorization Tier 2', 'Categorization Tier 3']] = df_encoded[['Categorization Tier 1', 'Categorization Tier 2', 'Categorization Tier 3']].astype('category')

# Response Time Prediction Model Training
X_regression_encoded = df_encoded[['Categorization Tier 1', 'Categorization Tier 2', 'Categorization Tier 3']]
y_regression_encoded = df_encoded['MTTR in Mins']

# Encode categorical variables using OrdinalEncoder
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_regression_encoded = ordinal_encoder.fit_transform(X_regression_encoded)

# Split the data into training and testing sets
X_train_reg_encoded, X_test_reg_encoded, y_train_reg_encoded, y_test_reg_encoded = train_test_split(
    X_regression_encoded, y_regression_encoded, test_size=0.2, random_state=42
)

# Model for Response Time Prediction (Random Forest Regressor)
random_forest_regressor_model = RandomForestRegressor()
random_forest_regressor_model.fit(X_train_reg_encoded, y_train_reg_encoded)

# Streamlit App
st.title("Response Time Prediction App")

# Create dropdowns for user input
user_input_cat1 = st.selectbox("Select Categorization Tier 1", df_encoded['Categorization Tier 1'].unique())
user_input_cat2 = st.selectbox("Select Categorization Tier 2", df_encoded[df_encoded['Categorization Tier 1'] == user_input_cat1]['Categorization Tier 2'].unique())
user_input_cat3 = st.selectbox("Select Categorization Tier 3", df_encoded[(df_encoded['Categorization Tier 1'] == user_input_cat1) & (df_encoded['Categorization Tier 2'] == user_input_cat2)]['Categorization Tier 3'].unique())

# Example of User Input for Regression
user_input_reg_encoded = {
    'Categorization Tier 1': [user_input_cat1],
    'Categorization Tier 2': [user_input_cat2],
    'Categorization Tier 3': [user_input_cat3]
}

user_input_reg_encoded_df = pd.DataFrame(user_input_reg_encoded)

# Ensure that the columns are in the same order as during training
user_input_reg_encoded_df = user_input_reg_encoded_df[['Categorization Tier 1', 'Categorization Tier 2', 'Categorization Tier 3']]

# Transform the input using the trained encoder
user_input_reg_encoded_df_transformed = ordinal_encoder.transform(user_input_reg_encoded_df)

# Predict using the trained Random Forest Regressor
predicted_response_time_rf = random_forest_regressor_model.predict(user_input_reg_encoded_df_transformed)

# Convert predicted response time from minutes to hours
predicted_response_time_hours = predicted_response_time_rf[0] / 60.0

# Display the result
st.write(f'\nPredicted Response Time for User Input (Random Forest): {predicted_response_time_rf[0]} minutes or {predicted_response_time_hours:.2f} hours')




###############  Sentiment Analysis ##############################################################



import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Dropdown options for ticket descriptions
ticket_options = [
    'The system is down, and I cannot access my files.',
    'I am experiencing slow internet connectivity issues.',
    'There is an error in the software application.',
    'My computer is not turning on.',
    'I need assistance with resetting my password.'
]

# Streamlit App
st.title('Ticket Sentiment Analysis App')
st.subheader('Select a ticket description from the dropdown menu:')

# Create a dropdown menu for user input
selected_ticket = st.selectbox('Choose a Ticket Description:', ticket_options)

# Function to get sentiment scores
def get_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

# Button to trigger sentiment analysis
if st.button('Analyze Sentiment'):
    # Apply sentiment analysis to the selected ticket description
    ticket_sentiment = get_sentiment(selected_ticket)

    # Display the sentiment scores for the selected ticket description
    st.subheader('Sentiment Analysis for Selected Ticket:')
    st.write(f'Text: {selected_ticket}')
    st.write(f'Sentiment Scores: {ticket_sentiment}')
