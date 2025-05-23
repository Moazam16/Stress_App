import streamlit as st
import pandas as pd
# import numpy as np
import os
import datetime

# Machine learning and text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Visualization and NLP
from wordcloud import WordCloud
from textblob import TextBlob
# import matplotlib.pyplot as plt

# Title
st.title("Human Stress Prediction App")
st.write("Answer the following questions to predict your stress level.")

@st.cache_resource
def train_model():

    """
    Train the stress detection model using a TF-IDF vectorizer and Logistic Regression.
    """
    # Load the dataset (Reddit posts text with binary stress label)
    df = pd.read_csv("Stress.csv")
    X = df['text']
    y = df['label']
    # Define the pipeline: TF-IDF vectorizer + Logistic Regression classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    # Train the model
    pipeline.fit(X, y)
    return pipeline

# Load or train the model (cached after first run)
model = train_model()
tfidf = model.named_steps['tfidf']
clf = model.named_steps['clf']

# Questionnaire - collect user inputs
st.header("Questionnaire")
q1 = st.text_area(
    "1. How have you been feeling recently?",
    placeholder="E.g., I've been feeling overwhelmed with work lately.",
    height=100
)

q2 = st.text_area(
    "2. Are there any situations or thoughts causing you stress?",
    placeholder="E.g., I’m stressed about upcoming deadlines and family expectations.",
    height=100
)

q3 = st.text_area(
    "3. What are you most worried about right now?",
    placeholder="E.g., I'm worried I won’t be able to finish everything on time.",
    height=100
)
# Predict button
if st.button("Predict Stress Level"):
    # Combine the responses into one string
    combined_text = " ".join([q1, q2, q3]).strip()
    if not combined_text:
        st.warning("Please answer at least one question to get a prediction.")
    else:
        # Preprocessing and prediction
        pred_prob = model.predict_proba([combined_text])[0][1]  # Probability of stress (label=1)
        pred_label = model.predict([combined_text])[0]
        # Determine stress level category
        if pred_prob >= 0.7:
            stress_level = "High Stress"
            st.error(f"Predicted Stress Level: **{stress_level}**")
        elif pred_prob >= 0.4:
            stress_level = "Medium Stress"
            st.warning(f"Predicted Stress Level: **{stress_level}**")
        else:
            stress_level = "Low Stress"
            st.success(f"Predicted Stress Level: **{stress_level}**")
        st.write(f"Stress probability: {pred_prob:.2f}")

        # Word Cloud of the responses
        wc = WordCloud(width=600, height=400, background_color='white').generate(combined_text)
        st.image(wc.to_array(), caption="Word Cloud of your responses", use_container_width=True)

        # Sentiment analysis
        blob = TextBlob(combined_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        st.subheader("Sentiment Analysis of Responses")
        st.write(f"**Polarity:** {polarity:.2f}   |   **Subjectivity:** {subjectivity:.2f}")
        # Display as bar chart
        sent_df = pd.DataFrame({
            'Metric': ['Polarity', 'Subjectivity'],
            'Score': [polarity, subjectivity]
        }).set_index('Metric')
        st.bar_chart(sent_df)

        # # Model interpretability - feature contributions
        # st.subheader("Model Interpretation: Word Contributions")
        # # Compute contributions: TF-IDF values * model coefficients
        # vec = tfidf.transform([combined_text])
        # contributions = vec.toarray()[0] * clf.coef_[0]
        # # Top positive and negative contributing words
        # top_pos_idx = np.argsort(contributions)[-5:][::-1]
        # top_neg_idx = np.argsort(contributions)[:5]
        # words = np.array(tfidf.get_feature_names_out())
        # pos_words = words[top_pos_idx]
        # neg_words = words[top_neg_idx]
        # pos_vals = contributions[top_pos_idx]
        # neg_vals = contributions[top_neg_idx]
        # # Prepare combined lists for plotting
        # words_expl = list(pos_words) + list(neg_words)
        # contrib_vals = list(pos_vals) + list(neg_vals)
        # # Plot horizontal bar chart (positive contributions in green, negative in red)
        # fig, ax = plt.subplots(figsize=(6,4))
        # bar_colors = ['green' if val > 0 else 'red' for val in contrib_vals]
        # ax.barh(words_expl, contrib_vals, color=bar_colors)
        # ax.set_xlabel("Contribution to log-odds (positive -> more stressed)")
        # ax.set_ylabel("Word")
        # ax.set_title("Top Positive (green) and Negative (red) Word Contributions")
        # st.pyplot(fig)

        # Log user responses and prediction to a local CSV file
        log_entry = {
            'question1': q1,
            'question2': q2,
            'question3': q3,
            'combined_text': combined_text,
            'predicted_label': int(pred_label),
            'predicted_prob': pred_prob,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        log_df = pd.DataFrame([log_entry])
        csv_file = "user_data.csv"
        if not os.path.isfile(csv_file):
            log_df.to_csv(csv_file, index=False)
        else:
            log_df.to_csv(csv_file, mode='a', header=False, index=False)
        st.info("Your responses and the prediction have been saved.")
