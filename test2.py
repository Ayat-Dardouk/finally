from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
import numpy as np
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Train and evaluate the Decision Tree model
def train_and_evaluate_model():
    try:
        X_train = joblib.load('X_train.pkl')
        X_test = joblib.load('X_test.pkl')
        y_train = joblib.load('y_train.pkl')
        y_test = joblib.load('y_test.pkl')
    except FileNotFoundError as e:
        logging.error(f"Required data file not found: {e}")
        return

    # Create and train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'cv_classifier_decision_tree.pkl')

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy:.2f}")
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred, zero_division=1))
    logging.info("Confusion Matrix:")
    logging.info(confusion_matrix(y_test, y_pred))

# Predict job category
def predict_job_category(job_description, model, vectorizer):
    job_vector = vectorizer.transform([job_description])
    predicted_category = model.predict(job_vector)
    return predicted_category[0], job_vector

# Match CVs to a job description
def match_cvs_to_job(job_description, data, model, vectorizer, top_n=3):
    job_category, job_vector = predict_job_category(job_description, model, vectorizer)
    matched_cvs = data[data['Category'] == job_category]

    if matched_cvs.empty:
        logging.info("No matching CVs found.")
        return pd.DataFrame(), []

    matched_cv_vectors = vectorizer.transform(matched_cvs['Resume_str'])
    similarities = cosine_similarity(job_vector, matched_cv_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    top_cvs = matched_cvs.iloc[top_indices]
    top_scores = similarities[top_indices]

    return top_cvs, top_scores

if __name__ == "__main__":
    # Train the model
    train_and_evaluate_model()

    # Load model, vectorizer, and data for matching
    model = joblib.load('cv_classifier_decision_tree.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    data = pd.read_csv('/Users/diasalehs/Desktop/Resume.csv')

    # Example job description
    job_description = """
    Graphic designers use their creativity, technical skills, and design expertise to craft compelling visual concepts.
    """
    matched_cvs, scores = match_cvs_to_job(job_description, data, model, vectorizer)

    # Display matched CVs
    if not matched_cvs.empty:
        logging.info(f"Matched CVs for category '{predict_job_category(job_description, model, vectorizer)[0]}':")
        for index, row in matched_cvs.iterrows():
            logging.info(
                f"ID: {row['ID']}, Category: {row['Category']}, Similarity Score: {scores[np.where(matched_cvs.index == index)[0][0]]:.4f}"
            )
