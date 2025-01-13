from pdfminer.high_level import extract_text
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

# Prepare and save data
def prepare_data(resume_file_path, vectorizer_path='tfidf_vectorizer.pkl'):
    try:
        # Load the CSV file with CV data
        data = pd.read_csv(resume_file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {resume_file_path}")
        return
    except pd.errors.EmptyDataError:
        logging.error("The file is empty.")
        return

    # Select the relevant columns
    X = data['Resume_str']
    y = data['Category']

    # Check the distribution of categories
    logging.info("Category Distribution:")
    logging.info(y.value_counts())

    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the resume text to create feature vectors
    X_vectors = vectorizer.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectors, y, test_size=0.2, random_state=42
    )

    # Save the vectorizer and data
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(X_train, 'X_train.pkl')
    joblib.dump(X_test, 'X_test.pkl')
    joblib.dump(y_train, 'y_train.pkl')
    joblib.dump(y_test, 'y_test.pkl')

    logging.info("Data preparation completed and saved.")

if __name__ == "__main__":
    prepare_data("/Users/diasalehs/Desktop/Resume.csv")
