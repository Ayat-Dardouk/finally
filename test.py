from pdfminer.high_level import extract_text
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


# Load the CSV file
resume_file_path = "/Users/diasalehs/Desktop/Resume.csv"
data = pd.read_csv(resume_file_path)

# Select the relevant columns
X = data['Resume_str']  # The text content of the resumes
y = data['Category']  # The corresponding categories (labels)

# Check the distribution of categories
print(y.value_counts())

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the resume text to create feature vectors
X_vectors = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Save the model and vectorizer for later use
joblib.dump(model, 'cv_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=1))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# Prediction Method
def predict_category(cv_text):
    # Load the model and vectorizer
    loaded_model = joblib.load('cv_classifier.pkl')
    loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Transform the CV text to match the training data format
    cv_vector = loaded_vectorizer.transform([cv_text])

    # Use the loaded model to make predictions
    predicted_category = loaded_model.predict(cv_vector)
    return predicted_category[0]


# Test the prediction method
resume_file_path = "/Users/diasalehs/Desktop/Dia%20Saleh%20-%20CV.pdf"
extracted_text = extract_text_from_pdf(resume_file_path)

# Predict the category of the extracted CV
predicted_category = predict_category(extracted_text)
print(f'The predicted category for the CV is: {predicted_category}')
