import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(data_dump):
    # Loading data from the data dumps
    data = []
    content = pd.read_csv(data_dump, sep=',',
                          usecols=['ProductName', 'ProductBrand', 'Gender', 'Description', 'PrimaryColor'])
    preprocess_data = preprocess(content)
    data.extend(preprocess_data)
    return data


def preprocess(df):
    preprocessed_data = []
    for _, row in df.iterrows():
        item = {
            'ProductName': row['ProductName'],
            'ProductBrand': row['ProductBrand'],
            'Gender': row['Gender'],
            'Description': preprocess_description(row['Description']),
            'PrimaryColor': row['PrimaryColor']
        }
        preprocessed_data.append(item)
    return preprocessed_data


def preprocess_description(description):
    # Remove special characters and digits from the description
    description = re.sub(r'[^a-zA-Z\s]', '', description)
    description = re.sub(r'\s+', ' ', description)  # Remove extra whitespaces
    description = description.lower()  # Convert to lowercase

    # Tokenize the description into individual words
    tokens = word_tokenize(description)

    # Remove stop words from the tokenized description
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Join the filtered tokens back into a single string
    filtered_description = ' '.join(filtered_tokens)

    return filtered_description


def build_model():
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    classifier = LogisticRegression(max_iter=10000)

    scaler = StandardScaler(with_mean=False)

    # Build the model pipeline
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('scaler', scaler),
        ('classifier', classifier)
    ])

    return model


def train_model(data_dir, model_version):
    print(f"Training for model v{model_version} has started.")
    # Load and preprocess the data from the data dumps in the specified directory
    data = load_data(data_dir)

    # Build the model
    model = build_model()

    # Split the data into features and labels
    x = [(item['Description'] + ' ' + item['ProductName']) for item in data]
    y_labels = ['ProductBrand', 'PrimaryColor', 'Gender']

    for label in y_labels:
        y = [item[label] for item in data]

        # Fit the model to the data
        model.fit(x, y)

        # Save the trained model for prediction
        model_dir = os.path.join('models', f'v{model_version}', label)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.joblib')
        dump(model, model_path)
        print(f"Model for label {label} trained and saved.")

    print(f"Model v{model_version} trained and saved.")


train_model('data/train1.csv', 3)
