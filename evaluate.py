import os
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


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
            'Description': row['Description'],
            'PrimaryColor': row['PrimaryColor']
        }
        preprocessed_data.append(item)
    return preprocessed_data


def evaluate_model(data_dir, model_v):
    # Load and preprocess the data from the data dumps in the specified directory
    data = load_data(data_dir)

    # Load the pre-trained models for productBrand, color, and gender prediction
    product_brand_model_path = os.path.join('models', f'v{model_v}', 'productBrand', 'model.joblib')
    color_model_path = os.path.join('models', f'v{model_v}', 'primaryColor', 'model.joblib')
    gender_model_path = os.path.join('models', f'v{model_v}', 'gender', 'model.joblib')
    product_brand_model = load(product_brand_model_path)
    color_model = load(color_model_path)
    gender_model = load(gender_model_path)

    # Split the data into features and labels
    x = [(item['Description'] + ' ' + item['ProductName']) for item in data]
    y_product_brand = [item['ProductBrand'] for item in data]
    y_color = [item['PrimaryColor'] for item in data]
    y_gender = [item['Gender'] for item in data]

    # Evaluate productBrand prediction
    evaluate_prediction("ProductBrand", y_product_brand, product_brand_model, x, model_v)

    # Evaluate color prediction
    evaluate_prediction("Color", y_color, color_model, x, model_v)

    # Evaluate gender prediction
    evaluate_prediction("Gender", y_gender, gender_model, x, model_v)

    print(f"Evaluation of Model v{model_v} completed.\n")


def evaluate_prediction(prediction_name, y_true, model, x, model_ver):
    y_pred = model.predict(x)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    # Print evaluation metrics for the specific prediction task
    print(f"{prediction_name} Accuracy: {accuracy}")
    print(f"{prediction_name} Precision: {precision}")
    print(f"{prediction_name} Recall: {recall}")
    print(f"{prediction_name} F1 Score: {f1}")

    # Plot evaluation metrics
    plot_metrics(prediction_name, accuracy, precision, recall, f1, model_ver)


def plot_metrics(prediction_name, accuracy, precision, recall, f1, model_versn):
    plot_dir = os.path.join('plot', 'evaluation', f'v{model_versn}')
    os.makedirs(plot_dir, exist_ok=True)

    # Plot accuracy
    plt.figure()
    plt.bar(prediction_name, accuracy)
    plt.xlabel('Prediction Task')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Prediction Tasks')
    plot_name = f"Accuracy_{prediction_name}.png"
    plt.savefig(os.path.join(plot_dir, plot_name), dpi=300)
    plt.close()

    # Plot precision
    plt.figure()
    plt.bar(prediction_name, precision)
    plt.xlabel('Prediction Task')
    plt.ylabel('Precision')
    plt.title('Precision of Prediction Tasks')
    plot_name = f"Precision_{prediction_name}.png"
    plt.savefig(os.path.join(plot_dir, plot_name), dpi=300)
    plt.close()

    # Plot recall
    plt.figure()
    plt.bar(prediction_name, recall)
    plt.xlabel('Prediction Task')
    plt.ylabel('Recall')
    plt.title('Recall of Prediction Tasks')
    plot_name = f"Recall_{prediction_name}.png"
    plt.savefig(os.path.join(plot_dir, plot_name), dpi=300)
    plt.close()

    # Plot F1-score
    plt.figure()
    plt.bar(prediction_name, f1)
    plt.xlabel('Prediction Task')
    plt.ylabel('F1-Score')
    plt.title('F1-Score of Prediction Tasks')
    plot_name = f"F1_score_{prediction_name}.png"
    plt.savefig(os.path.join(plot_dir, plot_name), dpi=300)
    plt.close()


model_version = 3  # Model version to evaluate
evaluate_model('data/train1.csv', model_version)
