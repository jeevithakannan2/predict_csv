import os

from joblib import load


def load_model(model_dir):
    model_path = os.path.join(model_dir, 'model.joblib')
    model = load(model_path)
    return model


def predict_product_brand(model, description, product_name):
    input_text = description + ' ' + product_name
    predicted_product_brand = model.predict([input_text])
    return predicted_product_brand[0]


def predict_color(model, description, product_name):
    input_text = description + ' ' + product_name
    predicted_color = model.predict([input_text])
    return predicted_color[0]


def predict_gender(model, description, product_name):
    input_text = description + ' ' + product_name
    predicted_gender = model.predict([input_text])
    return predicted_gender[0]


def main():
    # Specify the model version and directory
    model_version = 2
    model_dir = os.path.join('models', f'v{model_version}')

    # Load the model
    product_brand_model_dir = os.path.join(model_dir, 'productBrand')
    product_brand_model = load_model(product_brand_model_dir)

    color_model_dir = os.path.join(model_dir, 'primaryColor')
    color_model = load_model(color_model_dir)

    gender_model_dir = os.path.join(model_dir, 'gender')
    gender_model = load_model(gender_model_dir)

    # Example inputs for prediction
    description = "U.S. Polo Assn. Kids Boys Grey & Blue Printed Round Neck T-shirt"
    product_name = "Grey and Blue printed T-shirt, has a round neck, and long sleeves"

    # Make predictions
    predicted_product_brand = predict_product_brand(product_brand_model, description, product_name)
    predicted_color = predict_color(color_model, description, product_name)
    predicted_gender = predict_gender(gender_model, description, product_name)

    # Print the predicted features
    print("Predicted productBrand:", predicted_product_brand)
    print("Predicted Color:", predicted_color)
    print("Predicted Gender:", predicted_gender)


if __name__ == '__main__':
    main()
