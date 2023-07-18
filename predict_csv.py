import os
import csv
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename


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


def process_csv(input_file, output_file, model_dir):
    with open(input_file, 'r') as csv_input, open(output_file, 'w', newline='') as csv_output:
        reader = csv.reader(csv_input)
        writer = csv.writer(csv_output)

        # Write the header row in the output CSV file
        header = next(reader)
        writer.writerow(header + ['Predicted productBrand', 'Predicted Color', 'Predicted Gender'])

        # Load the models
        product_brand_model_dir = os.path.join(model_dir, 'productBrand')
        product_brand_model = load_model(product_brand_model_dir)

        color_model_dir = os.path.join(model_dir, 'primaryColor')
        color_model = load_model(color_model_dir)

        gender_model_dir = os.path.join(model_dir, 'gender')
        gender_model = load_model(gender_model_dir)

        predicted_product_brands = []
        predicted_genders = []
        predicted_colors = []

        for row in reader:
            # Get the description and product name from the CSV row
            description = row[0]
            product_name = row[1]

            # Make predictions
            predicted_product_brand = predict_product_brand(product_brand_model, description, product_name)
            predicted_color = predict_color(color_model, description, product_name)
            predicted_gender = predict_gender(gender_model, description, product_name)

            # Write the original row with the predicted features to the output CSV file
            writer.writerow(row + [predicted_product_brand, predicted_color, predicted_gender])

            # Collect predicted product brands, genders, and colors for plotting
            predicted_product_brands.append(predicted_product_brand)
            predicted_genders.append(predicted_gender)
            predicted_colors.append(predicted_color)

        # Calculate the count of each unique product brand
        unique_product_brands, brand_counts = np.unique(predicted_product_brands, return_counts=True)

        # Sort the brands and counts in descending order by frequency
        sorted_indices = np.argsort(brand_counts)[::-1]
        unique_product_brands = unique_product_brands[sorted_indices]
        brand_counts = brand_counts[sorted_indices]

        # Create a DataFrame from the unique product brands and brand counts
        df_brand = pd.DataFrame({'Product Brand': unique_product_brands, 'Count': brand_counts})

        # Sort the DataFrame by 'Count'
        df_brand = df_brand.sort_values(by='Count')

        # Specify the plot directory
        plot_dir = os.path.join('plot', 'prediction')
        os.makedirs(plot_dir, exist_ok=True)

        # Plot the product brand distribution
        plt.figure(figsize=(35, 28))
        plt.bar(df_brand['Product Brand'], df_brand['Count'])  # Plot the bars
        plt.xlabel('Product Brand', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.title('Distribution of Predicted Product Brands')

        plt.xticks(rotation=90)

        # Add values on top of the bars
        for i, count in enumerate(df_brand['Count']):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom', rotation=90)

        # Adjust the x-axis limits to remove the gap on either side
        plt.xlim(-0.5, len(df_brand) - 0.5)

        # Adjust the y-axis limits to provide more space at the top
        plt.ylim(0, max(df_brand['Count']) + 10)

        plt.savefig(os.path.join(plot_dir, 'product_brand_plot.png'), dpi=300)
        plt.show()

        # Calculate the count of each unique gender
        unique_genders, gender_counts = np.unique(predicted_genders, return_counts=True)

        # Sort the genders and counts in descending order by frequency
        sorted_indices = np.argsort(gender_counts)[::-1]
        unique_genders = unique_genders[sorted_indices]
        gender_counts = gender_counts[sorted_indices]

        # Create a DataFrame from the unique genders and gender counts
        df_gender = pd.DataFrame({'Gender': unique_genders, 'Count': gender_counts})

        # Sort the DataFrame by 'Count'
        df_gender = df_gender.sort_values(by='Count')

        # Plot the gender distribution
        plt.figure(figsize=(12, 10))
        plt.bar(df_gender['Gender'], df_gender['Count'])  # Plot the bars
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Genders')

        # Add values on top of the bars
        for i, count in enumerate(df_gender['Count']):
            plt.text(i, count + 2, str(count), ha='center', va='bottom', rotation=90)

        # Adjust the x-axis limits to remove the gap on either side
        plt.xlim(-0.5, len(df_gender) - 0.5)

        # Adjust the y-axis limits to provide more space at the top
        plt.ylim(0, max(df_gender['Count']) + 50)

        plt.savefig(os.path.join(plot_dir, 'gender_plot.png'), dpi=300)
        plt.show()

        # Calculate the count of each unique color
        unique_colors, color_counts = np.unique(predicted_colors, return_counts=True)

        # Sort the colors and counts in descending order by frequency
        sorted_indices = np.argsort(color_counts)[::-1]
        unique_colors = unique_colors[sorted_indices]
        color_counts = color_counts[sorted_indices]

        # Create a DataFrame from the unique colors and color counts
        df_color = pd.DataFrame({'Color': unique_colors, 'Count': color_counts})

        # Sort the DataFrame by 'Count'
        df_color = df_color.sort_values(by='Count')

        # Plot the color distribution
        plt.figure(figsize=(12, 10))
        plt.bar(df_color['Color'], df_color['Count'])  # Plot the bars
        plt.xlabel('Color')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Colors')

        plt.xticks(rotation=90)

        # Add values on top of the bars
        for i, count in enumerate(df_color['Count']):
            plt.text(i, count + 2, str(count), ha='center', va='bottom', rotation=90)

        # Adjust the y-axis limits to provide more space at the top
        plt.ylim(0, max(df_color['Count']) + 30)

        plt.savefig(os.path.join(plot_dir, 'color_plot.png'), dpi=300)
        plt.show()


# Specify the model version and directory
model_version = 2
model_dir = os.path.join('models', f'v{model_version}')

# Open a Tkinter file dialog to select the input file
Tk().withdraw()
input_file = askopenfilename(title='Select Input CSV File', filetypes=[('CSV Files', '*.csv')])

# Open a Tkinter file dialog to select the output file
output_file = asksaveasfilename(title='Save Output CSV File', defaultextension='.csv', filetypes=[('CSV Files', '*.csv')])

# Process the CSV file and generate predictions
process_csv(input_file, output_file, model_dir)

print("Predictions saved to output.csv")

