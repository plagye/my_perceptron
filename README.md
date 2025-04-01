# Perceptron Implementation in Python

This repository contains a Python implementation of Rosenblatt's Perceptron algorithm with a Flask web interface. The project demonstrates a simple binary classification model that predicts whether someone is a good business partner based on capital, amount of money saved each month, and age. However, by modifying the generate_dataset script you can create a custom dataset for a completely different use case. I'm a beginner enthusiast of machine learning and this is a training/portfolio project that will be updated in order to make it more personalized. Keep it real fellas

## Features
- Implementation of the Perceptron algorithm from scratch
- Flask web application with GUI
- Dataset generation with customizable parameters
- Min-max scaling for feature normalization
- Interactive prediction interface
- Visualizing the plane found by the Perceptron using matplotlib for 2D datasets

## Getting Started

### Prerequisites
Make sure you have Python installed on your system. Then install the required dependencies:

```bash
pip install -r requirements.txt
```

### Step 1: Generate the Dataset
First, generate a sample dataset using the provided script:

```bash
python generate_dataset.py
```

This will create a file called `sample_data.csv` in the `data` directory with features (capital, monthly savings, age) and labels (good_partner).

### Step 2: Scale the Dataset
Next, scale the dataset using min-max normalization:

```bash
python min_max_scaling.py
```

This will scale the dataset (`sample_data.csv`) and save the scaling parameters for future use.

### Step 3: Launch the Flask App
Finally, start the Flask web application:

```bash
python app.py
```

The application will be available at [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your web browser.

## Usage
1. Open the web application in your browser.
2. Enter the name of your dataset file (default: `sample_data_scaled.csv`).
3. Click "Train Model" to train the Perceptron.
4. Enter values for the three features (capital, monthly savings, age).
5. Click "Predict" to get the classification result.

## Project Structure
- `perceptron.py`: Core implementation of the Perceptron algorithm
- `train.py`: Functions for training and evaluating the model
- `utils.py`: Utility functions for data loading and visualization
- `app.py`: Flask web application
- `generate_dataset.py`: Script to generate synthetic data
- `min_max_scaling.py`: Script to scale the dataset
- `templates/`: HTML templates for the web interface
- `static/`: CSS and JavaScript files for the web interface
- `data/`: Directory for storing datasets

## Customization
You can customize the dataset generation by modifying the parameters in `generate_dataset.py`. The criteria for classifying a good business partner can be adjusted to create more balanced datasets.

## License
This project is open source and available for educational purposes.

## Acknowledgments
- Frank Rosenblatt for inventing the Perceptron algorithm
- Flask framework for the web interface
- Claude 3.7 Sonnet for help
