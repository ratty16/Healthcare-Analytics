# Healthcare Insurance Data Analysis Dashboard

## Overview

This project involves building an interactive web dashboard using Streamlit to analyze and visualize healthcare insurance data. The dataset includes various features such as age, BMI, gender, region, and smoking status, and it aims to predict medical charges based on these factors using a linear regression model.

## Project Structure

- `app.py`: Main Streamlit application file.
- `insurance.csv`: Dataset used for analysis and prediction.
- `requirements.txt`: Python dependencies for the project.

## Setup and Installation

To run this project, you'll need Python and the necessary libraries installed. You can install the required libraries using `pip` with the provided `requirements.txt` file.

1. **Clone the Repository**

    ```bash
    git clone https://github.com/ratty16/Healthcare-Analytics.git
    cd Healthcare-Analytics
    ```

2. **Create a Virtual Environment (Optional but recommended)**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To start the Streamlit dashboard, run:

```bash
streamlit run app.py
```
This will open the application in yourdefault web browser.

# Healthcare Analytics Dashboard

## Features

- **Data Exploration:** View and analyze the healthcare insurance dataset.
- **Visualization Options:** Select various visualizations from the sidebar to explore age distribution, gender distribution, BMI distribution, and medical charges.
- **Predictive Modeling:** Predict medical charges based on user input using a linear regression model.
- **Interactive Widgets:** Input parameters for predictions and see real-time results.

## Visualization Options

The dashboard provides multiple visualization options, including:

- **Age Distribution:** Histogram showing the distribution of ages in the dataset.
- **Gender Distribution:** Count plot of gender distribution.
- **BMI Distribution:** Histogram showing the distribution of BMI values.
- **Medical Charges vs Age:** Scatter plot of medical charges against age.
- **Medical Charges vs BMI:** Scatter plot of medical charges against BMI.
- **Age Group Distribution:** Count plot of different age groups.
- **BMI Category Distribution:** Count plot of BMI categories.
- **Medical Charges vs Age Group:** Scatter plot of medical charges against age groups.
- **Medical Charges vs BMI Category:** Scatter plot of medical charges against BMI categories.

## Predictive Model

A linear regression model is used to predict medical charges based on age, BMI, and smoking status. The model is evaluated using:

- **Mean Squared Error (MSE):** Indicates the average squared difference between actual and predicted values.
- **R-squared Score:** Represents the proportion of variance in medical charges explained by the features.

### Model Evaluation

- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values. A lower MSE indicates better model performance.
- **R-squared Score:** Shows the proportion of variance in medical charges explained by the model. A higher R-squared score indicates a better fit.

## Usage

To make a prediction based on user input:

1. Use the sidebar sliders and checkbox to enter values for age, BMI, and smoking status.
2. Click on "Predict Medical Charges" to see the estimated medical charges.

### Example Prediction

Here’s an example of how the prediction works:

- **Age:** 30
- **BMI:** 25
- **Smoker:** Yes

The model will output the predicted medical charges based on these inputs.

## Dependencies

The `requirements.txt` file includes the following dependencies:

- `streamlit`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Contribution

Feel free to contribute to this project by submitting issues or pull requests. For any questions or suggestions, please reach out.





