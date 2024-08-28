# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  # Add this import statement
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("/Users/ratnameenashivakumar/Downloads/insurance.csv")

# Feature Engineering
# Age Groups
df['Age Group'] = pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 100], labels=['18-30', '30-40', '40-50', '50-60', '60+'])

# BMI Categories
df['BMI Category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 40, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III'])

# One-Hot Encoding for 'sex' and 'region'
df_encoded = pd.get_dummies(df, columns=['sex', 'region'], drop_first=True)

# Title
st.title("Healthcare Insurance Data Analysis - Dashboard")
st.subheader('Data')
st.write(df)

# Sidebar for selecting visualizations
st.sidebar.header("Select Visualizations")

# Old Visualizations

viz_option = st.sidebar.multiselect("Visualizations", ['Age Distribution', 'Gender Distribution', 'BMI Distribution', 'Medical Charges vs Age', 'Medical Charges vs BMI', 'Age Group Distribution', 'BMI Category Distribution', 'Medical Charges vs Age Group', 'Medical Charges vs BMI Category'])



# Distribution of Age
if 'Age Distribution' in viz_option:
    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['age'], kde=True, ax=ax)
    st.pyplot(fig)

# Gender distribution
if 'Gender Distribution' in viz_option:
    st.write("### Gender Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='sex', data=df, ax=ax)
    st.pyplot(fig)

# BMI distribution
if 'BMI Distribution' in viz_option:
    st.write("### BMI Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['bmi'], kde=True, ax=ax)
    st.pyplot(fig)

# Medical charges vs Age
if 'Medical Charges vs Age' in viz_option:
    st.write("### Medical Charges vs Age")
    fig, ax = plt.subplots()
    sns.scatterplot(x='age', y='charges', data=df, ax=ax)
    st.pyplot(fig)

# Medical charges vs BMI
if 'Medical Charges vs BMI' in viz_option:
    st.write("### Medical Charges vs BMI")
    fig, ax = plt.subplots()
    sns.scatterplot(x='bmi', y='charges', data=df, ax=ax)
    st.pyplot(fig)

# New Visualizations
#viz_option = st.sidebar.multiselect("Choose New Visualizations", ['Age Group Distribution', 'BMI Category Distribution', 'Medical Charges vs Age Group', 'Medical Charges vs BMI Category'])

# Age Group distribution
if 'Age Group Distribution' in viz_option:
    st.write("### Age Group Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Age Group', data=df, ax=ax)
    st.pyplot(fig)

# BMI Category distribution
if 'BMI Category Distribution' in viz_option:
    st.write("### BMI Category Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='BMI Category', data=df, ax=ax)
    st.pyplot(fig)

# Updated Medical charges vs Age with a scatter plot
if 'Medical Charges vs Age Group' in viz_option:
    st.write("### Medical Charges vs Age Group")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age Group', y='charges', data=df, ax=ax)
    st.pyplot(fig)

# Updated Medical charges vs BMI with a scatter plot
if 'Medical Charges vs BMI Category' in viz_option:
    st.write("### Medical Charges vs BMI Category")
    fig, ax = plt.subplots()
    sns.scatterplot(x='BMI Category', y='charges', data=df, ax=ax)
    st.pyplot(fig)

# Add more visualizations as needed

# Footer
st.sidebar.markdown(
    "This app allows you to explore the healthcare insurance dataset with a dashboard-like experience. "
    "Feel free to customize the visualizations using the options on the sidebar."
)



# Features and target variable
X = df[['age', 'bmi', 'smoker']]
y = df['charges']

# One-Hot Encoding for 'smoker'
df_encoded = pd.get_dummies(df, columns=['smoker'], drop_first=True)

# Ensure the correct column names are used
X_encoded = df_encoded[['age', 'bmi', 'smoker_yes']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Features and target variable
X = df[['age', 'bmi', 'smoker']]
y = df['charges']

# One-Hot Encoding for 'smoker'
df_encoded = pd.get_dummies(df, columns=['smoker'], drop_first=True)

# Ensure the correct column names are used
X_encoded = df_encoded[['age', 'bmi', 'smoker_yes']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the results
st.write("##### Predicting Medical Charges")
st.write("##### Features: Age, BMI, Smoking Status")
st.write(f"Mean Squared Error: {mse}")
st.write(f"R-squared Score: {r2}")
st.write("Mean Squared Error (MSE):"
"The MSE is a measure of the average squared difference between the actual and predicted values."
"In this case, the MSE value of approximately 34,512,843.88 indicates the average squared difference between the actual and predicted medical charges. "
"A lower MSE is generally better, indicating less deviation from the true values."
"  "
"R-squared Score:"
"The R-squared score (or coefficient of determination) represents the proportion of the variance in the dependent variable (medical charges) that is predictable from the independent variables (age, BMI, and smoker status)."
"Your R-squared score of approximately 0.778 (77.77%) suggests that about 77.8% of the variance in medical charges can be explained by the features included in your model. A higher R-squared value indicates a better fit of the model to the data.")
st.write("Model seems to be performing reasonably well, explaining a significant portion of the variance in medical charges.")

# User input for prediction
st.sidebar.header("Predict Medical Charges")
age_input = st.sidebar.slider("Age:", min_value=18, max_value=100, value=30)
bmi_input = st.sidebar.slider("BMI:", min_value=10, max_value=50, value=25)
smoker_input = st.sidebar.checkbox("Smoker")

# Make a prediction based on user input
user_data = pd.DataFrame({'age': [age_input], 'bmi': [bmi_input], 'smoker_yes': [smoker_input]})

# Ensure the model is fitted before making predictions
if model is not None:
    # Ensure the feature names match those used during training
    user_data_encoded = user_data[['age', 'bmi', 'smoker_yes']]  # Select the relevant columns for prediction
    user_data_encoded = pd.get_dummies(user_data_encoded, columns=['smoker_yes'], drop_first=True)
    user_data_encoded = user_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    prediction = model.predict(user_data_encoded)

    # Display the prediction
    st.sidebar.subheader("Predicted Medical Charges:")
    st.sidebar.write(f"${prediction[0]:,.2f}")
else:
    st.sidebar.subheader("Model not fitted. Please fit the model before making predictions.")