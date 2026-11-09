import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import joblib to save the trained machine learning model into a file
import joblib


# Import train_test_split to divide the dataset into training and testing data
from sklearn.model_selection import train_test_split

# Import ColumnTransformer to apply different preprocessing to different columns
from sklearn.compose import ColumnTransformer

# Import OneHotEncoder to convert categorical text values into numerical values
# Import StandardScaler to scale numerical features
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import Pipeline to combine preprocessing and machine learning steps
from sklearn.pipeline import Pipeline

# Import Linear Regression model
from sklearn.linear_model import LinearRegression

# Import Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor

# Import evaluation metrics to measure model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Read the CSV dataset
# sep=";" means values are separated using semicolons
df = pd.read_csv("student-mat.csv", sep=";")
print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:", df.head())
print("\nDataset Information:", df.info())
print("\nMissing Values:", df.isnull().sum())


# Display statistical information about the final grade column G3
# It includes count, mean, standard deviation, minimum, and maximum
print("\nFinal Grade Statistics:")
print(df["G3"].describe())

# Select only numerical columns from the dataset
numeric_df = df.select_dtypes(include=np.number)

# Create one figure containing three graphs side by side
# figsize controls the width and height of the complete figure
# axes contains the three individual plotting areas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))


# Create a histogram of students' final grades
# bins=10 divides the grades into ten intervals
# edgecolor="black" adds black borders to the bars
axes[0].hist(df["G3"], bins=10, edgecolor="black")

# Set the label of the x-axis
axes[0].set_xlabel("Final Grade")
# Set the label of the y-axis
axes[0].set_ylabel("Number of Students")
# Set the title of the first graph
axes[0].set_title("Distribution of Final Grades")


# Create a scatter plot showing study time and final grades
# Each point represents one student
axes[1].scatter(df["studytime"], df["G3"])

# Set the x-axis label
axes[1].set_xlabel("Study Time")
# Set the y-axis label
axes[1].set_ylabel("Final Grade")
# Set the title of the second graph
axes[1].set_title("Study Time vs Final Grade")


# Calculate the correlation between numerical columns
# Correlation values range from -1 to +1
correlation = numeric_df.corr()

# Display the correlation matrix as a heatmap
# cmap="coolwarm" represents different correlation values using colors
# aspect="auto" adjusts the heatmap according to the available space
im = axes[2].imshow(
    correlation,
    cmap="coolwarm",
    aspect="auto"
)

# Set the positions of the x-axis ticks
axes[2].set_xticks(range(len(numeric_df.columns)))

# Set the positions of the y-axis ticks
axes[2].set_yticks(range(len(numeric_df.columns)))

# Display numerical column names on the x-axis
# rotation=90 rotates the labels vertically
axes[2].set_xticklabels( numeric_df.columns, rotation=90 )
# Display numerical column names on the y-axis
axes[2].set_yticklabels(numeric_df.columns)
# Set the title of the heatmap
axes[2].set_title("Correlation Heatmap")
# Add a colorbar to explain the correlation colors
# fraction controls its size
# pad controls the distance between heatmap and colorbar
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

# Adjust spacing so that graphs and labels do not overlap
plt.tight_layout()

# Display all three EDA graphs together
plt.show()


# --------------------------------------------------
# PREPARE FEATURES AND TARGET
# --------------------------------------------------

# X contains the input features used to predict the final grade
# G1 and G2 are removed because they contain previous exam grades
# G3 is also removed because it is the value we want to predict
X = df.drop(columns=["G1", "G2", "G3"])

# y contains the target variable, which is the final grade G3
y = df["G3"]
print("\nFeatures:", X.columns.tolist())
print("Target: G3")


# --------------------------------------------------
# IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# --------------------------------------------------

# Select numerical input features such as age, studytime, and absences
# int64 and float64 represent numerical data types
numerical_features = X.select_dtypes( include=["int64", "float64"] ).columns.tolist()

# Select categorical input features such as school, sex, and address
# object represents text data in pandas
categorical_features = X.select_dtypes( include=["object"] ).columns.tolist()

# Print the numerical feature names
print("\nNumerical Features:", numerical_features)

# Print the categorical feature names
print("\nCategorical Features:", categorical_features)


# --------------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------------

# Create a preprocessing object that handles numerical
# and categorical columns differently
preprocessor = ColumnTransformer(
    transformers=[
    # Inside transformers, each transformation has three parts:
    # ("name", transformer, columns)
        # Apply StandardScaler to numerical features
        # It standardizes values using mean and standard deviation
        (
            "num",
            StandardScaler(),
            numerical_features
        ),

        # Apply OneHotEncoder to categorical features
        # It converts text categories into numerical columns
        # handle_unknown="ignore" avoids errors for unseen categories
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# SPLIT DATASET
# --------------------------------------------------

# Divide the dataset into training and testing parts
# test_size=0.2 means 20% is used for testing
# 80% is used for training
# random_state=42 ensures the same split every time
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Print the number of training samples
print("\nTraining Samples:", len(X_train))

# Print the number of testing samples
print("Testing Samples:", len(X_test))


# --------------------------------------------------
# CREATE MACHINE LEARNING MODELS
# --------------------------------------------------

# Create a dictionary containing two regression models
models = {

    # Linear Regression learns a linear relationship
    # between input features and the target grade
    "Linear Regression": LinearRegression(),

    # Random Forest combines multiple decision trees
    # n_estimators=100 means it creates 100 trees
    # random_state=42 makes the model reproducible
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}


# --------------------------------------------------
# TRAIN AND EVALUATE MODELS
# --------------------------------------------------

# Create an empty dictionary to store model results
results = {}

# Loop through each model in the models dictionary
# name stores the model name
# model stores the actual model object
for name, model in models.items():

    # Create a pipeline combining preprocessing and model training
    pipeline = Pipeline(
        steps=[

            # First, preprocess the input features
            ("preprocessing", preprocessor),

            # Then, train the selected machine learning model
            ("model", model)
        ]
    )

    # Train the pipeline using training data
    # Preprocessing is learned from X_train only
    pipeline.fit(X_train, y_train)

    # Predict final grades for the test dataset
    y_pred = pipeline.predict(X_test)

    # Calculate Mean Absolute Error
    # It measures the average absolute difference
    # between actual and predicted grades
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate Root Mean Squared Error
    # It penalizes larger prediction errors more strongly
    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    # Calculate R2 Score
    # It measures how well the model explains variation in grades
    # A higher R2 is generally better
    r2 = r2_score(y_test, y_pred)

    # Store the evaluation metrics and trained pipeline
    # under the current model name
    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
        "Pipeline": pipeline
    }

    print("\n------------------------------")
    print("Model:", name)
    print("------------------------------")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))


# --------------------------------------------------
# COMPARE MODEL PERFORMANCE
# --------------------------------------------------

# Create a DataFrame containing evaluation metrics
# Each row represents a model
# Each column represents an evaluation metric
comparison = pd.DataFrame({
    name: {
        "MAE": result["MAE"],
        "RMSE": result["RMSE"],
        "R2 Score": result["R2 Score"]
    }
    for name, result in results.items()
}).T

print("\nModel Comparison:", comparison)

# Select the model with the lowest MAE
# idxmin() returns the name of the row with the minimum value
best_model_name = comparison["MAE"].idxmin()

# Retrieve the trained pipeline of the best model
best_pipeline = results[best_model_name]["Pipeline"]

print("\nBest Model:", best_model_name)


# --------------------------------------------------
# MODEL COMPARISON GRAPHS
# --------------------------------------------------

# Create one figure with three graphs side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))


# -----------------------------
# Graph 1: MAE Comparison
# -----------------------------

# Plot MAE values of both models as a bar chart
# ax=axes[0] places the graph in the first plotting area
comparison["MAE"].plot(
    kind="bar",
    ax=axes[0],
    edgecolor="black"
)

# Set the y-axis label
axes[0].set_ylabel("MAE")

# Set the graph title
axes[0].set_title("MAE Comparison")

# Rotate x-axis labels slightly for better readability
axes[0].tick_params(axis="x", rotation=15)


# -----------------------------
# Graph 2: RMSE Comparison
# -----------------------------

# Plot RMSE values of both models
comparison["RMSE"].plot(
    kind="bar",
    ax=axes[1],
    edgecolor="black"
)

# Set the y-axis label
axes[1].set_ylabel("RMSE")

# Set the graph title
axes[1].set_title("RMSE Comparison")

# Rotate x-axis labels
axes[1].tick_params(axis="x", rotation=15)


# -----------------------------
# Graph 3: R2 Score Comparison
# -----------------------------

# Plot R2 scores of both models
comparison["R2 Score"].plot(
    kind="bar",
    ax=axes[2],
    edgecolor="black"
)

# Set the y-axis label
axes[2].set_ylabel("R2 Score")

# Set the graph title
axes[2].set_title("R2 Score Comparison")

# Rotate x-axis labels
axes[2].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# PREDICT A NEW STUDENT'S FINAL GRADE
# --------------------------------------------------

# Create a DataFrame containing information about a new student
# The column names and values must match the training features
new_student = pd.DataFrame([{

    # Student's school: GP or MS
    "school": "GP",

    # Student's sex
    "sex": "F",

    # Student's age
    "age": 17,

    # Student's address type: U or R
    "address": "U",

    # Family size: GT3 means greater than 3
    "famsize": "GT3",

    # Parent's cohabitation status
    "Pstatus": "T",

    # Mother's education level
    "Medu": 3,

    # Father's education level
    "Fedu": 2,

    # Mother's job
    "Mjob": "services",

    # Father's job
    "Fjob": "other",

    # Reason for choosing the school
    "reason": "course",

    # Student's guardian
    "guardian": "mother",

    # Travel time to school
    "traveltime": 1,

    # Weekly study time
    "studytime": 3,

    # Number of previous class failures
    "failures": 0,

    # Whether the student receives school support
    "schoolsup": "no",

    # Whether the student receives family support
    "famsup": "yes",

    # Whether the student has paid extra classes
    "paid": "no",

    # Whether the student participates in activities
    "activities": "yes",

    # Whether the student attended nursery school
    "nursery": "yes",

    # Whether the student wants higher education
    "higher": "yes",

    # Whether the student has internet access
    "internet": "yes",

    # Whether the student is in a romantic relationship
    "romantic": "no",

    # Quality of family relationships
    "famrel": 4,

    # Amount of free time
    "freetime": 3,

    # Time spent going out with friends
    "goout": 2,

    # Workday alcohol consumption
    "Dalc": 1,

    # Weekend alcohol consumption
    "Walc": 2,

    # Current health status
    "health": 5,

    # Number of school absences
    "absences": 2
}])

# Use the selected trained model to predict the student's final grade
# [0] extracts the first prediction from the returned array
predicted_grade = best_pipeline.predict(new_student)[0]

# Restrict the predicted grade to the range 0 to 20
# np.clip prevents values below 0 or above 20
predicted_grade = np.clip(predicted_grade, 0, 20)

print("\n------------------------------")
print("New Student Prediction")
print("------------------------------")

print("Predicted Final Grade:", round(predicted_grade, 2))


# --------------------------------------------------
# SAVE THE BEST MODEL
# --------------------------------------------------

# Save the trained pipeline to a file named best_model.pkl
# The saved file contains preprocessing and the trained model
joblib.dump(best_pipeline, "best_model.pkl")
print("\nModel saved successfully as best_model.pkl")
