# Student Performance Prediction Using Machine Learning

## Objective
To predict a student's mathematics performance using machine learning based on academic, social, and personal factors.

This project compares two regression algorithms and provides a simple **Streamlit web application** where users can enter student details and view the predicted mathematics grade.

## Dataset
**UCI Student Performance Dataset**

- Dataset file: `student-mat.csv`
- Contains student information and final mathematics grades.
- Target variable: `G3` (final mathematics grade)

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning Algorithms
1. Linear Regression
2. Random Forest Regressor

## Methodology
1. Data Collection
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Best Model Selection
8. Student Grade Prediction
9. Model Saving
10. Streamlit Web Application

## Evaluation Metrics
- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted grades.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving more weight to larger errors.
- **R² Score:** Measures how well the model explains the variation in student grades.

## Project Pipeline

          Load Dataset
        (student-mat.csv)
                |
                v
    Exploratory Data Analysis
       (Statistics & Graphs)
                |
                v
     Separate Input X and Target y
                |
                v
   Preprocess Numerical and Categorical
                |
                v
       Split Training and Testing
                |
                v
     Train Linear Regression and
        Random Forest Regressor
                |
                v
       Evaluate Both Models
         (MAE, RMSE, R²)
                |
                v
    Select Model with Lowest MAE
                |
                v
      Predict Student's Grade
                |
                v
       Save Trained Model
        (best_model.pkl)
                |
                v
       Streamlit Web Application
                |
                v
   Enter Student Details in Browser
                |
                v
      Display Predicted Grade


## Web Application
The project includes a web interface developed using **Streamlit**.

Users can:
- Enter a student's academic, social, and personal details.
- Submit the details through the web interface.
- Get the predicted mathematics grade.
- View a simple performance category based on the predicted grade.

## How to Run the Project

### 1. Install Dependencies

```bash
python3 -m pip install pandas numpy matplotlib scikit-learn joblib streamlit
```

### 2. Train the Models

Run the main machine learning script:

```bash
python3 main.py
```

This performs EDA, trains and evaluates both models, and saves the selected model as `best_model.pkl`.

### 3. Start the Streamlit Web Application

```bash
streamlit run app.py
```

The application will open in your browser. If it does not open automatically, use the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Result
Linear Regression and Random Forest Regressor are compared using MAE, RMSE, and R² Score.

The model with the lowest MAE is selected and saved for making predictions through the Streamlit web application.

## Project Structure

```text
student-grade-prediction-model/
│
├── student-mat.csv
├── main.py
├── app.py
├── best_model.pkl
├── requirements.txt
└── README.md
```

## Limitation
- The prediction is based on a limited dataset.
- The model may not generalize to all students or educational environments.
- Predictions are estimates and should not be used to make important decisions about students.
