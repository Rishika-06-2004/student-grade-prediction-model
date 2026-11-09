
# Student Performance Prediction Using Machine Learning

## Objective
To predict student mathematics performance using machine learning based on academic, social, and personal factors.

## Dataset
UCI Student Performance Dataset.

The dataset contains student information and final mathematics grades.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Algorithms
1. Linear Regression
2. Random Forest Regressor

## Methodology
1. Data Collection
2. Exploratory Data Analysis
3. Data Preprocessing
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Prediction

## Evaluation Metrics
- MAE
- RMSE
- R2 Score

## Result
The models are compared using MAE, RMSE, and R2 Score.

The model with the lowest MAE is selected for prediction.

## Limitation
The prediction is based on a limited dataset.
It should not be used to make important decisions about students.

## Pipeline of this project:

 --------------------------------
| load dataset( student-mat.csv) |
 --------------------------------
               |
               v
 -------------------------------------------
| Explore dataset with statisticsand graphs |
 -------------------------------------------
               |
               v
 -------------------------------
| Separate input X and target y |
 -------------------------------
               |
               v
 ------------------------------------------
| Preprocess numerical andcategorical data |
 ------------------------------------------
               |
               v
 -------------------------------------
| Split into training and testingdata |
 -------------------------------------
              |
              v
 ------------------------------------------
| Train Linear Regression andRandom Forest |
 ------------------------------------------
              |
              v
 -----------------------------------
| Evaluate models using MAE,RMSE, R2 |
 -----------------------------------
              |
              v
 ------------------------------
| Select model with lowest MAE |
 ------------------------------
              |
              v
 -------------------------------
| Predict a new student's grade |
 -------------------------------
            |
            v
 --------------------
| Save trained model |
 --------------------