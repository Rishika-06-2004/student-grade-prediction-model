import streamlit as st
import pandas as pd
import joblib

# Configure the Streamlit application page
# page_title sets the browser tab title
# page_icon displays an icon in the browser tab
# layout="wide" uses the full width of the browser
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# Define a function to load the saved machine learning model
# @st.cache_resource stores the loaded model in memory
# This avoids loading the model again every time the app reruns
@st.cache_resource
def load_model():

    # Load the trained pipeline from the saved file
    # The file contains both preprocessing and the trained model
    return joblib.load("best_model.pkl")


# Call the function and store the loaded model
model = load_model()


# --------------------------------------------------
# APPLICATION TITLE
# --------------------------------------------------

# Display the main title of the application
st.title("🎓 Student Performance Prediction")

# Display a short description of the application
st.write(
    "This application predicts a student's final "
    "mathematics grade using Machine Learning."
)

# Display an information message
# The target grade G3 ranges from 0 to 20
# G1 and G2 previous grades are not used as input features
st.info(
    "The predicted grade is on a scale of 0 to 20. "
    "This model does not use G1 or G2 previous grades."
)


# --------------------------------------------------
# STUDENT INPUT FORM
# --------------------------------------------------

# Display the heading for the input form
st.header("Enter Student Details")

# Create a form to collect student information
# The prediction is performed only when the form is submitted
with st.form("student_form"):
    # Display the subsection heading
    st.subheader("Personal Information")

    # Divide the input area into three columns
    col1, col2, col3 = st.columns(3)

    # First column: school, sex, and age
    with col1:

        # Select the student's school
        # GP and MS are the two schools in the dataset
        school = st.selectbox(
            "School",
            ["GP", "MS"]
        )

        # Select the student's sex
        sex = st.selectbox(
            "Sex",
            ["F", "M"]
        )

        # Select the student's age
        # Slider allows values from 15 to 22
        age = st.slider(
            "Age",
            min_value=15,
            max_value=22,
            value=17
        )

    # Second column: address, family size, and parent status
    with col2:

        # Select the student's address type
        # U = Urban, R = Rural
        address = st.selectbox(
            "Address Type",
            ["U", "R"]
        )

        # Select the family size
        # GT3 = Greater than 3 family members
        # LE3 = Less than or equal to 3 family members
        famsize = st.selectbox(
            "Family Size",
            ["GT3", "LE3"]
        )

        # Select the parents' cohabitation status
        # T = Together, A = Apart
        Pstatus = st.selectbox(
            "Parents' Cohabitation Status",
            ["T", "A"]
        )

    # Third column: parents' education and guardian
    with col3:

        # Select mother's education level
        # 0 to 4 represent education levels in the dataset
        Medu = st.selectbox(
            "Mother's Education",
            [0, 1, 2, 3, 4]
        )

        # Select father's education level
        Fedu = st.selectbox(
            "Father's Education",
            [0, 1, 2, 3, 4]
        )

        # Select the student's guardian
        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )

    # Display the subsection heading
    st.subheader("Family and Education")

    # Divide the input area into three columns
    col1, col2, col3 = st.columns(3)

    # List of possible parent job categories
    jobs = [
        "teacher",
        "health",
        "services",
        "at_home",
        "other"
    ]

    # First column: parents' jobs
    with col1:

        # Select mother's job
        Mjob = st.selectbox(
            "Mother's Job",
            jobs
        )

        # Select father's job
        Fjob = st.selectbox(
            "Father's Job",
            jobs
        )

    # Second column: school choice reason and school support
    with col2:

        # Select the reason for choosing the school
        reason = st.selectbox(
            "Reason for Choosing School",
            ["home", "reputation", "course", "other"]
        )

        # Select whether the student receives extra educational support
        schoolsup = st.selectbox(
            "Extra Educational Support",
            ["yes", "no"]
        )

    # Third column: family support and paid classes
    with col3:

        # Select whether the student receives family educational support
        famsup = st.selectbox(
            "Family Educational Support",
            ["yes", "no"]
        )

        # Select whether the student attends paid extra classes
        paid = st.selectbox(
            "Extra Paid Classes",
            ["yes", "no"]
        )

    # Display the subsection heading
    st.subheader("Study Information")

    # Divide the input area into three columns
    col1, col2, col3 = st.columns(3)

    # First column: study time, failures, and absences
    with col1:

        # Select weekly study time
        # 1 = Less than 2 hours
        # 2 = 2 to 5 hours
        # 3 = 5 to 10 hours
        # 4 = More than 10 hours
        studytime = st.selectbox(
            "Weekly Study Time",
            [1, 2, 3, 4],
            index=1
        )

        # Select the number of past class failures
        failures = st.selectbox(
            "Past Class Failures",
            [0, 1, 2, 3]
        )

        # Enter the number of school absences
        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=100,
            value=2
        )

    # Second column: activities, nursery, and higher education
    with col2:

        # Select whether the student participates in extra-curricular activities
        activities = st.selectbox(
            "Extra-Curricular Activities",
            ["yes", "no"]
        )

        # Select whether the student attended nursery school
        nursery = st.selectbox(
            "Attended Nursery School",
            ["yes", "no"]
        )

        # Select whether the student wants to pursue higher education
        higher = st.selectbox(
            "Wants Higher Education",
            ["yes", "no"]
        )

    # Third column: internet, relationship, and travel time
    with col3:

        # Select whether the student has internet access at home
        internet = st.selectbox(
            "Internet Access at Home",
            ["yes", "no"]
        )

        # Select whether the student is in a romantic relationship
        romantic = st.selectbox(
            "In a Romantic Relationship",
            ["yes", "no"]
        )

        # Select travel time to school
        # 1 = Less than 15 minutes
        # 2 = 15 to 30 minutes
        # 3 = 30 minutes to 1 hour
        # 4 = More than 1 hour
        traveltime = st.selectbox(
            "Travel Time to School",
            [1, 2, 3, 4]
        )

    # Display the subsection heading
    st.subheader("Lifestyle Information")

    # Divide the input area into three columns
    col1, col2, col3 = st.columns(3)

    # First column: family relationship and free time
    with col1:

        # Select family relationship quality
        # 1 = Very bad, 5 = Excellent
        famrel = st.slider(
            "Family Relationship Quality",
            1,
            5,
            4
        )

        # Select the amount of free time
        # 1 = Very low, 5 = Very high
        freetime = st.slider(
            "Free Time",
            1,
            5,
            3
        )

    # Second column: going out and workday alcohol consumption
    with col2:

        # Select time spent going out with friends
        goout = st.slider(
            "Going Out with Friends",
            1,
            5,
            2
        )

        # Select workday alcohol consumption
        Dalc = st.slider(
            "Workday Alcohol Consumption",
            1,
            5,
            1
        )

    # Third column: weekend alcohol consumption and health
    with col3:

        # Select weekend alcohol consumption
        Walc = st.slider(
            "Weekend Alcohol Consumption",
            1,
            5,
            2
        )

        # Select the student's current health status
        health = st.slider(
            "Health Status",
            1,
            5,
            5
        )

    # Create a button to submit the form
    # submitted becomes True when the user clicks the button
    submitted = st.form_submit_button(
        "Predict Final Grade"
    )


# --------------------------------------------------
# MAKE PREDICTION
# --------------------------------------------------

# Execute this section only after the user submits the form
if submitted:

    # Create a DataFrame containing the student's input information
    # The column names must match the features used during model training
    new_student = pd.DataFrame([{

        # Student's school
        "school": school,

        # Student's sex
        "sex": sex,

        # Student's age
        "age": age,

        # Student's address type
        "address": address,

        # Student's family size
        "famsize": famsize,

        # Parents' cohabitation status
        "Pstatus": Pstatus,

        # Mother's education level
        "Medu": Medu,

        # Father's education level
        "Fedu": Fedu,

        # Mother's job
        "Mjob": Mjob,

        # Father's job
        "Fjob": Fjob,

        # Reason for choosing the school
        "reason": reason,

        # Student's guardian
        "guardian": guardian,

        # Travel time to school
        "traveltime": traveltime,

        # Weekly study time
        "studytime": studytime,

        # Number of previous class failures
        "failures": failures,

        # Extra educational support
        "schoolsup": schoolsup,

        # Family educational support
        "famsup": famsup,

        # Paid extra classes
        "paid": paid,

        # Extra-curricular activities
        "activities": activities,

        # Attended nursery school
        "nursery": nursery,

        # Wants higher education
        "higher": higher,

        # Internet access at home
        "internet": internet,

        # Romantic relationship status
        "romantic": romantic,

        # Family relationship quality
        "famrel": famrel,

        # Free time
        "freetime": freetime,

        # Going out with friends
        "goout": goout,

        # Workday alcohol consumption
        "Dalc": Dalc,

        # Weekend alcohol consumption
        "Walc": Walc,

        # Health status
        "health": health,

        # Number of absences
        "absences": absences

    }])

    # --------------------------------------------------
    # PREDICT FINAL GRADE
    # --------------------------------------------------

    # Pass the student's input to the trained ML pipeline
    # The pipeline performs preprocessing and prediction automatically
    # [0] extracts the first prediction from the returned array
    prediction = model.predict(new_student)[0]

    # Keep the predicted grade within the dataset's grade range
    # max() prevents values below 0
    # min() prevents values above 20
    prediction = max(0, min(20, prediction))


    # --------------------------------------------------
    # DISPLAY PREDICTION RESULT
    # --------------------------------------------------

    # Display a success message
    st.success("Prediction Completed!")

    # Display the predicted final mathematics grade
    # The grade is shown out of 20
    st.metric(
        label="Predicted Final Mathematics Grade (G3)",
        value=f"{prediction:.2f} / 20"
    )


    # --------------------------------------------------
    # PERFORMANCE INTERPRETATION
    # --------------------------------------------------

    # Display a simple interpretation of the predicted grade
    # These are application-defined labels, not official dataset categories

    if prediction >= 16:

        # Display this message for grades 16 or above
        st.write("**Performance:** Very Good")

    elif prediction >= 12:

        # Display this message for grades from 12 to below 16
        st.write("**Performance:** Good")

    elif prediction >= 10:

        # Display this message for grades from 10 to below 12
        st.write("**Performance:** Average")

    else:

        # Display this message for grades below 10
        st.write("**Performance:** Needs Improvement")

    # Explain that the prediction is an estimate
    # The model cannot guarantee the student's actual final grade
    st.caption(
        "This is a machine learning estimate, not a guaranteed result."
    )
