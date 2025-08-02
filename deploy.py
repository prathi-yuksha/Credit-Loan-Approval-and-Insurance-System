from flask import Flask, render_template, request
import joblib
import pandas as pd
import pickle
app = Flask(__name__)

app.config['STATIC_FOLDER'] = 'static'
app.config['TEMPLATES_AUTO_RELOAD'] = True
with open('loan_approval_model.pkl', 'rb') as file:
    loaded_objects = pickle.load(file)
#loaded_model = loaded_objects[0]
#insurance_model = joblib.load("insurance_recommendation.pkl")
with open('insurance_recommendation.pkl', 'rb') as file:
    insurance_model = pickle.load(file)

with open('loan_prediction.pkl', 'rb') as file:
    load_model = pickle.load(file)

with open('home_loan.pkl', 'rb') as file:
    home_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('homepra.html')

@app.route('/index_type')
def index_types():
    return render_template('index.html')

@app.route('/loan_types')
def loan_types():
    return render_template('loan_approval.html')


@app.route('/normal', methods=['POST', 'GET'])
def loan_approval():
    if request.method == 'POST':
        # Get input values from the form
        input_data = [int(request.form['no_of_dependents']),
                      int(request.form['education']),
                      int(request.form['self_employed']),
                      int(request.form['income_annum']),
                      int(request.form['loan_amount']),
                      int(request.form['loan_term']),
                      int(request.form['cibil_score']),
                      int(request.form['residential_assets_value'])]

        # Extract the model from the loaded dictionary
        loaded_model = loaded_objects.get('model')

        if loaded_model:
            # Make a prediction using the loaded model
            prediction = loaded_model.predict([input_data])
                    # Make sure feature names match those used during training
            if prediction == 1:
                prediction = "Loan Approved"
            if prediction == 0:
                prediction = "Loan was Not Approved"
            print("Loan Approval Prediction:", prediction)

            return render_template('index2.html', prediction=prediction)

    return render_template('index2.html', prediction=None)



@app.route('/home_loan', methods=['POST', 'GET'])
def home_loan():
    if request.method == 'POST':
        # Get input values from the form
        education = int(request.form['education'])
        self_employed = int(request.form['self_employed'])
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])

        # Create a dictionary with input data
        input_data = {
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount]
        }

        # Convert the dictionary to a DataFrame
        input_df = pd.DataFrame(input_data)
        # Use the model to make predictions on the input data
        prediction = home_model.predict(input_df)
        
        if prediction == 1:
            prediction = "Loan Approved"
        if prediction == 0:
            prediction = "Loan was Not Approved"

        return render_template('home_loan_form.html', prediction=prediction)

    return render_template('home_loan_form.html', prediction=None)

@app.route('/personal', methods=['POST', 'GET'])
def loan_input():
    if request.method == 'POST':
        # Get input values from the form
        gender = int(request.form['gender'])  # Assuming 'gender' is numerical
        married = int(request.form['married'])  # Assuming 'married' is numerical
        dependents = int(request.form['dependents'])
        self_employed = int(request.form['self_employed'])  # Assuming 'self_employed' is numerical
        education = int(request.form['education'])  # Assuming 'education' is numerical
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])

        # Create a dictionary with input data
        input_data = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Self_Employed': [self_employed],
            'Education': [education],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount]
        }

        # Convert the dictionary to a DataFrame
        input_df = pd.DataFrame(input_data)

        # Use the model to make predictions on the input data (replace with your actual model)
        prediction = load_model.predict(input_df)
        if prediction == 'Y':
            prediction = "Loan Approved"
        if prediction == 'N':
            prediction = "Loan was Not Approved"
        # You can return the prediction or use it for further processing

        return render_template('loan_input.html', prediction=prediction)

    return render_template('loan_input.html', prediction=None)


@app.route('/insurance_recommendation', methods=['POST', 'GET'])
def insurance_recommendation():
    if request.method == 'POST':
        try:
            # Assuming you have a form on insurance_recommendation.html with necessary input fields
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            occupation = int(request.form['occupation'])
            income = float(request.form['income'])
            premium_amount = float(request.form['premium_amount'])

            # Create a DataFrame with the input features
            insurance_data = pd.DataFrame([[age, gender, occupation, income, premium_amount]],
                                          columns=['Age', 'Gender', 'Occupation', 'Income', 'Premium_Amount'])

            # Print the feature names and recommendations for debugging
            print("Feature names in insurance_data:", insurance_data.columns)

            # Make sure feature names match those used during training
            recommendations = insurance_model.predict(insurance_data)

            # Ensure recommendations is a list with three elements
            if not isinstance(recommendations, list):
                recommendations = [recommendations] * 3

            print("Insurance Recommendations:", recommendations)

            # Pass three recommendations to the template
            return render_template('insurance_recommendation.html', recommendations=recommendations)

        except ValueError as e:
            # Handle the ValueError, for example, by displaying an error message to the user
            error_message = f"Error: {e}"
            return render_template('insurance_recommendation.html', error_message=error_message)

    return render_template('insurance_recommendation.html', recommendations=None)

# ...
# ...

if __name__ == '__main__':
    app.run(debug=True)
