from flask import Flask, render_template, jsonify, request
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
df_drug = pd.read_csv("C:\\Users\\pratham\\Downloads\\drug200.csv")

# Train multiple classifiers
NBclassifier = CategoricalNB()
RFclassifier = RandomForestClassifier()
LRclassifier = LogisticRegression()

X = df_drug.drop(["Drug"], axis=1)
y = df_drug["Drug"]
X = pd.get_dummies(X)

NBclassifier.fit(X, y)
RFclassifier.fit(X, y)
LRclassifier.fit(X, y)
@app.route('/submit', methods=['POST'])
def submit_form():
    age = request.form.get('age')
    gender = request.form.get('gender')
    na_k_ratio = request.form.get('na_k_ratio')
    cholesterol = request.form.get('cholesterol')
    blood_pressure = request.form.get('blood_pressure')

    # Create the input for prediction
    input_data = {
        'Age': int(age),
        'Sex': gender,
        'BP': blood_pressure,
        'Cholesterol': cholesterol,
        'Na_to_K': float(na_k_ratio)
    }
    input_df = pd.DataFrame(input_data, index=[0])

    # Apply one-hot encoding to input data
    input_df = pd.get_dummies(input_df)
    
    # Realign columns with training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Make predictions using multiple classifiers
    nb_prediction = NBclassifier.predict(input_df)
    rf_prediction = RFclassifier.predict(input_df)
    lr_prediction = LRclassifier.predict(input_df)

    
    # Determine the prediction from the classifier with the highest accuracy
    predictions = [nb_prediction[0], rf_prediction[0], lr_prediction[0]]
    prediction = max(set(predictions), key=predictions.count)

 
    response = {
       
        'prediction': prediction
    }
    return jsonify(response)



    # # Make predictions using multiple classifiers
    # nb_prediction = NBclassifier.predict(input_df)
    # rf_prediction = RFclassifier.predict(input_df)
    # lr_prediction = LRclassifier.predict(input_df)

    # # Determine the prediction from the classifier with the highest accuracy
    # predictions = [nb_prediction[0], rf_prediction[0], lr_prediction[0]]
    # prediction = max(set(predictions), key=predictions.count)

    # response = {
       
    #     'prediction': prediction
    # }
    # return jsonify(response)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
