from flask import Flask, render_template
from flask import jsonify
from flask import request
app = Flask(__name__)




@app.route('/submit', methods=['POST'])
def submit_form():
    name = request.form.get('name')
    email = request.form.get('email')

    # Process the form data (e.g., store it in a database, send an email, etc.)

    response = {
        'message': 'Form submitted successfully.'
    }
    return jsonify(response)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
