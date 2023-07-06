import pickle
from flask import Flask, request

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict/<value>', methods=['GET'])
def predict(value):
    try:
        float_value = float(value)
        prediction = model.predict([[float_value]])
        return f"Prediction: {prediction[0]}"
    except ValueError:
        return "Invalid input. Please provide a valid number."

if __name__ == '__main__':
    app.run()
