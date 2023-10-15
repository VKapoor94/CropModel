# Crop Recommendation System Using Machine Learning

# Tech stack
Python: Programming language used for data analysis and model development.
Pandas: Data manipulation library for csv or similar format data.
NumPy: Library for numerical computing and to work on arrays.
Scikit-learn: Machine learning library used for model training, evaluation, and prediction.
Flask: Web framework used for building the API.

# Installation and Usage
Install dependencies: pip install -r requirements.txt
Run the app: python app.py
Details of endpoint and purpose:
1. Crop prediction ([POST] method)
    Endpoint :  http://localhost:5000/predict
    
    input body : {
  "Potassium": 5,
  "Nitrogen": 10,
  "Phosporus": 8,
  "Humidity": 80,
  "Ph": 5.8,
  "Rainfall":108,
  "Temperature": 27
}
2. Input feature ranges ([GET] method)
    Endpoint : http://localhost:5000/ranges 
