import pickle
from flask import Flask,request,render_template,jsonify
import numpy as np
from flask_cors import CORS

# loading model
stand_scale = pickle.load(open('standscaler.pkl','rb'))
min_max_scale = pickle.load(open('minmaxscaler.pkl','rb'))
rf_model = pickle.load(open('model.pkl','rb'))

# flask app
app = Flask(__name__)
CORS(app)

@app.route('/ranges',methods=['GET'])
def ranges_for_input():
    # ranges of input feature from trained dataset
    val_ranges = {
        "Potassium": {"start":5,"end":205},
        "Nitrogen": {"start":0,"end":140},
        "Phosporus": {"start":5,"end":145},
        "Humidity": {"start":14,"end":100},
        "Ph": {"start":3,"end":10},
        "Rainfall":{"start":20,"end":300},
        "Temperature": {"start":14,"end":43}
        }
    return jsonify(val_ranges)

@app.route("/predict",methods=['POST'])
def predict():
    input = request.json
    N,P,K,temp,humidity,ph,rainfall = input['Nitrogen'],input['Phosporus'],input['Potassium'],input['Temperature'],\
                                        input['Humidity'],input['Ph'],input['Rainfall']

    
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = min_max_scale.transform(single_pred)
    final_features = stand_scale.transform(scaled_features)
    prediction = rf_model.predict(final_features)

    crop_dict = {'rice': 1,
 'maize': 2,
 'chickpea': 3,
 'kidneybeans': 4,
 'pigeonpeas': 5,
 'mothbeans': 6,
 'mungbean': 7,
 'blackgram': 8,
 'lentil': 9,
 'pomegranate': 10,
 'banana': 11,
 'mango': 12,
 'grapes': 13,
 'watermelon': 14,
 'muskmelon': 15,
 'apple': 16,
 'orange': 17,
 'papaya': 18,
 'coconut': 19,
 'cotton': 20,
 'jute': 21,
 'coffee': 22}
    
    crop_dict = {v: k for k, v in crop_dict.items()}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        output = jsonify({"result":crop,"status":"success","org_val":str(prediction[0])})
    else:
        output = jsonify({"result":"","status":"failed","org_val":str(prediction[0])})
    return output

if __name__ == "__main__":
    app.run(debug=True)