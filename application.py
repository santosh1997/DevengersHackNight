import json
import os
from flask import Flask,jsonify,request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)
@app.route("/getLeadScore/",methods=['GET'])
def return_price():
    LeadSource = float(request.args.get('LeadSource'))
    DoNotEmail = float(request.args.get('DoNotEmail'))
    DoNotCall = float(request.args.get('DoNotCall'))
    Converted = float(request.args.get('Converted'))
    Country = float(request.args.get('Country'))
    WhatIsYourCurrentOccupation = float(request.args.get('WhatIsYourCurrentOccupation'))
    City = float(request.args.get('City'))
    AsymmetriqueProfileIndex = float(request.args.get('AsymmetriqueProfileIndex'))
    Product = float(request.args.get('Product'))

    with open('leadPredict.pkl', 'rb') as handle:
        leadPredict = pickle.load(handle)

    result = leadPredict.predict(np.array([[LeadSource,DoNotEmail,DoNotCall,Converted,Country,WhatIsYourCurrentOccupation,City,AsymmetriqueProfileIndex,Product]]))
    response = {
                'model':'lr',
                'score': result[0],
                }
    print(response)
    return jsonify(response)

@app.route("/",methods=['GET'])
def default():
    return "<h1> Welcome to Lead Score Predictor <h1>"

if __name__ == "__main__":
    app.run() 