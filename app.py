import joblib
from flask import Flask, render_template, request, redirect
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/cropRecommendation')
def crop_recommendation():
    return render_template("cropRecommendation.html")

@app.route('/diseaseIdentification')
def diseaseIdentification_Yeild():
    return render_template("diseaseIdentification.html")

@app.route('/soilManagement')
def soil_Management():
    return render_template("soilManagement.html")

@app.route("/cropRecommendation/predict", methods=['POST'])
def predict():
    Nitrogen=float(request.form['Nitrogen'])
    Phosphorus=float(request.form['Phosphorus'])
    Potassium=float(request.form['Potassium'])
    Temperature=float(request.form['Temperature'])
    Humidity=float(request.form['Humidity'])
    Ph=float(request.form['ph'])
    Rainfall=float(request.form['Rainfall'])
     
    values=[Nitrogen,Phosphorus,Potassium,Temperature,Humidity,Ph,Rainfall]
    
    if Ph>0 and Ph<=14 and Temperature<100 and Humidity>0:
        joblib.load('crop_app','r')
        model = joblib.load(open('crop_app','rb'))
        arr = [values]
        acc = model.predict(arr)
        # print(acc)
        return render_template('prediction.html', prediction=str(acc))
    else:
        return "Sorry...  Error in entered values in the form Please check the values and fill it again"




if __name__ == "__main__":
    app.run(debug=True)  



