from flask import Flask,render_template,request,jsonify
import numpy as np
import joblib

app=Flask(__name__)

f=open(r'/home/ec2-user/environment/mliris/iris.pkl','rb')
ml_model=joblib.load(f)

@app.route("/",methods=["POST","GET"])
def index():
        return render_template("index.html")

@app.route("/predict",methods=["POST","GET"])
def predict():
        if request.method=="POST":
                sl = np.float64(request.form['SepalLength'])
                sw = np.float64(request.form['SepalWidth'])
                pl= np.float64(request.form['PetalLength'])
                pw =np.float64(request.form['PetalWidth'])
                
                mysample = np.array([sl,sw,pl,pw])
                ex1 = mysample.reshape(1,-1)
                ypred=ml_model.predict(ex1)
                output=ypred[0]
                
                if output==0:
                        ans="Iris-Setosa"
                elif output==1:
                        ans="Iris-Versicolor"
                else:
                        ans="Iris-Virginica"
                
                return render_template("index.html",prediction_text=ans)


if __name__=="__main__":
        app.run(debug=True,host='0.0.0.0',port=8080)
