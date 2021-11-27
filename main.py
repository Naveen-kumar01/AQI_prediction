from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import json

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/home2', methods=['GET'])
@cross_origin()
def home2():
    return render_template('head.html')

class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def ValuePredictor(to_predict_list):
	loaded_model=pickle.load(open("basic_rf_model.pkl", "rb"))
	result = loaded_model.predict(to_predict_list)
	return result[0]

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		pred_list = request.form.to_dict()
		pred_list['City'] = (pred_list['City'])
		pred_list['PM2.5'] = float(pred_list['PM2.5'])
		pred_list['PM10'] = float(pred_list['PM10'])
		pred_list['NO'] = float(pred_list['NO'])
		pred_list['NO2'] = float(pred_list['NO2'])
		pred_list['NOx'] = float(pred_list['NOx'])
		pred_list['NH3'] = float(pred_list['NH3'])
		pred_list['CO'] = float(pred_list['CO'])
		pred_list['SO2'] = float(pred_list['SO2'])
		pred_list['O3'] = float(pred_list['O3'])
		pred_list['Benzene'] = float(pred_list['Benzene'])
		pred_list['Toluene'] = float(pred_list['Toluene'])
		pred_list['Xylene'] = float(pred_list['Xylene'])
		sample_input = pd.DataFrame(pd.Series(pred_list)).T
		result = ValuePredictor(sample_input)
		result = round(result, 2)
	return render_template("res.html", prediction=result)

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']

            pred_val = pred_validation(path) #object initialization

            pred_val.prediction_validation() #calling the prediction_validation function

            pred = prediction(path) #object initialization

            # predicting for dataset present in database
            path, json_predictions = pred.predictionFromModel()
            return Response("Prediction File created at !!!"  +str(path) +'and few of the predictions are '+str(json.loads(json_predictions) ))
        elif request.form is not None:
            path = request.form['filepath']

            pred_val = pred_validation(path) #object initialization

            pred_val.prediction_validation() #calling the prediction_validation function

            pred = prediction(path) #object initialization

            # predicting for dataset present in database
            path,json_predictions = pred.predictionFromModel()
            return Response("Prediction File created at !!!"  +str(path) +'and few of the predictions are '+str(json.loads(json_predictions) ))
        else:
            print('Nothing Matched')
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)



@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRouteClient():

    try:
        #if request.json['folderPath'] is not None:
            #path = request.json['folderPath']
        folderpath = "Training_Batch_Files"
        if folderpath is not None:
            path = folderpath
            train_valObj = train_validation(path) #object initialization

            train_valObj.train_validation()#calling the training_validation function


            trainModelObj = trainModel() #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table


    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")

port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    #port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
