# pyright: reportMissingModuleSource=false
# pyright: reportShadowedImports=false
# pyright: reportMissingImports=false

import os
from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
import pandas as pd
from src import constants
from src.logger import logging

application=Flask(__name__)

app=application

app.config['UPLOAD_FOLDER'] = constants.DATA_FOLDER_PATH
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
results_dicts = dict()

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/',  methods=["POST", "GET"])
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']
 
        logging.info('flask upload file to database (defined uploaded folder in static path)')
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], constants.FILE_NAME))

        logging.info('Initiate Train Pipeline to data built')
        initiate_data_obj = TrainPipeline()
        initiate_data_obj.initiate_data_built()

        result = 1
        return render_template('index.html', results = result)
    
 
@app.route('/viewData', methods= ['GET','POST'])
def showData():
    if request.method=='GET':
        return render_template('viewData.html')
    else:
        uploaded_df = pd.read_csv(constants.RAW_DATA_FILE_PATH)
    
        logging.info('pandas dataframe to html table flask')
        uploaded_df_html = uploaded_df.to_html()

        return render_template('viewData.html', data = uploaded_df_html)
 


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('predict.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction") 
        return render_template('predict.html',results=results[0])
    

@app.route('/model/<model_name>',methods=['GET','POST'])
def model(model_name):
    if request.method=='GET':
            if str(model_name) not in results_dicts.keys():
                logging.info('Initiating Train pipeline to call {}'.format(str(model_name)))
                train = TrainPipeline()
                result_dict = train.train_pipe(str(model_name), results_dicts)
            else:
                logging.info('Model Exists !!! Fetching diectly from results dict ..')
                result_dict = results_dicts

            return render_template('model.html', model = model_name, results = result_dict[str(model_name)])

    else:
        return render_template('model.html')
    


if __name__=="__main__":
    app.run(host="0.0.0.0", debug = True)        

