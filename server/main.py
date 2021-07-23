import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, File
import pycaret.classification as pycl
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

app = FastAPI()

class Model:
    def __init__(self, modelname, bucketname):
        self.model = pycl.load_model(modelname, platform = "aws", authentication = {  'bucket' : bucketname})

    def predict(self, data):
         predictions = pycl.predict_model(self.model, data = data).Label.to_list()
         return predictions

et = Model('et_deployed', 'mlopsassignment')
rf = Model('rf_deployed', 'mlopsassignment')

def check_env():
    if os.getenv("AWS_ACCESS_KEY_ID") == None or os.getenv("AWS_SECRET_ACCESS_KEY") == None:
        print("AWS Credentials missing. Please set required environment variables.")
        exit(1)

def read_file(fi):
    with open(fi.filename, "wb") as f:
        f.write(fi.file.read())
    data = pd.read_csv(fi.filename)
    os.remove(fi.filename)
    return data

@app.post("/et/predict")

async def create_upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        data = read_file(file)
        try:
            pred = et.predict(data)
            return {
                "Labels" : pred
            }
        except:
            raise HTTPException(status_code = 422, detail = "Invalid csv format")
    else:
        raise HTTPException(status_code = 400, detail = "Invalid file format. Only CSV files accepted.")
    check_env()

@app.post("/rf/predict")

async def create_upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        data = read_file(file)
        try:
            pred = rf.predict(data)
            return {
                "Labels" : pred
            }
        except:
            raise HTTPException(status_code = 422, detail = "Invalid csv format")
    else:
        raise HTTPException(status_code = 400, detaile = "Invalid file format. Only CSV files accepted.")
    check_env()
