from fastapi import FastAPI
import pandas as pd
import glob

app = FastAPI()

DATA_FOLDER = "processed_data"


@app.get("/")
def home():
    return {"message": "Anomaly Detection Backend Running"}


@app.get("/get-data")
def get_data():

    files = glob.glob(f"{DATA_FOLDER}/*.csv")

    if not files:
        return {"message": "No data available"}

    df_list = [pd.read_csv(f) for f in files]

    df = pd.concat(df_list)

    return df.to_dict(orient="records")