from fastapi import FastAPI
import uvicorn
import pandas as pd
import json
from pydantic import BaseModel
from src.predict import run_predict

# Initiate the API
app = FastAPI()


# Structure of drive stats data to receive via the API
class Item(BaseModel):
    date: dict
    serial_number: dict
    model: dict
    capacity_bytes: dict
    failure: dict
    smart_1_normalized: dict
    smart_1_raw: dict
    smart_2_normalized: dict
    smart_2_raw: dict
    smart_3_normalized: dict
    smart_3_raw: dict
    smart_4_normalized: dict
    smart_4_raw: dict
    smart_5_normalized: dict
    smart_5_raw: dict
    smart_7_normalized: dict
    smart_7_raw: dict
    smart_8_normalized: dict
    smart_8_raw: dict
    smart_9_normalized: dict
    smart_9_raw: dict
    smart_10_normalized: dict
    smart_10_raw: dict
    smart_11_normalized: dict
    smart_11_raw: dict
    smart_12_normalized: dict
    smart_12_raw: dict
    smart_13_normalized: dict
    smart_13_raw: dict
    smart_15_normalized: dict
    smart_15_raw: dict
    smart_16_normalized: dict
    smart_16_raw: dict
    smart_17_normalized: dict
    smart_17_raw: dict
    smart_18_normalized: dict
    smart_18_raw: dict
    smart_22_normalized: dict
    smart_22_raw: dict
    smart_23_normalized: dict
    smart_23_raw: dict
    smart_24_normalized: dict
    smart_24_raw: dict
    smart_160_normalized: dict
    smart_160_raw: dict
    smart_161_normalized: dict
    smart_161_raw: dict
    smart_163_normalized: dict
    smart_163_raw: dict
    smart_164_normalized: dict
    smart_164_raw: dict
    smart_165_normalized: dict
    smart_165_raw: dict
    smart_166_normalized: dict
    smart_166_raw: dict
    smart_167_normalized: dict
    smart_167_raw: dict
    smart_168_normalized: dict
    smart_168_raw: dict
    smart_169_normalized: dict
    smart_169_raw: dict
    smart_170_normalized: dict
    smart_170_raw: dict
    smart_171_normalized: dict
    smart_171_raw: dict
    smart_172_normalized: dict
    smart_172_raw: dict
    smart_173_normalized: dict
    smart_173_raw: dict
    smart_174_normalized: dict
    smart_174_raw: dict
    smart_175_normalized: dict
    smart_175_raw: dict
    smart_176_normalized: dict
    smart_176_raw: dict
    smart_177_normalized: dict
    smart_177_raw: dict
    smart_178_normalized: dict
    smart_178_raw: dict
    smart_179_normalized: dict
    smart_179_raw: dict
    smart_180_normalized: dict
    smart_180_raw: dict
    smart_181_normalized: dict
    smart_181_raw: dict
    smart_182_normalized: dict
    smart_182_raw: dict
    smart_183_normalized: dict
    smart_183_raw: dict
    smart_184_normalized: dict
    smart_184_raw: dict
    smart_187_normalized: dict
    smart_187_raw: dict
    smart_188_normalized: dict
    smart_188_raw: dict
    smart_189_normalized: dict
    smart_189_raw: dict
    smart_190_normalized: dict
    smart_190_raw: dict
    smart_191_normalized: dict
    smart_191_raw: dict
    smart_192_normalized: dict
    smart_192_raw: dict
    smart_193_normalized: dict
    smart_193_raw: dict
    smart_194_normalized: dict
    smart_194_raw: dict
    smart_195_normalized: dict
    smart_195_raw: dict
    smart_196_normalized: dict
    smart_196_raw: dict
    smart_197_normalized: dict
    smart_197_raw: dict
    smart_198_normalized: dict
    smart_198_raw: dict
    smart_199_normalized: dict
    smart_199_raw: dict
    smart_200_normalized: dict
    smart_200_raw: dict
    smart_201_normalized: dict
    smart_201_raw: dict
    smart_202_normalized: dict
    smart_202_raw: dict
    smart_206_normalized: dict
    smart_206_raw: dict
    smart_210_normalized: dict
    smart_210_raw: dict
    smart_218_normalized: dict
    smart_218_raw: dict
    smart_220_normalized: dict
    smart_220_raw: dict
    smart_222_normalized: dict
    smart_222_raw: dict
    smart_223_normalized: dict
    smart_223_raw: dict
    smart_224_normalized: dict
    smart_224_raw: dict
    smart_225_normalized: dict
    smart_225_raw: dict
    smart_226_normalized: dict
    smart_226_raw: dict
    smart_230_normalized: dict
    smart_230_raw: dict
    smart_231_normalized: dict
    smart_231_raw: dict
    smart_232_normalized: dict
    smart_232_raw: dict
    smart_233_normalized: dict
    smart_233_raw: dict
    smart_234_normalized: dict
    smart_234_raw: dict
    smart_235_normalized: dict
    smart_235_raw: dict
    smart_240_normalized: dict
    smart_240_raw: dict
    smart_241_normalized: dict
    smart_241_raw: dict
    smart_242_normalized: dict
    smart_242_raw: dict
    smart_244_normalized: dict
    smart_244_raw: dict
    smart_245_normalized: dict
    smart_245_raw: dict
    smart_246_normalized: dict
    smart_246_raw: dict
    smart_247_normalized: dict
    smart_247_raw: dict
    smart_248_normalized: dict
    smart_248_raw: dict
    smart_250_normalized: dict
    smart_250_raw: dict
    smart_251_normalized: dict
    smart_251_raw: dict
    smart_252_normalized: dict
    smart_252_raw: dict
    smart_254_normalized: dict
    smart_254_raw: dict
    smart_255_normalized: dict
    smart_255_raw: dict


@app.get('/')
def get_root():
    """Landing page with fastAPI decorator

    Returns:
        _type_: Welcome message
    """
    return {
        'message':
        'Welcome to the API for the prediction of hard drive health.'}


@app.post("/receive_dataframe")
async def receive_dataframe(dataframe_as_json: Item):
    """Receive dataframe through API, make predictions and return the results

    Args:
        dataframe_as_json (Item): RAW data

    Returns:
        _type_: predictions
    """
    df = pd.DataFrame(dataframe_as_json.dict())
    # Make predictions
    y_pred = pd.Series(run_predict(df)[:, 1])
    # Convert to json
    out = json.loads(y_pred.to_json(orient='columns'))
    return out


@app.post("/receive_preprocessed_dataframe")
async def receive_preprocessed_dataframe(dataframe_as_json: str):
    """Receive preprocessed dataframe through API, make predictions and return
    the results

    Args:
        dataframe_as_json (Item): Preprocessed data

    Returns:
        _type_: predictions
    """
    df = pd.read_json(dataframe_as_json)
    # Make predictions
    y_pred = pd.Series(run_predict(df)[:, 1])
    # Convert to json
    out = json.loads(y_pred.to_json(orient='columns'))
    return out


# run the app
if __name__ == '__main__':
    uvicorn.run(app)
