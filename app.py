import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1) Load the Model (change filename to yours)
with open("insurance_gb_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# 2) Prediction Function
def predict_charges(age, sex, bmi, children, smoker, region):
    
    input_df = pd.DataFrame([[
        age, sex, bmi, children, smoker, region
    ]], columns=[
        "age", "sex", "bmi", "children", "smoker", "region"
    ])
    
    pred = model.predict(input_df)[0]
    
    # Charges should not be negative
    pred = max(0, pred)
    
    return f"Predicted Insurance Charges: ${pred:,.2f}"

# 3) Gradio Inputs
inputs = [
    gr.Number(label="Age", value=25),
    gr.Radio(["male", "female"], label="Sex"),
    gr.Number(label="BMI", value=24.0),
    gr.Number(label="Children", value=0),
    gr.Radio(["yes", "no"], label="Smoker"),
    gr.Dropdown(["northeast", "northwest", "southeast", "southwest"], label="Region")
]

app = gr.Interface(
    fn=predict_charges,
    inputs=inputs,
    outputs="text",
    title="Medical Insurance Charges Predictor",
    description="Enter your details to predict estimated insurance charges."
)

app.launch(share=True)
