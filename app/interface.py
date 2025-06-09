import gradio
import joblib
import numpy as np
import shap
import pandas as pd
import os

# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
model_path = os.path.join(project_root, 'models', 'best_model.pkl')

# Load the trained model
model = joblib.load(model_path)

# Define the input features (adjust as per your model's expected input)
feature_names = [
    'pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'diabetespedigreefunction', 'age'
]

def predict_diabetes(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age):
    try:
        # Prepare input for prediction
        input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]])
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
    except Exception as e:
        return f"Error making prediction: {str(e)}\n\nPlease check your input values and try again."
    # SHAP explanation
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)
        
        # Ensure SHAP values are 1-dimensional
        if hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) > 2:
                shap_vals = shap_values.values[0, :, 0]
            else:
                shap_vals = shap_values.values[0]
        else:
            # Fallback if shap_values doesn't have .values attribute
            shap_vals = np.array(shap_values[0]) if isinstance(shap_values, list) else shap_values[0]
        
        # Create DataFrame with feature names and SHAP values
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_vals
        })
        shap_df = shap_df.sort_values('SHAP Value', key=abs, ascending=False)
        explanation = shap_df.to_string(index=False)
    except Exception as e:
        # Fallback if SHAP explanation fails
        explanation = f"SHAP explanation unavailable: {str(e)}"
    result = f"Prediction: {'Diabetic' if pred == 1 else 'Not Diabetic'}\nProbability: {proba:.2f}\n\nFeature Contribution (SHAP):\n{explanation}"
    return result

iface = gradio.Interface(
    fn=predict_diabetes,
    theme="soft",
    inputs=[
        gradio.Number(label='Pregnancies', minimum=0, step=1, value=0),
        gradio.Number(label='Glucose', minimum=0, info="mg/dL", value=120),
        gradio.Number(label='Blood Pressure', minimum=0, info="mm Hg", value=70),
        gradio.Number(label='Skin Thickness', minimum=0, info="mm", value=20),
        gradio.Number(label='Insulin', minimum=0, info="μU/mL", value=79),
        gradio.Number(label='BMI', minimum=0, info="kg/m²", value=25),
        gradio.Number(label='Diabetes Pedigree Function', minimum=0, info="A score of diabetes hereditary influence", value=0.5),
        gradio.Number(label='Age', minimum=0, step=1, value=30),
    ],
    outputs=gradio.Textbox(label='Prediction & Feature Contribution'),
    title='Diabetes Predictor',
    description='Enter patient data to predict diabetes and see feature contributions. All values should be non-negative.',
    article="""
    <div style="text-align: center; max-width: 650px; margin: 0 auto;">
        <h3>About this Predictor</h3>
        <p>This tool uses machine learning to predict the likelihood of diabetes based on several health metrics.</p>
        <p>The model was trained on the Pima Indians Diabetes Database and uses a Random Forest classifier.</p>
        <p>SHAP values show how each feature contributes to the prediction - positive values push toward a diabetes diagnosis, while negative values push away from it.</p>
        <h4>Typical Ranges for Input Values:</h4>
        <ul style="text-align: left;">
            <li><strong>Pregnancies:</strong> 0-17</li>
            <li><strong>Glucose:</strong> 70-200 mg/dL</li>
            <li><strong>Blood Pressure:</strong> 40-120 mm Hg</li>
            <li><strong>Skin Thickness:</strong> 10-50 mm</li>
            <li><strong>Insulin:</strong> 0-850 μU/mL</li>
            <li><strong>BMI:</strong> 18-50 kg/m²</li>
            <li><strong>Diabetes Pedigree Function:</strong> 0.08-2.5</li>
            <li><strong>Age:</strong> 21-81 years</li>
        </ul>
    </div>
    """,
    examples=[
        # Non-diabetic example
        [1, 85, 66, 29, 0, 26.6, 0.351, 31],
        # Diabetic example
        [8, 183, 64, 0, 0, 23.3, 0.672, 32],
        # Borderline example
        [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    ]
)

if __name__ == '__main__':
    iface.launch()
