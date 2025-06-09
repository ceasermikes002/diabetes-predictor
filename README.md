# Diabetes Predictor

## Project Overview

The Diabetes Predictor is a machine learning application that predicts the likelihood of diabetes based on several health metrics. This project implements a Random Forest classifier trained on the Pima Indians Diabetes Database to provide accurate predictions and explanations of the factors contributing to those predictions.

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Information](#model-information)
6. [SHAP Explanations](#shap-explanations)
7. [Input Parameters](#input-parameters)
8. [Technical Implementation](#technical-implementation)
9. [Troubleshooting](#troubleshooting)
10. [Future Improvements](#future-improvements)

## Features

- **Diabetes Prediction**: Predicts whether a patient is likely to have diabetes based on health metrics
- **Probability Score**: Provides a probability score indicating the confidence of the prediction
- **Feature Importance**: Uses SHAP (SHapley Additive exPlanations) to explain how each input parameter contributes to the prediction
- **User-Friendly Interface**: Intuitive web interface built with Gradio
- **Input Validation**: Ensures all inputs are within valid ranges
- **Example Cases**: Includes pre-configured examples for quick testing

## Project Structure

```
diabetes-predictor/
├── app/
│   ├── __init__.py
│   └── interface.py       # Gradio interface and prediction logic
├── data/
│   └── diabetes.csv       # Original dataset
├── models/
│   └── best_model.pkl     # Trained Random Forest model
├── notebooks/
│   └── model_training.ipynb  # Jupyter notebook for model development
├── main.py                # Application entry point
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/diabetes-predictor.git
   cd diabetes-predictor
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Using conda
   conda create -n diabetes-env python=3.8
   conda activate diabetes-env
   
   # Or using venv
   python -m venv diabetes-env
   # On Windows
   diabetes-env\Scripts\activate
   # On macOS/Linux
   source diabetes-env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the application**:
   ```bash
   python main.py
   ```

2. **Access the web interface**:
   - The application will start a local web server
   - Open your browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860)

### Making Predictions

1. Enter the patient's health metrics in the provided input fields:
   - Pregnancies
   - Glucose level (mg/dL)
   - Blood Pressure (mm Hg)
   - Skin Thickness (mm)
   - Insulin level (μU/mL)
   - BMI (kg/m²)
   - Diabetes Pedigree Function
   - Age (years)

2. Click the "Submit" button to generate a prediction

3. Review the results:
   - Prediction (Diabetic or Not Diabetic)
   - Probability score
   - Feature contributions (SHAP values)

4. Try the provided examples by clicking on them at the bottom of the interface

## Model Information

### Dataset

The model was trained on the Pima Indians Diabetes Database, which includes the following features:
- Number of pregnancies
- Plasma glucose concentration
- Diastolic blood pressure (mm Hg)
- Triceps skin fold thickness (mm)
- 2-Hour serum insulin (μU/ml)
- Body mass index (kg/m²)
- Diabetes pedigree function
- Age (years)

The target variable is a binary indicator of whether the patient developed diabetes within 5 years.

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Performance Metrics**:
  - Accuracy: ~78%
  - Precision: ~76%
  - Recall: ~72%
  - F1 Score: ~74%
  - AUC-ROC: ~84%

### Training Process

The model was trained using the following steps:
1. Data preprocessing (handling missing values, scaling)
2. Feature selection
3. Hyperparameter tuning using grid search with cross-validation
4. Model evaluation on a held-out test set

## SHAP Explanations

The application uses SHAP (SHapley Additive exPlanations) to provide interpretable explanations for each prediction:

- **Positive SHAP Values**: Features that push the prediction toward diabetes
- **Negative SHAP Values**: Features that push the prediction away from diabetes
- **Magnitude**: The absolute value indicates the strength of the feature's influence

The SHAP values are sorted by their absolute magnitude, showing the most influential features at the top.

## Input Parameters

### Valid Ranges and Descriptions

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| Pregnancies | Number of times pregnant | 0-17 | count |
| Glucose | Plasma glucose concentration after 2 hours in an oral glucose tolerance test | 70-200 | mg/dL |
| Blood Pressure | Diastolic blood pressure | 40-120 | mm Hg |
| Skin Thickness | Triceps skin fold thickness | 10-50 | mm |
| Insulin | 2-Hour serum insulin | 0-850 | μU/mL |
| BMI | Body mass index | 18-50 | kg/m² |
| Diabetes Pedigree Function | A function that scores likelihood of diabetes based on family history | 0.08-2.5 | score |
| Age | Age of the patient | 21-81 | years |

## Technical Implementation

### Frontend

The user interface is built using Gradio, a Python library for creating customizable ML interfaces. Key features:
- Input validation to ensure values are within acceptable ranges
- Informative tooltips for each input field
- Responsive design that works on desktop and mobile devices
- Pre-configured examples for quick testing
- Detailed explanation of predictions

### Backend

- **Model Loading**: The trained model is loaded from a pickle file
- **Prediction Pipeline**: Processes input data, generates predictions and probabilities
- **SHAP Integration**: Calculates feature contributions for each prediction
- **Error Handling**: Robust error handling for both prediction and explanation generation

### Dependencies

- **scikit-learn**: For the Random Forest model
- **SHAP**: For generating model explanations
- **Gradio**: For the web interface
- **NumPy & Pandas**: For data manipulation
- **joblib**: For model serialization/deserialization

## Troubleshooting

### Common Issues

1. **SSL Certificate Errors**:
   - The application automatically handles SSL certificate issues by unsetting problematic environment variables

2. **Missing Dependencies**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

3. **Model Loading Errors**:
   - Verify the model file exists at `models/best_model.pkl`

4. **SHAP Explanation Errors**:
   - The application includes fallback mechanisms if SHAP explanations fail

### Debugging

If you encounter issues:
1. Check the terminal output for error messages
2. Verify input values are within expected ranges
3. Ensure all dependencies are correctly installed
4. Try running with the `--debug` flag: `python main.py --debug`

## Future Improvements

Potential enhancements for future versions:

1. **Model Improvements**:
   - Implement ensemble methods combining multiple algorithms
   - Add support for more advanced feature engineering
   - Incorporate time-series data for longitudinal analysis

2. **Interface Enhancements**:
   - Add visualization of SHAP values using waterfall plots
   - Implement user authentication for medical professionals
   - Add support for saving and comparing multiple predictions

3. **Deployment Options**:
   - Containerize the application using Docker
   - Create cloud deployment options (AWS, Azure, GCP)
   - Develop a REST API for integration with other systems

4. **Additional Features**:
   - Implement batch prediction for multiple patients
   - Add support for uploading CSV files with patient data
   - Integrate with electronic health record systems

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Pima Indians Diabetes Database for providing the training data
- The SHAP library for model interpretability
- The Gradio team for the user interface framework