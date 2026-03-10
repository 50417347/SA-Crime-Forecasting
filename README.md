# South African Crime Analysis and Prediction Project

This project analyses South African crime statistics, identifies crime patterns, and predicts future crime cases using machine learning.

## Features
- Load Excel or CSV crime data
- Clean and reshape wide-format crime data
- Generate summary reports
- Train a forecasting model
- Predict future crime cases
- Visualise crime patterns in a Streamlit dashboard

## Files
- `load_data.py` – loads the source dataset
- `clean_data.py` – cleans and reshapes the dataset
- `train_model.py` – trains the machine learning model
- `predict.py` – predicts future crime cases
- `report_generator.py` – creates summary outputs
- `dashboard.py` – interactive dashboard
- `requirements.txt` – required Python packages

## Run order
```bash
python clean_data.py
python train_model.py
python report_generator.py
python predict.py
streamlit run dashboard.py