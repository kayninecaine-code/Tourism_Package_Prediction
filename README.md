# Tourism Package Prediction - MLOps Pipeline

Predicts whether a "Visit with Us" customer will purchase the Wellness Tourism Package.

- `tourism_project/data/` - registered raw dataset (`tourism.csv`)
- `tourism_project/model_building/` - `data_register.py`, `prep.py`, `train.py` (XGBoost + GridSearchCV, tracked with MLflow)
- `tourism_project/deployment/` - Streamlit app, its requirements, and the model committed by the pipeline
- `.github/workflows/pipeline.yml` - register-dataset -> data-prep -> model-traning, runs on every push to main
