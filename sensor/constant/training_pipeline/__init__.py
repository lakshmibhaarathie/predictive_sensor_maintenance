import os

"""
Global constants:
"""
PIPLEINE_NAME:str = "sensor"
ARTIFACT_DIR:str = "artifacts"
SCHEMA_FILE_PATH:str = os.path.join("config","schema.yaml")
SCHEMA_DROP_COLS:str = "drop_columns"

MAIN_FILE_NAME:str = "sensor.csv"
TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"
TARGET_COLUMN:str = "class"


MODEL_FILE_NAME = "model.pkl"
SAVED_MODEL_DIR = os.path.join("saved_models")

"""
Data Ingestion constants:
"""
DATA_INGESTION_DIR:str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str ="feature_store"
DATA_INGESTION_DATASET_DIR:str = "dataset"
DATA_INGESTION_TEST_SPLIT_RATIO:float = 0.2

"""
Data Validation constants:
"""
DATA_VALIDATION_DIR:str = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "validated_data"
DATA_VALIDATION_VALID_DIR:str = "invalid_data"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRFIT_REPORT_FILE_NAME:str = "report.yaml"

"""
Data Transformation constants:
"""

DATA_TRANSFORMATAION_DIR_NAME:str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str = "transformed_dataset"
DATA_TRANSFORMATAION_TRANSFORMED_OBJ_DIR:str = "transformed_object"
DATA_PREPROCESSING_OBJECT_FILE_NAME:str = "preprocessing.pkl"

"""
Model Training constants:
"""

MODEL_TRAINER_DIR:str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME:str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE:float = 0.9
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD:float = 0.05

"""
Model Evaluation constants:
"""
MODEL_EVALUATION_DIR_NAME:str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD:float = 0.02
MODEL_EVALUATION_REPORT_NAME = "report.yaml"

"""
Model Pusher constants.
"""
MODEL_PUSHER_DIR_NAME:str = "model_pusher"