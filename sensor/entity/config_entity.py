import os
from datetime import datetime
from sensor.logger import logging
from sensor.exceptions import SensorException
from sensor.constant import training_pipeline, database


class TrainingPipelineConfig:

    def __init__(self)->None:
        try:
            self.timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
            self.pipeline_name:str = training_pipeline.PIPLEINE_NAME
            self.artifact_dir:str = os.path.join(
                training_pipeline.ARTIFACT_DIR, self.timestamp
            )
        except Exception as e:
            logging.ERROR(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig)->None:
        try:
            logging.info("Accessing DataIngestionConfig.")
            self.data_ingestion_dir:str = os.path.join(
                training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR
            )
            self.feature_store_file_path:str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR
                , training_pipeline.MAIN_FILE_NAME
            )
            self.train_file_path = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_DATASET_DIR
                , training_pipeline.TRAIN_FILE_NAME
            )
            self.test_file_path = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_DATASET_DIR
                , training_pipeline.TEST_FILE_NAME
            )
            self.test_split_ratio:float = training_pipeline.DATA_INGESTION_TEST_SPLIT_RATIO
            self.collection_name = database.COLLECTION_NAME

        except Exception as e:
            logging.ERROR(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    def to_dict(self):
        return self.__dict__


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            logging.info("Accessing DataValidationConfig.")
            self.data_validation_dir:str = os.path.join(
                training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR
            )
            self.drift_report_dir:str = os.path.join(
                self.data_validation_dir, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR
            )
            self.drift_report_file_path:str = os.path.join(
                self.drift_report_dir, training_pipeline.DATA_VALIDATION_DRFIT_REPORT_FILE_NAME
            )
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)


class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig)->None:
        try:
            logging.info(msg="Accessing DataTransformationConfig.")
            self.data_transformation_dir:str = os.path.join(
                training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATAION_DIR_NAME
            )
            self.transformed_train_file_path:str = os.path.join(
                self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
                , training_pipeline.TRAIN_FILE_NAME.replace("csv","npy")
            )
            self.transformed_test_file_path:str = os.path.join(
                self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
                , training_pipeline.TEST_FILE_NAME.replace("csv","npy")
            )
            self.transformed_object_file_path:str = os.path.join(
                self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATAION_TRANSFORMED_OBJ_DIR
                , training_pipeline.DATA_PREPROCESSING_OBJECT_FILE_NAME
            )
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
class ModelTrainerConfig:
    
    def __init__(self, training_pipleine_config:TrainingPipelineConfig):
        try:
            logging.info("Accessing ModelTrainerConfig")
            self.model_trainer_dir = os.path.join(
                training_pipleine_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR
            )
            self.trained_model_file_path = os.path.join(
                self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR
                , training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME
            )
            self.expected_accuracy:float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
            self.over_fitting_under_fitting_threshold:float = \
                training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    

class ModelEvaluationConfig:
    
    def __init__(self, training_pipeline_config:TrainingPipelineConfig) -> None:
        try:
            logging.info("Accessing ModelEvaluationConfig.")
            self.model_evaluation_dir = os.path.join(
                training_pipeline_config.artifact_dir, training_pipeline.MODEL_EVALUATION_DIR_NAME
            )
            self.model_evaluation_report_path = os.path.join(
                self.model_evaluation_dir, training_pipeline.MODEL_EVALUATION_REPORT_NAME
            )
            self.change_threshold = training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)


class ModelPusherConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig)->None:
        try:
            logging.info("Accessing ModelEvaluationConfig.")
            self.model_pusher_dir:str = os.path.join(
                training_pipeline_config.artifact_dir, training_pipeline.MODEL_EVALUATION_DIR_NAME
            ) 
            self.model_file_path = os.path.join(
                self.model_pusher_dir, training_pipeline.MODEL_FILE_NAME
            )
            model_timestamp = round(datetime.now().timestamp())
            self.saved_model_path = os.path.join(
                training_pipeline.SAVED_MODEL_DIR, f"{model_timestamp}", training_pipeline.MODEL_FILE_NAME
            )
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)