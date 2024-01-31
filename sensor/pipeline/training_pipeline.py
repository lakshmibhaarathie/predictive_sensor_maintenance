# user-defined modules
from sensor.logger import logging
from sensor.exceptions import SensorException
from sensor.entity.config_entity import (TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig
                                        , ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig)
from sensor.entity.artifact_entity import (DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
                                        , ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact)
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.data_access.sensor_data import SensorData

class TrainingPipeline:
    is_pipeline_running=False

    def __init__(self) -> None:
        self.training_pipeline_config = TrainingPipelineConfig()
        self.sensor_data = SensorData()
    
    def start_data_ingestion(self,)->DataIngestionArtifact:
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config, sensor_data=self.sensor_data)
            data_ingestion_artifact =data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            logging.error(str(SensorException(e)))
            raise SensorException(error_message=e)
        
    def start_data_validation(
            self, data_ingestion_artifact: DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact
                , data_validation_config=data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()

            return data_validation_artifact
        except Exception as e:
            logging.error(str(SensorException(e)))
            raise SensorException(error_message=e)
    
    def start_data_transformation(
            self, data_ingestion_artifact:DataIngestionArtifact)->DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact
                , data_transformation_config=data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            return data_transformation_artifact
        except Exception as e:
            logging.error(str(SensorException(e)))
            raise SensorException(error_message=e)
    
    def start_model_trainer(
            self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(
                training_pipleine_config=self.training_pipeline_config
            )
            model_trainer =ModelTrainer(
                data_transformation_artifact=data_transformation_artifact
                ,model_trainer_config=model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact
        except Exception as e:
            logging.error(str(SensorException(e)))
            raise SensorException(error_message=e)
    
    def start_model_evaluation(
            self, data_ingestion_artifact:DataIngestionArtifact
            , model_trainer_artifact:ModelTrainerArtifact)->ModelEvaluationArtifact:
        try:
            model_evaluation_config = ModelEvaluationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_evaluation = ModelEvaluation(
                data_ingestion_artifact=data_ingestion_artifact
                , model_evaluation_config=model_evaluation_config
                , model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            return model_evaluation_artifact
        except Exception as e:
            logging.error(str(SensorException(e)))
            raise SensorException(error_message=e)
    
    def start_model_pusher(
            self, model_evaluation_artifact:ModelEvaluationArtifact)->ModelPusherArtifact:
        try:
            model_pusher_config = ModelPusherConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact
                , model_pusher_config=model_pusher_config
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()

            return model_pusher_artifact
        except Exception as e:
            logging.error(str(SensorException(e)))
            raise SensorException(error_message=e)
        
    
    def run_pipeline(self,):
        try:
            TrainingPipeline.is_pipeline_running=True

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact, 
            )
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact
                , model_trainer_artifact=model_trainer_artifact
            )

            if not model_evaluation_artifact.is_model_accepted:
                raise Exception("Trained model is not better than best model.")
            model_pusher_artifact  = self.start_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact
            )

            TrainingPipeline.is_pipeline_running=False

        except Exception as e:
            TrainingPipeline.is_pipeline_running=False
            logging.error(str(SensorException(e)))
            raise SensorException(error_message=e)
