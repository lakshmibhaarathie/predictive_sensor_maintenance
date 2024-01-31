import os
import shutil
from sensor.logger import logging
from sensor.utils.main_utils import Utils
from sensor.exceptions import SensorException
from sensor.ml.metric.classification_metric import ClassificationMetrics
from sensor.entity.config_entity import ModelPusherConfig
from sensor.entity.artifact_entity import (ModelPusherArtifact, ModelEvaluationArtifact)

class ModelPusher:
    def __init__(self, model_evaluation_artifact:ModelEvaluationArtifact
                , model_pusher_config:ModelPusherConfig)->None:
        try:
            logging.info("Model pusher initiated.")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    
    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:
            logging.info("Extracting trained model path.")
            trained_model_path = self.model_evaluation_artifact.trained_model_path

            model_file_path = self.model_pusher_config.model_file_path
            model_file_dir = os.path.dirname(model_file_path)
            os.makedirs(model_file_dir, exist_ok=True)
            logging.info("Saving the trained model in model pusher dir.")
            shutil.copy(src=trained_model_path, dst=model_file_path)

            saved_model_path = self.model_pusher_config.saved_model_path
            saved_model_dir = os.path.dirname(saved_model_path)
            os.makedirs(saved_model_dir, exist_ok=True)
            logging.info("Saving the model in saved models dir.")
            shutil.copy(src=trained_model_path, dst=saved_model_path)

            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=saved_model_path, model_file_path=model_file_path
            )
            logging.info("Model Pusher complete.")
            return model_pusher_artifact
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)

