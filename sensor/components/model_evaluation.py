import numpy as np
import pandas as pd
from sensor.logger import logging
from sensor.utils.main_utils import Utils
from sensor.exceptions import SensorException
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.entity.config_entity import ModelEvaluationConfig
from sensor.ml.metric.classification_metric import ClassificationMetrics
from sensor.ml.model.estimator import (TargetValueMapping, ModelResolver)
from sensor.entity.artifact_entity import (DataIngestionArtifact, ModelTrainerArtifact, ModelEvaluationArtifact)

class ModelEvaluation:

    def __init__(self, model_evaluation_config:ModelEvaluationConfig 
                , data_ingestion_artifact:DataIngestionArtifact
                , model_trainer_artifact:ModelTrainerArtifact)->None:
        try:
            logging.info("ModelEvaluation initiated.")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            logging.info(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    

    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            logging.info("Extracting Train dataset for Model Evaluation.")
            train_df = Utils.read_csv(file_path=train_file_path)
            logging.info("Extracting Test dataset for Model Evaluation.")
            test_df = Utils.read_csv(file_path=test_file_path)

            df = pd.concat([train_df, test_df])

            logging.info("Split the data into input fetaure and target feature for prediction.")
            X = df.drop(TARGET_COLUMN, axis=1)
            y = df[TARGET_COLUMN]
            logging.info("Encoding Target variables to standard form.")
            y.replace(TargetValueMapping().to_dict(), inplace=True)

            trained_model_metric_artifact = self.model_trainer_artifact.test_metric_artifact
            trained_model_file_path = self.model_trainer_artifact.trained_model_path
            model_resolver = ModelResolver()

            if not model_resolver.is_model_exists():
                logging.info("There is no previously saved models.")
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True
                    , improved_accuracy=None
                    , latest_model_path=None
                    , trained_model_path=trained_model_file_path
                    , trained_model_metric_artifact=trained_model_metric_artifact
                    , latest_model_metric_artifact=None
                )
                return model_evaluation_artifact
            
            latest_model_path = model_resolver.get_latest_model_path()
            logging.info("Extracting lastes previous saved model")
            latest_model = Utils.load_object(file_path=latest_model_path)
            logging.info("Extracting current trained model.")
            train_model = Utils.load_object(file_path=trained_model_file_path)

            logging.info("Predicting feature store data with current trained model.")
            y_trained_model = train_model.predict(X)
            logging.info("Predicting feature store data with latest previously stored model.")
            y_latest_model = latest_model.predict(X)

            trained_model_metrics = ClassificationMetrics.get_classfication_metric(
                y_true=y, y_pred=y_trained_model
            )
            logging.info("Model scores for curretnt trained model [{0}]".format(
                trained_model_metrics.__dict__
            ))
            latest_model_metrics = ClassificationMetrics.get_classfication_metric(
                y_true=y, y_pred=y_latest_model
            )
            logging.info("Model scores for latest previously saved model [{0}]".format(
                latest_model_metrics.__dict__
            ))

            improved_accuracy = trained_model_metrics.f1_score-latest_model_metrics.f1_score
            expected_increase_accuracy = self.model_evaluation_config.change_threshold
            
            if expected_increase_accuracy < improved_accuracy:
                is_model_accepted=True
            else:
                is_model_accepted=False
            
            logging.info("The expected accuracy: [{0}], the improved accuracy: [{1}]".format(
                expected_increase_accuracy, improved_accuracy
            ))
            logging.info("is model accepted [{0}]".format(is_model_accepted))
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted
                , improved_accuracy=improved_accuracy
                , latest_model_path=latest_model_path
                , latest_model_metric_artifact=latest_model_metrics
                , trained_model_path=trained_model_file_path
                , trained_model_metric_artifact=trained_model_metrics
            )
            logging.info("Model Evaluation complete.")

            return model_evaluation_artifact
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)