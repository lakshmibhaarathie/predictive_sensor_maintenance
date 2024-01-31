import os
from xgboost import XGBClassifier
from sensor.logger import logging
from sensor.utils.main_utils import Utils
from sensor.ml.model.estimator import SensorModel
from sensor.exceptions import SensorException
from sensor.entity.config_entity import ModelTrainerConfig
from sensor.ml.metric.classification_metric import ClassificationMetrics
from sensor.entity.artifact_entity import (DataTransformationArtifact, ModelTrainerArtifact)


class ModelTrainer:
    
    def __init__(self, model_trainer_config:ModelTrainerConfig
                ,data_transformation_artifact:DataTransformationArtifact)->None:
        logging.info("ModelTrainer initiated.")
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)

    def train_model(self, X_train, y_train)->object:
        """
        Description: 
            This function train the dataset for model prediction.
        Params:
        ----------
        X_train: np.array
            input features dataset
        y_train: np.array
            target features dataset
        
        Returns: object
            trained model object
        """
        try:
            # initializing ml model
            xgb_clf = XGBClassifier()
            logging.info(msg="Model getting trained with the dataset.")
            
            # fitting with train data
            xgb_clf.fit(X_train, y_train)
            logging.info("Model training complete and ready for prediction.")

            return xgb_clf
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    
    def initiate_model_trainer(self,)->ModelTrainerArtifact:
        """
        Description:
            This function initiates model training and perform all the model training operations.
        
        Returns:ModelTrainerArtifact
        """
        try:
            # get transformed train file and test file path
            logging.info("Accessing DataTransfromationArtifact for transformed train and test file paths.")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            # extract data from train file and trest file path
            train_arr = Utils.load_numpy_array(train_file_path)
            test_arr = Utils.load_numpy_array(test_file_path)
            
            # data split into Input feature and Target feature
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1]
            )
            logging.info("Data split into input feature and target features")

            # ml model creation
            clf_model = self.train_model(X_train=X_train, y_train=y_train)
            
            # prediction for X_train
            logging.info("Predicting y_train with X_train.")
            y_train_pred = clf_model.predict(X_train)
            
            # classfication metrics for predicted Train data
            train_metrics = ClassificationMetrics.get_classfication_metric(
                y_true=y_train, y_pred=y_train_pred
            )
            logging.info("Classification metrics for Train data [{0}]".format(
                train_metrics.__dict__
            ))
            if train_metrics.f1_score<=self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not meeting the standard accuracy.")
            logging.info("Trained model has met with the standard accuracy.")
            
            # predicting test data
            logging.info("Predicting y_test with X_test.")
            y_test_pred = clf_model.predict(X_test)
            # get classification metric for test data
            test_metrics = ClassificationMetrics.get_classfication_metric(
                y_true=y_test, y_pred=y_test_pred
            )
            logging.info("Classification metrics for Test data [{0}]".format(
                test_metrics.__dict__
            ))
        
            # overfitting underfitting threshold
            threshold_diff = abs(train_metrics.f1_score-test_metrics.f1_score)
            if threshold_diff>self.model_trainer_config.over_fitting_under_fitting_threshold:
                raise Exception("Model has failed to meet the over-fiting under-fitting threshold.")
            logging.info("Model has passed over-fitting under-fitting threshold.")
            
            # get transformer object
            logging.info("Extracting transformer object from DataTransformationArtifact.")
            preprocessor = Utils.load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            # create model directory
            model_dir = os.path.dirname(p=self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)

            # save transformed object and fmodel object for future prediction
            sensor_model = SensorModel(preprocessor=preprocessor, model=clf_model)
            Utils.save_object(
                file_path=self.model_trainer_config.trained_model_file_path
                , obj=sensor_model
            )
            logging.info("Saved Transformation object and Model object for future prediction.")

            # model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_file_path
                , train_metric_artifact=train_metrics
                , test_metric_artifact=test_metrics
            )
            logging.info("Model Training complete.")

            return model_trainer_artifact
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message= e)