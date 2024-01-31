# standard modules
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
# user-defined modules
from sensor.logger import logging
from sensor.utils.main_utils import Utils
from sensor.exceptions import SensorException
from sensor.ml.model.estimator import TargetValueMapping
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.entity.config_entity import DataTransformationConfig
from sensor.entity.artifact_entity import (DataIngestionArtifact, DataTransformationArtifact)

class DataTransformation:

    def __init__(self, data_ingestion_artifact:DataIngestionArtifact
                , data_transformation_config: DataTransformationConfig)->None:
        """
        Desciption:
            This class helps in data transformation operations, \
                like scaling, imputing, handling data imbalance.
        Params:
            data_ingestion_artifact: Output reference of Data Ingestion pipeline.
            data_transformation_artifact: Necessary configurations for transforming the data.
        """
        try:
            logging.info(msg="Data Transformation initiated.")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise SensorException(e)
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        """
        Description:
            This function creates a preprocessing pipeline that could perform\
            robust scaling and simple imputing.
        
        Returns:
            pre-processing Pipeline
        """
        try:
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant" 
                                        , fill_value=0)
            preprocessor = Pipeline(
                steps=[
                ("Imputer",simple_imputer)
                , ("RobustScaler", robust_scaler)
                ]
            )
            logging.info(msg ="Data Pre-processing pipeline get initialized.")
            return preprocessor
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)

    def initiate_data_transformation(self,)->DataTransformationArtifact:
        """
        Description:
            This function initiate Data Transformation.
        
        Returns:
            DataTransformationArtifact.
        """
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            logging.info(msg="Reading train data for Data Transformation.")
            train_df = Utils.read_csv(file_path=train_file_path)
            logging.info(msg="Reading test data for Data Transformation.")
            test_df = Utils.read_csv(file_path=test_file_path)

            logging.info(msg="Seperating input feature from target feature.")
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info(msg="Encoding target class of train and test datasets.")
            target_encoder = TargetValueMapping().to_dict()
            target_feature_train_df = target_feature_train_df.replace(target_encoder)
            target_feature_test_df = target_feature_test_df.replace(target_encoder)

            logging.info(msg="Performing simple imputation and robust scaling on train and test data.")
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_train_input = preprocessor_object.transform(input_feature_train_df)
            transformed_test_input = preprocessor_object.transform(input_feature_test_df)

            logging.info(msg="Performing minority sampling to balance the datasets.")
            smt = SMOTETomek(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_train_input, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_test_input, target_feature_test_df
            )

            train_arr = np.c_[input_feature_train_final, target_feature_train_final]
            test_arr = np.c_[input_feature_test_final, target_feature_test_final]

            Utils.save_numpy_array(
                file_path=self.data_transformation_config.transformed_train_file_path
                , array=train_arr
            )
            logging.info(msg="Transformed train data saved succesfully.")
            Utils.save_numpy_array(
                file_path=self.data_transformation_config.transformed_test_file_path
                , array=test_arr
            )
            logging.info(msg="Transformed test data saved succesfully.")
            Utils.save_object(
                file_path=self.data_transformation_config.transformed_object_file_path
                , obj=preprocessor_object
            )
            logging.info(msg="Pre-processing object saved succesfully.")

            transformed_object_file_path = self.data_transformation_config.transformed_object_file_path
            transformed_train_file_path = self.data_transformation_config.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_config.transformed_test_file_path

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=transformed_object_file_path
                , transformed_train_file_path=transformed_train_file_path
                , transformed_test_file_path=transformed_test_file_path
            )
            logging.info(msg="Data Transformation complete.")
            return data_transformation_artifact
        except Exception as e:
            logging.error(msg="Data Transformation failed.")
            raise SensorException(error_message=e)

