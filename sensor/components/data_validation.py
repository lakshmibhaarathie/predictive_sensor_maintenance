# standard modules
import os
import pandas as pd
from scipy.stats import ks_2samp
# user-defined modules
from sensor.logger import logging
from sensor.utils.main_utils import Utils
from sensor.exceptions import SensorException
from sensor.entity.config_entity import DataValidationConfig
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

class DataValidation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact
                , data_validation_config:DataValidationConfig)->None:
        """
        Description: 
        ------------------------
            This class performs validation of data \
            from the dataset and, prepares Data drift report.
        
        Params:
        -----------------------
            - data_ingestion_artifact: Output reference from Data Ingestion pipeline.
            - data_validation_config: Essential configurations for performing Data Validation.

        
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = Utils.read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            logging.error(str(e))
            raise SensorException(error_message=e)
    
    def validate_number_of_columns(self, df:pd.DataFrame, name:str)->bool:
        """
        Description: 
        ------------------------
            This functions checks if all the standard columns required are present in the dataset.

        Params:
        ------------------------
        - df: pd.Dataframe
        - name: dataset to be checked

        Returns:
        ------------------------
            - bool
        """
        try:
            number_of_columns = len(self._schema_config["columns"])
            if len(df.columns)==number_of_columns:
                logging.info("The required number of columns is present in given [{0}].".format(name))
                return True
            logging.error("The [{0}] fail to meet the required number of columns.".format(name))
            return False
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    
    def is_numerical_column_exists(self, df:pd.DataFrame, name:str)->bool:
        """
        Description:
        ----------------------
            This functions checks if all the standard numerical columns present in the dataset.

        Params:
        -----------------------
            - df: pandas dataframe

        Returns:
        -----------------------
            - bool
        """
        try:
            required_numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = df.columns
            numerical_columns_present = True
            missing_numerical_columns = list()
            for num_column in required_numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_columns_present = False
                    missing_numerical_columns.append(num_column)
            if numerical_columns_present:
                logging.info("The [{0}] has all the standard numerical columns.".format(name))
            else:
                logging.info("The [{0}] fails to meet the standard numerical column requirement.\n[{1}]".format(
                    name, missing_numerical_columns
                ))

            return numerical_columns_present
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    
    def check_data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame 
                        , base_df_name:str, curr_df_name:str, threshold=0.7)->bool:
        """
        Description: 
        -----------------------
            This function prepares data drift report for the given two dataframe.

        Params:
        -----------------------
        base_df: standard dataframe
        current_df: new dataframe
        threshold: drift threshold, default=0.7

        Returns: 
        -----------------------
            - bool
        """
        try:
            drift_status=False
            report = dict()
            logging.info("Checking data drift between [{0}] and [{1}]".format(base_df_name, curr_df_name))
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    drift_status=True
                
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue)
                    ,"drift_status":is_found
                    }})
            
            if drift_status:
                logging.info("There is Data drift between the [{0}] and [{1}]".format(
                    base_df_name, curr_df_name
                    ))
            
            # write drift report in the file path
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            drift_report_dir = os.path.dirname(drift_report_file_path)
            os.makedirs(drift_report_dir, exist_ok=True)

            # write drift report as yaml file
            Utils.write_yaml_file(file_path=drift_report_file_path,
                                content=report)
            logging.info("Drift report generated successfully.")
            
            return drift_status
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
        
    
    def initiate_data_validation(self,)->DataValidationArtifact:
        
        try:
            error_message = ""
            
            # get dataset path from data ingestion artifact
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # read dataframe from path
            train_df = Utils.read_data(file_path=train_file_path)
            test_df = Utils.read_data(file_path=test_file_path)

            # validate number of columns
            status = self.validate_number_of_columns(df=train_df, name="Train data")
            if not status:
                error_message=f"{error_message}Train dataframe is missing the standard columns.\n"
            status = self.validate_number_of_columns(df=test_df, name="Test data")
            if not status:
                error_message=f"{error_message}Test dataframe is missing the standard columns.\n"

            # validate numerical columns
            status = self.is_numerical_column_exists(df=train_df,name="Train data")
            if not status:
                error_message = f"{error_message}Train datframe is missing the standard numerical columns.\n"
            status = self.is_numerical_column_exists(df=test_df, name="Test data")
            if not status:
                error_message = f"{error_message}Test dataframe is missing the standard numerical columns.\n"


            if len(error_message)>0:
                logging.info(msg=error_message)
                raise Exception("Validation Failed.")
            logging.info("Data has passed all the standard validation tests.")

            # generate drift report
            drift_status = self.check_data_drift(base_df=train_df, current_df=test_df,
                                                base_df_name="Train data", curr_df_name="Test data")
            drift_file_path = self.data_validation_config.drift_report_file_path

            data_validation_artifact = DataValidationArtifact(drift_status=drift_status
                                                            , drift_report_path=drift_file_path)
            
            logging.info("Data Validation complete.")
            return data_validation_artifact
        
        except Exception as e:
            raise SensorException(error_message=e)