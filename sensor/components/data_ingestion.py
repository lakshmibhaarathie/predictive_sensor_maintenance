# standard modules
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# user-defined modules
from sensor.logger import logging
from sensor.utils.main_utils import Utils
from sensor.exceptions import SensorException
from sensor.data_access.sensor_data import SensorData
from sensor.entity.config_entity import DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.constant.database import COLLECTION_NAME

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig,  sensor_data:SensorData)->None:
        """
        Description: This class performs several operations and, \
                        performs initial data pre-processing for \
                        further pipelines.
        Params:
            - data_ingestion_config: Essential configurations for performing Data Ingestion.
            - sensor_data: MongoDB connection object to import data.
        """
        try:
            logging.info("Data Ingestion initiated.")
            self.data_ingestion_config = data_ingestion_config
            self.sensor_data = sensor_data
            self._schema_config = Utils.read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    
    def import_data_as_feature_store(self,)->pd.DataFrame:
        """
        Description: Import data from mongoDB database and gives a pandas Dataframe.

        Returns: pandas Dataframe.
        """
        try:
            logging.info("Importing data as a feature store.")
            df = self.sensor_data.import_data_from_mongodb(collection_name=COLLECTION_NAME)
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            feature_store_dir = os.path.dirname(feature_store_file_path)
            os.makedirs(name=feature_store_dir, exist_ok= True)
            df.to_csv(feature_store_file_path, index=False, header=True)
            logging.info("File got stored in feature_store as [{0}]".format(
                os.path.basename(feature_store_file_path)
            ))
            return df
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message="errorr in importing data")
        
    def drop_unnecessary_columns(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Description: Our schema has certain columns to be dropped. \
            This functions checks whether the drop columns are present in \
                the dataframe and remove the unecessary columns in the dataset.

        Params:
        -------
        df: pandas dataframe
        
        Returns: pandas dataframe.
        """
        try:
            schema_drop_columns = self._schema_config["drop_columns"]      # drop column names
            logging.info("Schema drop column names [{0}]".format(schema_drop_columns))
            drop_column_names = list()
            for column in schema_drop_columns:                             # check whether drop columns present in dataset
                if column in list(df.columns):
                    drop_column_names.append(column)
            if schema_drop_columns:
                df.drop(drop_column_names, axis=1, inplace=True)             # drop the unnecessary columns
                logging.info("Columns [{0}] are dropped from the dataset".format(drop_column_names))
                return df
            logging.info("No column to drop in the dataset.")
            return df
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
        
    def data_split(self, df:pd.DataFrame)->None:
        """
        Description: Perform train test split on the data and.\
                        store them as train file and test file seperately.
        
        Params:
        --------
        df: DataFrame
            pandas datframe to be split

        """
        try:
            train_data, test_data = train_test_split(
                df, test_size=self.data_ingestion_config.test_split_ratio
            )
            logging.info("Performed train test split on the given dataframe.")
            
            train_file_path = self.data_ingestion_config.train_file_path
            test_file_path = self.data_ingestion_config.test_file_path
        
            train_file_dir = os.path.dirname(train_file_path)
            os.makedirs(train_file_dir, exist_ok=True)
            
            train_data.to_csv(train_file_path, index=False, header=True)
            logging.info("Train file stored as [{0}]".format(
                os.path.basename(train_file_path)
            ))
            test_data.to_csv(test_file_path, index=False, header=True
            )
            logging.info("Train file stored as [{0}]".format(
                os.path.basename(test_file_path)
            ))
            logging.info("Train and test data split completed and stored as seperate files successfully.")
        
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)

    def initiate_data_ingestion(self,)->DataIngestionArtifact:
        try:
            df = self.import_data_as_feature_store()

            # drop the schema_drop_columns if present
            df = self.drop_unnecessary_columns(df=df, name="Train data")

            # train test split
            self.data_split(df=df)
            
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_path=self.data_ingestion_config.feature_store_file_path
                , train_file_path=self.data_ingestion_config.train_file_path
                , test_file_path=self.data_ingestion_config.test_file_path
            )
            
            logging.info("Data Ingestion completed.")
            return data_ingestion_artifact
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
        

    

