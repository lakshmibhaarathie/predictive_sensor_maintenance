import os, sys
import json
import numpy as np
import pandas as pd
from typing import Optional

from sensor.connection.mongodb_connection import MongoDBClient
from sensor.constant.database import DATABASE_NAME
from sensor.logger import logging
from sensor.exceptions import SensorException

class SensorData:
    """
    Description: This class helps in importing exporting data from mongoDB database.
    """
    def __init__(self,):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
            logging.info("Connected to database [{0}]".format(DATABASE_NAME))
        except Exception as e:
            logging.ERROR("Failed to connect [{0}] database.".format(DATABASE_NAME))
            raise SensorException(error_message="error in SensorData")
        
    def export_to_mongodb(self, file_path:str,
                        collection_name:str, database_name:Optional[str]=None):
        """
        Description: This function export any csv file into the mongoDB database.
        
        Params:
        -----------
        file_path: str
            data file path
        collection_name: str
            mongoDB collection name
        database_name: str
            mongoDB database name

        Returns: Number of records dumped in mongoDB
        """
        try:
            logging.info("Exporting data to mongoDB.")
            logging.info("Reading the file [{0}] as pandas dataframe.".format(
                os.path.basename(p=file_path)
                ))
            df = pd.read_csv(filepath_or_buffer=file_path)
            df.reset_index(drop=True, inplace=True)
            records = list(json.loads(df.T.to_json()).values())
            if database_name is None:
                collections = self.mongo_client.database_name[collection_name]
                logging.info("Connected to default database collection [{0}]".format(
                    collection_name
                ))
            else:
                collections = self.mongo_client[database_name][collection_name]
                logging.info("Connected to new database [{0}] collection [{1}]".format(
                    database_name, collection_name
                ))
            collections.insert_many(records)
            logging.info("Inserted [{0}] records into mongoDB.".format(len(records)))
            return len(records)
        except Exception as e:
            raise SensorException(error_message=e)
    
    def import_data_from_mongodb(self, collection_name:str
                                , database_name:Optional[str]=None)->pd.DataFrame:
        """
        Description: This function is going to import data collections from mongoDB as a pandas Dataframe.
        
        Params:
        --------
        database_name: str
            mongoDB database name to connect
        collection_name: str
            database collection name to import
        
        Returns: Pandas Dataframe
        """
        try:
            logging.info("Importing data from mongoDB.")
            if database_name is None:
                collections = self.mongo_client.database_name[collection_name]
                logging.info("Connected to default database collection [{0}].".format(
                    collection_name
                ))
            else:
                collections = self.mongo_client[database_name][collection_name]
                logging.info("Connected to new database [{0}] colection [{1}].".format(
                    database_name, collection_name
                ))
            df = pd.DataFrame(list(collections.find()))
            logging.info("MongoDB collections --> Pandas DataFrame.")
            if "_id" in df.columns.to_list():
                df.drop(columns=["_id"], axis=1, inplace=True)
            df.replace({"na":np.nan},inplace=True)
            logging.info("Imported [{0}] records from mongoDB.".format(df.shape[0]))

            return df
        except Exception as e:
            logging.ERROR(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
