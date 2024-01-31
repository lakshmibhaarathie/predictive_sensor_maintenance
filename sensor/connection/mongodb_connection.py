import os
import pymongo
import certifi
from sensor.constant.database import DATABASE_NAME
from sensor.constant.env_variables import MONGO_DB_URL
from sensor.logger import logging
from sensor.exceptions import SensorException

ca =certifi.where()

class MongoDBClient:
    client=None
    def __init__(self, database_name=DATABASE_NAME)->None:
        try:
            if MongoDBClient.client is None:
                logging.info(msg = "Client Status: None, extracting system environment variable.")
                mongo_db_url = MONGO_DB_URL
                if "localhost" in mongo_db_url:
                    MongoDBClient.client = pymongo.MongoCLient(mongo_db_url)
                    logging.info("Conneted to the localhost.")
                else:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
                    logging.info("Connected to the cloud.")
            self.client = MongoDBClient.client
            self.database_name = self.client[database_name]
        except Exception as e:
            logging.error("MongoDB connection failed.")
            raise SensorException(error_message=e)
    
