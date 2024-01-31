import os
import dill
import yaml
import numpy as np
import pandas as pd
from sensor.exceptions import SensorException
from sensor.logger import logging

class Utils:
    @staticmethod
    def read_csv(file_path:str)->pd.DataFrame:
        """
        Description:
            This function reads the dataset from csv file and returns as a pandas Dataframe.

        Params:
        ---------
        file_path: csv containing file_path

        Returns:
            pandas Dataframe
        """
        try:
            df =pd.read_csv(filepath_or_buffer=file_path)
            logging.info("Reading [{0}] as pandas dataframe.".format(
                os.path.basename(p=file_path)
            ))
            
            return df
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    @staticmethod
    def read_yaml_file(file_path:str)->dict:
        """
        Description:
            This function read the yaml file safely.
        Params:
        -------
        file_path: str
            yaml file path
        
        Returns: 
            A dict of contents in yaml file.
        """
        try:
            logging.info("Reading yaml_file [{0}].".format(
                os.path.basename(p=file_path)
            ))
            with open(file=file_path, mode="rb") as yaml_file:
                return yaml.safe_load(stream=yaml_file)
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    
    @staticmethod
    def write_yaml_file(
        file_path:str, content:object, replace:bool=False
    )->None:
        """
        Description: 
            This functions writes yaml file to the given path.

        Params:
        ---------
        file_path: str
            path to store the yaml file
        content: object
            data to be stored as yaml file
        replace: bool
            replace the previously existing file
        """
        try:
            logging.info("Writing yaml file [{0}]".format(os.path.basename(file_path)))
            if replace:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info("The path already exists, replacing with new yaml file.")
            yaml_file_dir = os.path.dirname(file_path)
            os.makedirs(yaml_file_dir, exist_ok=True)

            with open(file=file_path, mode="w") as file:
                yaml.dump(data=content, stream=file)
            logging.info("Completed writing yaml file [{0}].".format(os.path.basename(file_path)))  
            
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)        
    
    @staticmethod
    def save_numpy_array(file_path:str, array:np.array)->None:
        """
        Description: 
            This function saves numpy array.
        
        Params:
        -------
        file_path:str
            path to save the numpy array
        array: np.array
            array to be saved
        """
        try:
            numpy_array_dir = os.path.dirname(file_path)
            os.makedirs(numpy_array_dir, exist_ok=True)

            with open(file=file_path, mode="wb") as file_obj:
                np.save(file=file_obj, arr=array)
            logging.info("Array saved as [{0}] in the given path successfully.".format(
                os.path.basename(p=file_path)
            ))
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    
    @staticmethod
    def load_numpy_array(file_path:str)->np.array:
        """
        Description: 
            This function extracts the numpy array from the given path.

        Params:
        -------
        file_path: str
            file path to extract numpy array.
        
        Returns: 
            numpy array
        """
        try:
            logging.info("Extracting numpy array from the file [{0}].".format(
                os.path.basename(p=file_path)
            ))
            with open(file=file_path, mode="rb") as file_obj:
                return np.load(file=file_obj)

        except Exception as e:
            logging.error(str(SensorException(e)))
            raise SensorException(e)
        
    @staticmethod
    def save_object(file_path:str, obj:object)->None:
        """
        Description: 
            This function saves object to the given filepath.

        Params:
        ----------
        file_path:str
            file path to be saved
        obj: object
            object to be saved
        """
        try:
            object_dir = os.path.dirname(file_path)
            os.makedirs(object_dir, exist_ok=True)
            with open(file=file_path, mode="wb") as obj_path:
                dill.dump(file=obj_path, obj=obj)
                logging.info("Object saved successfully as [{0}].".format(
                    os.path.basename(file_path)
                ))
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    
    @staticmethod
    def load_object(file_path:str)->object:
        """
        Description: 
            This function extract the object from the given file path
        
        Params:
        --------
        file_path:str
            object file path
        
        Returns: 
            object
        """
        try:
            if not os.path.exists(file_path):
                #logging.ERROR("File path [{0}] doesnot exists.".format(file_path))
                raise Exception("File path [{0}] doesnot exists.".format(file_path))
            with open(file=file_path, mode="rb") as obj_path:
                logging.info("Extracting file object from [{0}]".format(
                    os.path.basename(file_path)
                ))
                return dill.load(file=obj_path)
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)



