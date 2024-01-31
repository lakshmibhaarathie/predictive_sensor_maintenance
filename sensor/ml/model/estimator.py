import os
import numpy as np
from sensor.logger import logging
from sensor.exceptions import SensorException
from sensor.constant.training_pipeline import (MODEL_FILE_NAME, SAVED_MODEL_DIR)

class TargetValueMapping:
    
    def __init__(self)->None:
        self.pos:int = 1
        self.neg:int = 0
    
    def to_dict(self)->dict:
        """
        Description: 
            This function maps the categorical target to numbers.
        Returns:
            A dict of encoded target values.
        """
        try:
            mapping_response = self.__dict__
            logging.info("Target values encoded.")
            return mapping_response
        except Exception as e:
            logging.error("Target encoding failed.")
            raise SensorException(error_message=e)

    def reverse_mapping(self)->dict:
        """
        Description:
            This function reverse the encoded target varibale into its original form.
        Returns:
            A dict of decoded target values.
        """
        try:
            mapping_response = self.to_dict()
            reverse_map = dict(zip(
                mapping_response.values(), mapping_response.keys()
            ))
            logging.info("Target values decoded.")
            return reverse_map
        except Exception as e:
            logging.error("Target decoding failed.")
            raise SensorException(error_message=e)
        
class SensorModel:
    """
    Description:
        This class perform data tranaformation and target prediction for \
        the given input feature dataset.

    Params:
        preprocessor: data transformation object
        model: model object
    """
    def __init__(self, preprocessor:object, model:object) -> None:
        try:
            logging.info("SensorModel Initiated.")
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            logging.error(str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
        
    def predict(self,X_test:np.array)->np.array:
        """
        Description: 
            This function makes prediction for the new input features.
        
        Returns: predicted target variables
        """
        try:
            X_transformed = self.preprocessor.transform(X_test)
            logging.info("Data transformation completed for prediction.")
            y_pred = self.model.predict(X_transformed)
            logging.info("Model prediction completed for prediction.")
            
            return y_pred
        except Exception as e:
            logging.error(msg=str(SensorException(error_message=e)))
            raise SensorException(error_message=e)
    
class ModelResolver:
    def __init__(self, model_dir=SAVED_MODEL_DIR)->None:
        self.model_dir = model_dir
    
    def get_latest_model_path(self,)->str:
        try:
            model_list = os.listdir(self.model_dir)
            model_timestamps = list(map(int,model_list))

            latest_model_dir = max(model_timestamps)
            latest_model_path = os.path.join(
                self.model_dir, f"{latest_model_dir}", MODEL_FILE_NAME
            )

            return latest_model_path
        except Exception as e:
            raise SensorException(error_message=e)
    
    def is_model_exists(self,)->bool:
        try:
            if not os.path.exists(self.model_dir):
                return False
            
            model_timestamps = os.listdir(self.model_dir)
            if len(model_timestamps)==0:
                return False
            
            latest_model_path = self.get_latest_model_path()
            if not os.path.exists(latest_model_path):
                return False
            
            return True
        except Exception as e:
            raise SensorException(error_message=e)