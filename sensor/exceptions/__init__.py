import sys

class SensorException(Exception):
    """
    Description: Raise custom exception.
    params: error_message-> exception occurred
    Returns: user defined exception message.
    """
    def __init__(self,error_message:str):
        self.error_message = SensorException.error_message_detail(error=error_message)
    
    @staticmethod
    def error_message_detail(error, error_detail:sys=sys):
        """
        Description:Get exception details to provide user define exeption
        Returns:Error script name, Error line number and Exception message combined as a string->error_message."""
        _, _, exception_traceback = error_detail.exc_info()
        
        script_name = exception_traceback.tb_frame.f_code.co_filename
        script_line = exception_traceback.tb_lineno
        error_message = "Error occured in python script name[{0}] line no [{1}] message ```[{2}]```".format(
            script_name, script_line, str(error)
        )
        
        return error_message
    def __str__(self,):
        return self.error_message

      
      
      
