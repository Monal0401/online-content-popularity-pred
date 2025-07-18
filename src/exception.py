import sys
import logging

def error_message_detail(error, error_detail: sys):
    """
    Returns a formatted error message string including filename, line number and error text.
    If traceback is not available, returns just the error string.
    """
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
        error_message = f"Error occurred in python script name [{file_name}] line number [{line_no}] error message [{str(error)}]"
    else:
        error_message = f"Error message: {str(error)} (No traceback available)"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message



    


#if __name__=="__main__":
#    try:
#        a=1/0
#    except Exception as e:
#        logging.info("Divide by zero")
#       raise CustomException(e,sys)