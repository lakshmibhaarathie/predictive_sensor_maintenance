from typing import List
from setuptools import setup, find_packages

# package details
PROJECT_NAME = "aps-sensor"
VERSION = "0.0.0"
AUTHOR = "lbhaarathi"
AUTHOR_EMAIL = "lakshmibhaarathiofcl@gmail.com"
DESCRIPTION = "This is a ML project to find Fault in APS sensors."

REQUIREMENT_FILE = "requirements.txt"

HYPHEN_E_DOT = "-e ."

def get_requirements()->List[str]:
    """
    Description: This function helps in extracting the required 
    list of libraries from requirements.txt file.

    Returns: A list of libraries
    """

    with open(REQUIREMENT_FILE) as requiremnt_file:
        requirement_list = requiremnt_file.readlines()
        requirement_list = [requirement_name.replace("\n","")
                            for requirement_name in requirement_list]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
    return requirement_list
        

setup(
    name=PROJECT_NAME
   ,version=VERSION
   ,author=AUTHOR
   ,author_email=AUTHOR_EMAIL
   ,packages=find_packages() 
   ,install_requires=get_requirements()
)