import pandas as pd
import pymysql
from dotenv import load_dotenv
import os
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging



load_dotenv()

username = os.getenv("USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
database = os.getenv("DATABASE")

def get_mysql_data():

    try:
        logging.info("Attempting to connect to MySQL database...")
        mydb = pymysql.connect(host=host, user=username, passwd=password, db=database)
        logging.info("Successfully connected to MySQL.")

        df = pd.read_sql_query("SELECT * FROM students", mydb)
        # print(df.head(5))
        return df


    except Exception as e:
        raise CustomException(e)
        logging.info(f"Error connecting to MySQL database: {e}")







