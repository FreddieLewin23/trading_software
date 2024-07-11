import schedule
import time
import subprocess
import datetime
import os
import json
import pandas as pd
import qi_client
import datetime
import csv
import numpy as np
import concurrent.futures
import statistics
import warnings
import math
import threading
import Qi_wrapper
from backtest_json_redo_MVG import grab_data2, find_sd_from_model_data, subtract_days_from_date

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['QI_API_KEY'] = ''
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = ''
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))

def job():
    now = datetime.datetime.now()
    if 14 <= now.hour < 21 and now.weekday() < 5:
        print("Running job at:", now)
        subprocess.run(["python", "/Users/FreddieLewin/PycharmProjects/pythonProject19/alpaca_trading_api.py"])

if __name__ == '__main__':
    schedule.every(20).minutes.do(job) # 5 minutes since the count-down is from the finish of the code which takes 25
    while True:
        schedule.run_pending()
        time.sleep(1)

