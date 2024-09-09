import schedule
import time
import subprocess
import datetime
import os
import json
import pandas as pd
import qi_client
from datetime import datetime, timedelta
import csv
import numpy as np
import concurrent.futures
import statistics
import warnings
import math
import threading
import Qi_wrapper

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['QI_API_KEY'] = ''
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = ''
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))


def job():
    now = datetime.now()
    if 14 <= now.hour < 21 and now.weekday() < 5:
        print("Running job at:", now)
        subprocess.run(["python", "/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/execute_trades_latest.py"])


if __name__ == '__main__':
    job()
    schedule.every(12).minutes.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)

