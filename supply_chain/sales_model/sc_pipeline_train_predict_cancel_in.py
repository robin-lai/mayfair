from datetime import timedelta, date, datetime
import pandas as pd
import numpy as np
import math
import torch
from torch import nn
import boto3
import json
import random
import time
from pyarrow import parquet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


s3_cli = boto3.client('s3')
time_delta = 0
BUCKET = "warehouse-algo"
train_and_predict_data_path = "sc_forecast_sequence_ts_model_train_and_predict_skc/"
base_dir = "./data_cancel/"
saved_model_path = base_dir + "best_model.pth"
local_train_data_path = base_dir +  "sequence_data.csv"
local_future_dau_plan_path = base_dir +  "savana_future_daus.csv"
local_evaluated_result_path = base_dir +  "evaluated_result.parquet"
local_predicted_result_path = base_dir +  'output.parquet'
s3_saved_model_path = 's3://warehouse-algo/sequence_model_predict_best_model/ds=%s/'
EPOCH = 2
sample_thresh = 0.8