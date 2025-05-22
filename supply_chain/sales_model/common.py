from datetime import timedelta, date, datetime
import pandas as pd
import numpy as np
import math
import torch
from torch import nn
import boto3
import os
import json
import random
import time
from pyarrow import parquet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

s3_cli = boto3.client('s3')
EPOCH = 2
sample_thresh = 0.8
BUCKET = "warehouse-algo"


def parquet_iter(parquet_file):
    """
    读取parquet文件
    """
    f = parquet.ParquetFile(parquet_file)
    for batch in f.iter_batches():
        for msg in batch.to_pylist():
            yield msg
    f.close()


def load_record(tmp_file, fmt):
    """
    对文件进行行级别的遍历
    """
    if fmt == 'parquet':
        for v in parquet_iter(tmp_file):
            yield v
    elif fmt == 'lines':
        with open(tmp_file, 'rt') as f:
            for v in f.readlines():
                yield v
    elif fmt == 'csv':
        with open(tmp_file, 'rt') as f:
            header = {name: idx for idx, name in enumerate(f.readline().strip('\n').split('\t'))}
            for line in f:
                items = line.strip('\n').split('\t')
                yield {k: items[v] for k, v in header.items()}


def load_s3_dir(bucket, prefix, days_range, save_path, fmt='parquet'):
    """
    从s3下载某个目录下的所有文件，并读取这些文件，逐行返回
    """
    days_range = {k: 1 for k in days_range}
    paginator = s3_cli.get_paginator('list_objects_v2')
    page_iter = paginator.paginate(Bucket=bucket, Prefix=prefix)
    file_list = sum([[v['Key'] for v in page.get('Contents', [])] for page in page_iter], [])

    # print(file_list)

    def get_file_ds(filename):
        if "ds=" not in filename:
            return ""
        else:
            ds = filename.split("ds=")[1].split("/")[0]
            return ds

    if days_range is not None:
        file_list = [v for v in file_list if get_file_ds(v) in days_range]

    print(file_list)

    index = 0
    for key in file_list:
        success = False
        for _ in range(10):
            s3_cli.download_file(Bucket=bucket, Key=key, Filename=save_path)
            success = True
            break

        if success:
            for record in load_record(save_path, fmt):
                yield record
                index += 1
                if not index % 100000:
                    print('Loading ', index, 'records', end=', ')


def get_s3_file_list_by_prefix(bucket, prefix, with_bucket_prefix=False):
    paginator = s3_cli.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    file_list = sum([[v['Key'] for v in page.get('Contents', [])] for page in pages], [])
    print('List S3 %d files in prefix: %s' % (len(file_list), prefix))
    return [k if with_bucket_prefix else k for k in file_list]


def wait_for_ready(tfr_sample_dir, target_date, wait_window=3600 * 24, wait_interval=60 * 5):
    begin = time.time()
    while True:
        now = time.time()
        if now - begin > wait_window:
            raise
        done_file = tfr_sample_dir + 'ds=' + target_date
        print('done_file', done_file)
        done_file_exist = len(get_s3_file_list_by_prefix(BUCKET, done_file))
        if (not done_file_exist) or done_file_exist < 1:
            print('done_file_exist', done_file_exist)
            print('Done file %s not exists, continue wait after %.2f hours' % (
                done_file, (now - begin) / 3600))
            time.sleep(wait_interval)
            continue

        print('Done file %s updated, start processing.' % done_file, done_file_exist)
        return target_date


class MultiLSTM(nn.Module):
    def __init__(self, num_id_features, id_embedding_dims, numeric_sequence_features,
                 numeric_extra_features, hidden_size, num_layers):
        super(MultiLSTM, self).__init__()

        # 编码ID类特征
        self.id_embeddings = nn.ModuleList([
            nn.Embedding(num_id_features[i], id_embedding_dims[i])
            for i in range(len(num_id_features))
        ])
        self.numeric_extra_features = numeric_extra_features
        self.embedded_id_dim = sum(id_embedding_dims)
        self.week_emb = nn.Embedding(4, 4)
        # LSTM输入维度：嵌入后的ID特征维度 + 数值序列特征维度
        self.lstm_input_dim = self.embedded_id_dim + numeric_sequence_features
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False  # 单向LSTM
        )

        # 全连接层
        fc_input_dim = hidden_size + numeric_extra_features + 4  # LSTM输出维度 + 额外数值特征维度
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # 输出一个连续值
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_sequence, x_numeric_extra):
        k = len(self.id_embeddings)
        id_features = x_sequence[:, :, :k]
        numeric_sequence = x_sequence[:, :, k:]

        week = x_numeric_extra[:, self.numeric_extra_features].long()
        week_emb = self.week_emb(week)
        x_numeric_extra = x_numeric_extra[:, :self.numeric_extra_features]

        # 嵌入ID类特征
        embedded_ids = []
        for i in range(k):
            # 提取每个时间步的第i个ID特征
            id_feature = id_features[:, :, i].long()
            embedding = self.id_embeddings[i](id_feature)
            embedded_ids.append(embedding)
        # 将嵌入后的ID特征拼接
        embedded_ids_concat = torch.cat(embedded_ids, dim=-1)

        # 与数值序列特征拼接
        lstm_input = torch.cat((embedded_ids_concat, numeric_sequence), dim=-1)

        # LSTM处理序列数据
        lstm_out, _ = self.lstm(lstm_input)
        # 取最后一个时间步的LSTM输出作为特征表示
        lstm_features = lstm_out[:, -1, :]

        # 融合额外的数值特征
        combined = torch.cat((lstm_features, x_numeric_extra, week_emb), dim=1)
        # combined = self.dropout(combined)

        # 全连接层预测
        output = self.fc(combined)
        output = output.flatten()
        return output


def smooth_flash(df):
    if df["is_flashsale"].sum() < 1:
        return df
    df['adjusted_sales'] = df['sales_1d']
    df = df.reset_index(drop=True)

    # 遍历处理每一行
    for i in range(len(df)):
        if df.iloc[i]['is_flashsale'] == 1:
            window_data = df.iloc[i - 3:i]
            window_data = window_data['adjusted_sales']
            # 计算平均值（忽略窗口不足情况）
            if not window_data.empty:
                df.at[i, 'adjusted_sales'] = int(window_data.mean())
    # 将结果替换回原列
    df['sales_1d'] = df['adjusted_sales']
    df.drop('adjusted_sales', axis=1, inplace=True)
    return df


def max_digit_place_double(n: int) -> int:
    """
    找到整数的最高位数字对应的位权值，并返回该位权值乘以2的结果。
    参数:
        n (int): 输入的整数（可以为正数或负数）
    返回:
        int: 最高位位权值的两倍
    """
    if n == 0:
        return 0  # 特殊情况处理
    num_str = str(abs(n))  # 转换为正数并转为字符串
    highest_digit = int(num_str[0])  # 提取最高位数字
    place_value = 10 ** (len(num_str) - 1)  # 计算最高位对应的位权值
    return highest_digit * place_value * 2  # 返回位权值的两倍


class DataConfig:
    def __init__(self, target_field, time_delta=0):
        self.code = "skc_id"
        self.static_id_features = ["skc_id", "cid1", "cid2", "cid3", "cid4", "goods_id"]
        self.dynamic_id_features = ["is_flashsale"]
        self.id_features = self.static_id_features + self.dynamic_id_features
        self.target = target_field
        self.time_idx = "target_date"
        self.normalized_features = ["full_size", "discount_rate"]
        self.to_normalized_features = {"temperature": 30, "dau": 500000, "show_times": 10000, "click_times": 2000}
        self.na_features = {"discount_rate": 1.0, "temperature": 30}
        self.features = self.id_features + self.normalized_features + list(self.to_normalized_features) + [self.target]
        self.pos_weight = 1.5

        self.min_predict_length = 3
        self.ideal_predict_length = 28
        self.future_days = 28
        self.min_predict_num = 0
        self.div = 100

        self.future_date2dau = {}
        self.df = None
        self.codes = set()
        self.id_feature_num = {}
        self.future_28_day_daus = []

        today = date.today()
        today = today - timedelta(days=time_delta)
        yesterday = today - timedelta(days=1)
        self.today = today
        self.yesterday = yesterday

    def init_df(self, filename):
        self.df = pd.read_csv(filename)
        for na_feature, default in self.na_features.items():
            self.df[na_feature] = self.df[na_feature].fillna(default)
        self.codes = set(self.df[self.code].values.tolist())
        self.id_feature_num = {id_feature: max_digit_place_double(self.df[id_feature].max()) for id_feature in
                               self.id_features}

    def process_code(self, split, mode="train"):
        sequence_features = []
        to_predict_week_features = []
        real_sell_num = []
        train_period_mean = []
        labels = []
        iter_codes = self.codes
        if mode == "train":
            self.df = self.df[self.df[self.time_idx] < split]
            print("the max train date is", self.df[self.time_idx].max())
        if mode == "eval":
            # iter_codes = list(self.codes)[-1000:]
            split_date = datetime.strptime(split, "%Y-%m-%d")
            split_date = split_date - timedelta(days=28)
            split_date = split_date.strftime("%Y-%m-%d")
            self.df = self.df[self.df[self.time_idx] >= split_date]
            print("the eval date min date is", self.df[self.time_idx].min(), "eval date max date is:", self.df[self.time_idx].max())

        count = 0
        for code in iter_codes:
            if count % 500 == 0:
                print("step: %d / %d" % (count, len(iter_codes)))
            count += 1
            sub_df = self.df[self.df[self.code] == code].copy(deep=True)
            if sub_df[self.target].sum() < self.min_predict_num:
                continue
            if len(sub_df) < self.min_predict_length + self.future_days:
                continue
            sub_df = sub_df.sort_values(by=self.time_idx)
            sub_df = smooth_flash(sub_df)
            if len(sub_df) < self.ideal_predict_length + self.future_days:
                tile_df = sub_df.iloc[0]
                repeated_arr = np.repeat(tile_df.values.reshape(1, -1),
                                         self.ideal_predict_length + self.future_days - len(sub_df), axis=0)
                repeated_df = pd.DataFrame(repeated_arr, columns=sub_df.columns)
                for id_feature in self.id_features:
                    repeated_df[id_feature] = self.id_feature_num[id_feature] + 1
                repeated_df[self.target] = 0
                for normalized_feature in self.normalized_features:
                    repeated_df[normalized_feature] = 1.0
                for to_normalized_feature in self.to_normalized_features.keys():
                    repeated_df[to_normalized_feature] = 0.0
                sub_df = pd.concat([repeated_df, sub_df])
            for feature_name, div in self.to_normalized_features.items():
                sub_df[feature_name] /= div
            sub_df = sub_df[self.features]
            for i in range(len(sub_df) - self.ideal_predict_length - self.future_days + 1):
                if random.random() < sample_thresh and mode == "train":
                    continue
                train_df = sub_df.iloc[i:i + self.ideal_predict_length].copy()
                train_df[self.target] = np.log(train_df[self.target] + 1)
                for j in range(4):
                    to_predict = sub_df[
                                 i + self.ideal_predict_length + j * 7:i + self.ideal_predict_length + (j + 1) * 7]
                    mean_target_rate = np.log(to_predict[self.target].mean() + 1)
                    mean_dau_rate = train_df["dau"].mean()
                    sequence_features.append(train_df)
                    to_predict_week_features.append([mean_dau_rate, j])
                    real_sell_num.append(to_predict[self.target].sum())
                    train_period_mean.append(0)
                    labels.append(mean_target_rate)
        return sequence_features, to_predict_week_features, real_sell_num, train_period_mean, labels

    def process_to_predict_code(self):
        sequence_features = []
        to_predict_week_features = []
        to_predict_codes = []
        iter_codes = self.codes
        self.df = self.df[
            self.df["target_date"] >= (self.today - timedelta(days=self.ideal_predict_length)).strftime("%Y-%m-%d")]
        count = 0
        for code in iter_codes:
            if code is None:
                continue
            if count % 500 == 0:
                print("step: %d / %d" % (count, len(iter_codes)))
            count += 1
            sub_df = self.df[self.df[self.code] == code].copy(deep=True)
            if sub_df[self.target].sum() < self.min_predict_num:
                continue
            if len(sub_df) < self.min_predict_length:
                continue
            sub_df = sub_df.sort_values(by=self.time_idx)
            sub_df = smooth_flash(sub_df)
            if len(sub_df) < self.ideal_predict_length:
                tile_df = sub_df.iloc[0]
                repeated_arr = np.repeat(tile_df.values.reshape(1, -1),
                                         self.ideal_predict_length - len(sub_df), axis=0)
                repeated_df = pd.DataFrame(repeated_arr, columns=sub_df.columns)
                for id_feature in self.id_features:
                    repeated_df[id_feature] = self.id_feature_num[id_feature] + 1
                repeated_df[self.target] = 0
                for normalized_feature in self.normalized_features:
                    repeated_df[normalized_feature] = 1.0
                for to_normalized_feature in self.to_normalized_features.keys():
                    repeated_df[to_normalized_feature] = 0.0
                sub_df = pd.concat([repeated_df, sub_df])
            for feature_name, div in self.to_normalized_features.items():
                sub_df[feature_name] /= div
            sub_df = sub_df[self.features]
            for i in range(len(sub_df) - self.ideal_predict_length + 1):
                train_df = sub_df.iloc[i:i + self.ideal_predict_length].copy()
                train_df[self.target] = np.log(train_df[self.target] + 1)
                for j in range(4):
                    # mean_dau_rate = np.mean(self.future_28_day_daus[i * 7:(i + 1) * 7])
                    mean_dau_rate = train_df["dau"].mean()
                    sequence_features.append(train_df)
                    to_predict_week_features.append([0 if mean_dau_rate is not None else mean_dau_rate, j])
                    to_predict_codes.append(code)
        return sequence_features, to_predict_week_features, to_predict_codes


def prepare_train_valid_data(dc, split):
    sequence_features_, to_predict_week_features_, real_sell_num_, train_period_mean_, labels_ = dc.process_code(split,
                                                                                                                 mode="train")
    labels_ = np.asarray(labels_)
    # labels_ = (np.log(labels_ + 1) - 1).tolist()
    zip_temp = list(zip(sequence_features_, to_predict_week_features_, labels_))
    train_zip, test_zip = train_test_split(zip_temp, test_size=0.05)

    train_sequence_features_ = [item[0] for item in train_zip]
    train_to_predict_features_ = [item[1] for item in train_zip]
    train_labels_ = [item[2] for item in train_zip]

    print(np.mean(train_labels_), np.std(train_labels_))

    test_sequence_features_ = [item[0] for item in test_zip]
    test_to_predict_features_ = [item[1] for item in test_zip]
    test_labels_ = [item[2] for item in test_zip]

    print(np.mean(test_labels_), np.std(test_labels_))

    train_sequence_features_ = np.asarray(train_sequence_features_).astype(np.float32)
    train_to_predict_features_ = np.asarray(train_to_predict_features_).astype(np.float32)
    train_labels_ = np.asarray(train_labels_).astype(np.float32)

    test_sequence_features_ = np.asarray(test_sequence_features_).astype(np.float32)
    test_to_predict_features_ = np.asarray(test_to_predict_features_).astype(np.float32)
    test_labels_ = np.asarray(test_labels_).astype(np.float32)

    print(train_sequence_features_.shape, test_sequence_features_.shape)

    train_dataset = TensorDataset(torch.from_numpy(train_sequence_features_),
                                  torch.from_numpy(train_to_predict_features_),
                                  torch.from_numpy(train_labels_))
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_dataset = TensorDataset(torch.from_numpy(test_sequence_features_), torch.from_numpy(test_to_predict_features_),
                                 torch.from_numpy(test_labels_))
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)
    return train_loader, test_loader


def evaluate_train_loss(test_model, test_loader, dc):
    test_model.eval()  # 设置为评估模式
    test_losses = []
    abs_losses = []
    diffs = []
    with torch.no_grad():
        for test_sequence_features_batch, test_to_predict_features_batch, test_labels_batch in test_loader:
            test_outputs = test_model(test_sequence_features_batch, test_to_predict_features_batch)
            weight = torch.where(test_outputs - test_labels_batch > 0, dc.pos_weight, 1)
            loss = torch.mean((test_outputs - test_labels_batch) * (test_outputs - test_labels_batch) * weight)
            diffs.extend((test_outputs - test_labels_batch).numpy().tolist())
            abs_loss = torch.mean(torch.abs(test_outputs - test_labels_batch))
            test_losses.append(loss.item())
            abs_losses.append(abs_loss.item())
    test_loss = sum(test_losses) / len(test_losses)
    abs_loss = sum(abs_losses) / len(abs_losses)
    abs_diffs = [abs(item) for item in diffs]
    print("the test_loss is %f ,the abs_loss is %f" % (test_loss, abs_loss))
    print("diffs:", np.mean(diffs), np.std(diffs))
    print("abs diffs:", np.mean(abs_diffs), np.std(abs_diffs))
    return test_loss


def save_model(to_save_model, path):
    torch.save(to_save_model.state_dict(), path)


def train(dc, train_loader, test_loader, model_path):
    id2num = dc.id_feature_num
    num_id_features = [id2num[item] + 2 for item in dc.id_features]
    id_embedding_dims = [4 for _ in num_id_features]
    numeric_sequence_features = len(dc.features) - len(dc.id_features)
    numeric_extra_features = 1
    hidden_size = 16
    num_layers = 1
    model = MultiLSTM(
        num_id_features=num_id_features,
        id_embedding_dims=id_embedding_dims,
        numeric_sequence_features=numeric_sequence_features,
        numeric_extra_features=numeric_extra_features,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    # model.load_state_dict(torch.load(saved_model_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = EPOCH
    train_batch_count = 0
    train_losses = []
    max_self_defined_score = 10000
    model.train()
    for epoch in range(num_epochs):
        for i, (train_sequence_batch, train_to_predict_features_batch, train_labels_batch) in enumerate(train_loader):
            if train_batch_count % 800 == 0:
                self_defined_score = evaluate_train_loss(model, test_loader, dc)
                model.train()
                if self_defined_score < max_self_defined_score:
                    max_self_defined_score = self_defined_score
                    save_model(model, model_path)
                    print("the max self defined score is %f, saved model to %s" % (
                        self_defined_score, model_path))
                model.train()
                pass
            outputs = model(train_sequence_batch, train_to_predict_features_batch)
            weight = torch.where(outputs - train_labels_batch > 0, dc.pos_weight, 1)
            loss = torch.mean((outputs - train_labels_batch) * (outputs - train_labels_batch) * weight)
            train_losses.append(loss.item())

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_batch_count % 800 == 0:
                mean_loss = np.mean(train_losses)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {mean_loss:.4f}')
                train_losses = []
                # print(outputs)

            train_batch_count += 1


def get_saved_model(dc, local_predict_dir):
    models = []
    saved_model_pths = os.listdir(local_predict_dir)
    print('local_predict_dir', saved_model_pths)
    id2num = dc.id_feature_num
    num_id_features = [id2num[item] + 2 for item in dc.id_features]
    id_embedding_dims = [4 for _ in num_id_features]
    numeric_sequence_features = len(dc.features) - len(dc.id_features)
    numeric_extra_features = 1
    hidden_size = 16
    num_layers = 1
    for filename in saved_model_pths:
        if "best_model" not in filename:
            continue
        filename = local_predict_dir + filename
        print("load model state from: %s" % filename)
        model = MultiLSTM(
            num_id_features=num_id_features,
            id_embedding_dims=id_embedding_dims,
            numeric_sequence_features=numeric_sequence_features,
            numeric_extra_features=numeric_extra_features,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        model.load_state_dict(torch.load(filename))
        model.eval()
        models.append(model)
    return models


def daily_predict(dc, local_predict_dir):
    def reverse_predict(val):
        val = math.exp(val) - 1
        return val

    saved_models = get_saved_model(dc, local_predict_dir)
    sequence_features, to_predict_week_features, to_predict_codes = dc.process_to_predict_code()
    code_predict_results = []
    debug_model = set()
    debug_n = 2
    for code, to_predict_week_feature, sequence_feature in zip(to_predict_codes, to_predict_week_features,
                                                               sequence_features):
        week_num = to_predict_week_feature[1] + 1
        sequence_feature = torch.from_numpy(np.asarray([sequence_feature], dtype=np.float32))
        to_predict_week_feature = torch.from_numpy(np.asarray([to_predict_week_feature], dtype=np.float32))

        for i, saved_model in enumerate(saved_models):
            # print('predict use model:', i)
            model_pred = saved_model(sequence_feature, to_predict_week_feature)
            model_pred = model_pred.detach().numpy().tolist()[0]
            real_predict_num = int(0.001 * 7)
            try:
                if model_pred is not None:
                    pred_new = reverse_predict(model_pred)
                    real_predict_num = int(pred_new * 7)
            except Exception as e:
                debug_model.add(i)
                if debug_n > 0:
                    print('exception', e)
                    debug_n -= 1
                    print(sequence_feature, to_predict_week_feature)
                pass

            code_predict_results.append([code, week_num, real_predict_num])
    print('debug_model', debug_model)
    code_predict_results = pd.DataFrame(code_predict_results, columns=["skc_id", "week_num", "predict"])
    return code_predict_results


def daily_predict_thread(proc_id, shared_list,saved_model, sequence_features, to_predict_week_features, to_predict_codes):
    def reverse_predict(val):
        val = math.exp(val) - 1
        return val
    st = time.time()

    code_predict_results = []
    except_id = set()
    debug_n = 2
    count = 0
    except_n = 0
    n = len(to_predict_codes)
    for code, to_predict_week_feature, sequence_feature in zip(to_predict_codes, to_predict_week_features,
                                                               sequence_features):
        if count % 5000 == 0:
            print(f"proc_id:{proc_id} process count:{count} / {n}")
        week_num = to_predict_week_feature[1] + 1
        sequence_feature = torch.from_numpy(np.asarray([sequence_feature], dtype=np.float32))
        to_predict_week_feature = torch.from_numpy(np.asarray([to_predict_week_feature], dtype=np.float32))

        # print('predict use model:', i)
        model_pred = saved_model(sequence_feature, to_predict_week_feature)
        model_pred2 = model_pred.detach().numpy().tolist()[0]
        real_predict_num = int(0.3 * 7)
        count += 1
        try:
            if model_pred2 is not None:
                pred_new = reverse_predict(model_pred2)
                real_predict_num = int(pred_new * 7)
        except Exception as e:
            except_n += 1
            except_id.add(code)
            if debug_n > 0:
                print('exception', e)
                debug_n -= 1
                print(sequence_feature, to_predict_week_feature)
            pass

        code_predict_results.append([code, week_num, real_predict_num])
    print('except_n', except_n)
    print('except_id_', len(except_id))
    print('except_id', except_id)
    shared_list.append(code_predict_results)
    print(f"proc_id:{proc_id} cost:{time.time()-st}")
    # code_predict_results = pd.DataFrame(code_predict_results, columns=["skc_id", "week_num", "predict"])
    # return code_predict_results


def evaluate_model(dc, local_eval_path, model_path, data_smooth_eval_path):
    def reverse_predict(val):
        val = math.exp(val) - 1
        return val

    saved_model = get_saved_model(dc, model_path)[0]
    abs_diffs = []
    real_abs_diffs = []
    fast_ratio = []
    hive_parquet = []
    sequence_features_, to_predict_week_features_, real_sell_nums_, train_period_means_, labels_ = dc.process_code(
        split=(dc.today - timedelta(days=28)).strftime('%Y-%m-%d'),mode="eval"
    )
    labels_ = np.asarray(labels_)
    for sequence_feature, to_predict_week_feature, real_sell_num, train_period_mean, label in zip(sequence_features_,
                                                                                                  to_predict_week_features_,
                                                                                                  real_sell_nums_,
                                                                                                  train_period_means_,
                                                                                                  labels_):
        week_num = to_predict_week_feature[1] + 1
        skc_id = str(sequence_feature.iloc[-1]["skc_id"])
        goods_id = str(sequence_feature.iloc[-1]["goods_id"])
        sequence_feature = torch.from_numpy(np.asarray([sequence_feature], dtype=np.float32))
        to_predict_week_feature = torch.from_numpy(np.asarray([to_predict_week_feature], dtype=np.float32))
        model_pred = saved_model(sequence_feature, to_predict_week_feature)
        model_pred = model_pred.detach().numpy().tolist()[0]
        abs_diffs.append(abs(model_pred - label))
        real_predict_num = int(reverse_predict(model_pred) * 7)
        if real_sell_num > 500:
            print(model_pred, label, int(reverse_predict(label) * 7),
                  real_sell_num, real_predict_num)

        real_abs_diffs.append((real_sell_num - real_predict_num, real_predict_num))
        fast_ratio.append((min(real_sell_num, real_predict_num), real_sell_num))
        hive_parquet.append(
            [goods_id, skc_id, week_num, dc.yesterday.strftime("%Y%m%d"), real_sell_num, real_predict_num])

    # print(np.mean(abs_diffs), np.std(abs_diffs))
    #
    # # 卖的比备货多的
    # real_pos_diff_sum = sum([item[0] if item[0] > 0 else 0 for item in real_abs_diffs])
    # # 备货比卖的多的占比
    # real_neg_diff_sum = sum([-item[0] if item[0] < 0 else 0 for item in real_abs_diffs])
    # predict_sell_sum = sum([item[1] for item in real_abs_diffs])
    # print(real_pos_diff_sum, predict_sell_sum, real_pos_diff_sum / predict_sell_sum)
    # print(real_neg_diff_sum, predict_sell_sum, real_neg_diff_sum / predict_sell_sum)
    #
    # total_sell_num = sum([item[1] for item in fast_ratio])
    # fast_num = sum([item[0] for item in fast_ratio])
    # print(fast_num, total_sell_num, fast_num / total_sell_num)

    hive_parquet = pd.DataFrame(hive_parquet,
                                columns=["goods_id", "skc_id", "week_num", "date", "real_sell_num", "real_predict_num"])
    hive_parquet.to_parquet(local_eval_path)
    print("hive_parquet:", hive_parquet.shape)


def download_file(from_file, to_file):
    try:
        os.system('aws s3 cp %s %s'%(from_file, to_file))
        # s3_cli.download_file(Bucket="warehouse-algo", Key=key, Filename=fname)
    except Exception as e:
        # traceback.print_exc()
        print(Exception, e)
        raise ValueError('Failed to download file: %s to %s' % (from_file, to_file))
