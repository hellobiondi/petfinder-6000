import boto3
import os
import shutil
import re
import logging
import pandas as pd
import glob

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def load_data(data_bucket, object, data_type):
    object_prefix = f"{object}_"
    local_path = "/opt/ml/processing/data/{object}/"

    if not os.path.exists(local_path):
        os.makedirs(local_path)
    else:
        shutil.rmtree(local_path)
        os.makedirs(local_path)

    pattern = r"[0-9]+"
    s3 = boto3.client("s3")

    result = s3.list_objects(Bucket=data_bucket, Prefix=object_prefix, Delimiter="/")
    subfolders = [
        re.search(pattern, o.get("Prefix")).group()
        for o in result.get("CommonPrefixes")
    ]
    subfolders.sort(reverse=True)

    object_path = object_prefix + subfolders[0] + "/"
    files = s3.list_objects(Bucket=data_bucket, Prefix=object_path, Delimiter="/")

    pattern = rf"{object_path}(.+)"
    for content in files.get("Contents"):
        file_path = content.get("Key")
        filename = re.findall(pattern, file_path)[0]
        print(filename)

        with open(local_path + filename, "wb") as file:
            s3.download_fileobj(Bucket=data_bucket, Key=file_path, Fileobj=file)

    file_list = glob.glob(local_path + "*")

    dfs = []  # an empty list to store the data frames
    for file in file_list:
        if data_type == "json":
            data = pd.read_json(file, lines=True)  # read data frame from json file
        else:
            data = pd.read_csv(file)  # read data frame from csv file
        dfs.append(data)  # append the data frame to the list

    return pd.concat(dfs, ignore_index=True)
