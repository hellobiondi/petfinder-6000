import os
import pandas as pd
import pickle
import subprocess
import numpy as np
import logging
from io import StringIO

result = subprocess.run(["ldd", "--version"], stdout=subprocess.PIPE)
print(result.stdout)

SEED = 2023
VERBOSE = True

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

try:
    from sagemaker_containers.beta.framework import (
        content_types,
        encoders,
        env,
        modules,
        transformer,
        worker,
        server,
    )
except ImportError:
    pass

# Model serving
"""
Deserialize fitted model
"""


def model_fn(model_dir):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as file:
        model = pickle.load(file)
    return model


"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""


def input_fn(request_body, request_content_type):
    logger.info(f"Received: {request_body}, content_type: {request_content_type}")
    if request_content_type == "text/csv":
        df = pd.read_csv(request_body, header=None, names=["userID", "seen"])
        return df
    else:
        raise ValueError("This model only supports text_csv input")


"""
predict_fn
    input_data: returned array from input_fn above
    model returned model loaded from model_fn above
"""


def generate_rank(row, model):
    user_id2idx = model.train_set.uid_map
    item_idx2id = list(model.train_set.item_ids)
    all_items = list(range(len(item_idx2id)))

    if row["userID"] in user_id2idx:
        userIDX = user_id2idx.get(row["userID"])
        test_items = np.setdiff1d(all_items, row["seen"], assume_unique=True)
        reco, score = model.rank(userIDX, test_items)
        str_reco = f"{','.join([str(item_idx2id[r]) for r in reco])}"
    else:
        str_reco = ""
    return str_reco


def predict_fn(input_data, model):
    input_data["reco"] = input_data.apply(lambda x: generate_rank(x, model), axis=1)
    return input_data[["userID", "reco"]]


"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string
"""


def output_fn(prediction, content_type):
    out = StringIO()
    prediction.to_csv(out, header=True, index=False)
    result = out.getvalue()
    return result
