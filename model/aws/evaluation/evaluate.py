# import sys
# import subprocess
#
# # install requirements
# subprocess.check_call([
#     sys.executable, "-m", "pip", "install", "-r",
#     "/opt/ml/processing/requirements/requirements.txt",
# ])

import os
import json
import pandas as pd
import pathlib
import tarfile
import pickle
import logging

from cornac.metrics import FMeasure, NDCG, NCRR
from metrics.serendipity_wrapper import Serendipity
from metrics.combined_eval_method import CombinedBaseMethod
from metrics.harmonic_mean import HarmonicMean

SEED = 2023
VERBOSE = True

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall("./model")

    with open(os.path.join("./model", "model.pkl"), "rb") as file:
        model = pickle.load(file)

    train_path = "/opt/ml/processing/train/"
    test_path = "/opt/ml/processing/test/"
    train_data = pd.read_csv(train_path + "/train.csv", sep=",")
    test_data = pd.read_csv(test_path + "/test.csv", sep=",")
    train_data = train_data[["userID", "catID", "like"]]
    test_data = test_data[["userID", "catID", "like"]]

    eval_method = CombinedBaseMethod.from_splits(
        train_data=train_data.values,
        test_data=test_data.values,
        rating_threshold=1.0,
        seed=SEED,
        verbose=VERBOSE,
        exclude_unknowns=True,
    )

    logger.info("Defining metrics ...")
    metrics = [
        HarmonicMean(10, Serendipity(), FMeasure(k=10), NDCG(), NCRR()),
        Serendipity(),
        FMeasure(k=10),
        NDCG(),
        NCRR(),
    ]

    result = eval_method.evaluate(
        model=model, metrics=metrics, user_based=False, show_validation=False
    )

    # Available metrics to add to model
    logger.info("Printing metrics ...")
    report_dict = {"ranking_metrics": {}}
    for metric, value in result[0].metric_avg_results.items():
        m = ",".join(metric.lower().split(" "))
        report_dict["ranking_metrics"][m] = {
            "value": value,
            "standard_deviation": "NaN",
        }
        logger.info(f"{metric}: {value}")

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
