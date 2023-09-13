import glob
import os
import shutil
import argparse
import pandas as pd
import pickle
import subprocess
from tqdm import tqdm
import numpy as np

from sagemaker.session import Session
from sagemaker.experiments import load_run
import boto3

import cornac
from cornac.models import VBPR
from cornac.metrics import FMeasure, NDCG, NCRR
from cornac.data import ImageModality
from metrics.serendipity_wrapper import Serendipity
from metrics.combined_eval_method import CombinedBaseMethod
from metrics.harmonic_mean import HarmonicMean

result = subprocess.run(["ldd", "--version"], stdout=subprocess.PIPE)
print(result.stdout)

SEED = 2023
VERBOSE = True


def parse():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # parser.add_argument('--k', type=int, default=50)
    # parser.add_argument('--max_iter', type=int, default=200)
    # parser.add_argument('--learning_rate', type=float, default=0.001)
    # parser.add_argument('--lambda_reg', type=float, default=0.001)

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k2", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--lambda_w", type=float, default=0.01)
    parser.add_argument("--lambda_b", type=float, default=0.01)
    parser.add_argument("--lambda_e", type=float, default=0.0)

    # input data and model directories
    parser.add_argument(
        "--sm_model_dir", type=str, default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument("--model_dir", type=str)
    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )

    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--eval", type=str, default=os.environ.get("SM_CHANNEL_EVAL"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument(
        "--img_features", type=str, default=os.environ.get("SM_CHANNEL_IMAGE_FEATURES")
    )
    parser.add_argument("--catID", type=str, default=os.environ.get("SM_CHANNEL_CATID"))

    args, _ = parser.parse_known_args()
    return args


def load_data(train, eval, test):
    # Read in data
    print("Reading data from S3 URIs ...")
    # files = glob.glob(f"{train}/*.*")
    # print(files)

    train_data = pd.read_csv(train + "/train.csv", sep=",")
    val_data = pd.read_csv(eval + "/validation.csv", sep=",")
    test_data = pd.read_csv(test + "/test.csv", sep=",")

    # select data
    train_data = train_data[["userID", "catID", "like"]]
    val_data = val_data[["userID", "catID", "like"]]
    test_data = test_data[["userID", "catID", "like"]]

    return train_data, val_data, test_data


def define_experiment(train_data, val_data, test_data, img_features, catID):
    # load the data and create eval_method
    print("Initialising features ...")

    # Instantiate ImageModality and TextModality, it makes it convenient to work with visual auxiliary information
    item_image_modality = ImageModality(
        features=img_features, ids=catID, normalized=True
    )

    print("Creating evaluation method ...")
    rs = CombinedBaseMethod.from_splits(
        train_data=train_data.values,
        val_data=val_data.values,
        test_data=test_data.values,
        rating_threshold=1.0,
        exclude_unknowns=True,
        item_image=item_image_modality,
        seed=SEED,
        verbose=VERBOSE,
    )
    print("Defining models ...")
    model_name = "VBPR"
    model = VBPR(
        name=model_name,
        k=args.k,
        k2=args.k2,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_w=args.lambda_w,
        lambda_b=args.lambda_b,
        lambda_e=args.lambda_e,
        trainable=True,
        verbose=VERBOSE,
    )
    # initialize models
    models = [model]

    # define metrics to evaluate the models
    print("Defining metrics ...")
    metrics = [
        HarmonicMean(10, Serendipity(), FMeasure(k=10), NDCG(), NCRR()),
        Serendipity(),
        FMeasure(k=10),
        NDCG(),
        NCRR(),
    ]

    # put it together in an experiment,
    experiment = cornac.Experiment(
        eval_method=rs, models=models, metrics=metrics, save_dir=output_dir
    )
    return experiment, model_name


if __name__ == "__main__":

    print("Train.py started.")
    args = parse()

    input_dir = "/input"
    if not os.path.exists(input_dir):
        print("Creating input dir ...")
        os.makedirs(input_dir)

    output_dir = "/output"
    if not os.path.exists(output_dir):
        print("Creating output dir ...")
        os.makedirs(output_dir)

    # load and preprocess data from S3
    train_data, val_data, test_data = load_data(args.train, args.eval, args.test)

    # define cornac experiment
    experiment, model_name = define_experiment(
        train_data, val_data, test_data, img_features, catID
    )

    print("Running experiment ...")
    experiment.run()

    print("Run completed. Writing metrics to Sagemaker Experiments ...")
    if "EXP_NAME" in os.environ and "RUN_NAME" in os.environ:
        # setup sagemaker for run tracking
        exp_name = os.environ.get("EXP_NAME")
        run_name = os.environ.get("RUN_NAME")
        region = os.environ.get("REGION")
        boto_session = boto3.session.Session(region_name=region)
        sagemaker_session = Session(boto_session=boto_session)

        with load_run(
            experiment_name=exp_name,
            run_name=run_name,
            sagemaker_session=sagemaker_session,
        ) as run:
            # print metrics to sagemaker
            result = experiment.result
            for r in result:
                for metric, value in r.metric_avg_results.items():
                    print(f"{metric}: {value}")
                    run.log_metric(name=metric, value=value)
    else:
        result = experiment.result
        for r in result:
            for metric, value in r.metric_avg_results.items():
                print(f"{metric}: {value}")

    print("Copying log to output data dir ...")
    log_output = glob.glob(f"{output_dir}/*.log")
    print(log_output)
    for log_file in log_output:
        shutil.copy(log_file, args.output_data_dir)

    print("Copying model to output model dir ...")
    model_pkl = glob.glob(f"{output_dir}/{model_name}/*.pkl")
    print(model_pkl)
    shutil.copy(
        model_pkl[0], os.path.join(args.sm_model_dir, "model.pkl").replace("\\", "/")
    )

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
    pass
    # if request_content_type == "application/json":
    #     request_body = json.loads(request_body)
    #     inpVar = request_body["Input"]
    #     return inpVar
    # else:
    #     raise ValueError("This model only supports application/json input")


"""
predict_fn
    input_data: returned array from input_fn above
    model returned model loaded from model_fn above
"""


def predict_fn(input_data, model):
    item_idx2id = list(model.train_set.item_ids)
    user_idx2id = list(model.train_set.user_ids)
    user_data = model.train_set.user_data

    num_users = len(user_idx2id)
    all_items = list(range(len(item_idx2id)))

    user_recomm = {}
    TOPK = 50
    for userIDX in tqdm(range(num_users)):
        userID = user_idx2id[userIDX]

        # Remove seen items (i.e. items above rating threshold considered seen)
        user_seen = user_data[userIDX][0]
        test_items = np.setdiff1d(all_items, user_seen, assume_unique=True)

        reco, scores = model.rank(userIDX, test_items)
        recommendation = f"{' '.join([str(item_idx2id[r]) for r in reco[:TOPK]])}\n"
        user_recomm[userID] = recommendation

    all_reco = ""
    for userID in range(1, num_users + 1):
        all_reco += user_recomm[userID]

    return all_reco
    # return model.predict(input_data)


"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string
"""


def output_fn(prediction, content_type):
    res = int(prediction[0])
    respJSON = {"Output": res}
    return respJSON
