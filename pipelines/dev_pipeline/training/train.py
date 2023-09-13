import glob
import os
import shutil
import argparse
import pandas as pd
import pickle
import subprocess
import logging

from sagemaker.session import Session
from sagemaker.experiments import load_run
import boto3

import cornac
from cornac.models import MostPop
from cornac.metrics import FMeasure, NDCG, NCRR
from metrics.serendipity_wrapper import Serendipity
from metrics.combined_eval_method import CombinedBaseMethod
from metrics.harmonic_mean import HarmonicMean

SEED = 2023
VERBOSE = True

result = subprocess.run(["ldd", "--version"], stdout=subprocess.PIPE)
print(result.stdout)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def parse():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--batch_size', type=int, default=100)
    # parser.add_argument('--learning_rate', type=float, default=0.1)

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

    args, _ = parser.parse_known_args()
    return args


def load_data(train, eval):
    # Read in data
    print("Reading data from S3 URIs ...")
    # files = glob.glob(f"{train}/*.*")
    # print(files)

    train_data = pd.read_csv(train + "/train.csv", sep=",")
    test_data = pd.read_csv(eval + "/validation.csv", sep=",")

    # select data
    train_data = train_data[["userID", "catID", "like"]]
    test_data = test_data[["userID", "catID", "like"]]

    return train_data, test_data


def define_experiment(train_data, test_data):
    # load the data and create eval_method
    print("Creating evaluation method ...")
    rs = CombinedBaseMethod.from_splits(
        train_data=train_data.values,
        test_data=test_data.values,
        rating_threshold=1.0,
        exclude_unknowns=False,
        seed=SEED,
        verbose=VERBOSE,
    )

    print("Defining models ...")
    model_name = "most_pop"
    most_pop = MostPop(name=model_name)
    # initialize models
    models = [most_pop]

    # define metrics to evaluate the models
    logger.info("Defining metrics ...")
    metrics = [
        HarmonicMean(10, Serendipity(), FMeasure(k=10), NDCG(), NCRR()),
        Serendipity(),
        FMeasure(k=10),
        NDCG(),
        NCRR(),
    ]

    # put it together in an experiment,
    experiment = cornac.Experiment(
        eval_method=rs,
        models=models,
        metrics=metrics,
        user_based=True,
        save_dir=output_dir,
    )
    return experiment, model_name, most_pop


if __name__ == "__main__":
    logger.info("Training started.")
    args = parse()

    input_dir = "/input"
    if not os.path.exists(input_dir):
        logger.info("Creating input dir ...")
        os.makedirs(input_dir)

    output_dir = "/output"
    if not os.path.exists(output_dir):
        logger.info("Creating output dir ...")
        os.makedirs(output_dir)

    # load and preprocess data from S3
    train_data, test_data = load_data(args.train, args.eval)

    # define cornac experiment
    experiment, model_name, model = define_experiment(train_data, test_data)

    logger.info("Running experiment ...")
    experiment.run()

    logger.info("Run completed. Writing metrics to Sagemaker Experiments ...")
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
                    logger.info(f"{metric}: {value}")
                    run.log_metric(name=metric, value=value)
    else:
        result = experiment.result
        for r in result:
            for metric, value in r.metric_avg_results.items():
                logger.info(f"{metric}: {value}")

    logger.info("Copying log to output data dir ...")
    log_output = glob.glob(f"{output_dir}/*.log")
    logger.debug(log_output)
    for log_file in log_output:
        shutil.copy(log_file, args.output_data_dir)

    logger.info("Copying model to output model dir ...")
    pickle.dump(model, open("model.pkl", "wb"))
    # model_pkl = glob.glob(f"{output_dir}/{model_name}/*.pkl")
    shutil.copy(
        "model.pkl", os.path.join(args.sm_model_dir, "model.pkl").replace("\\", "/")
    )
