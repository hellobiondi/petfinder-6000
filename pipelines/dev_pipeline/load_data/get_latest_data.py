import boto3
import logging
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def handler(event, context):
    data_bucket = event["bucket"]
    object_path = event["object_path"]

    strat_object = get_latest_data(data_bucket, object_path)
    train_uri = f"{strat_object}/train/train.csv"
    eval_uri = f"{strat_object}/validation/validation.csv"
    test_uri = f"{strat_object}/test/test.csv"

    return {
        "statusCode": 200,
        "TrainUri": train_uri,
        "EvalUri": eval_uri,
        "TestUri": test_uri,
    }


def get_latest_data(data_bucket, object_path):
    pattern = r"[0-9-]+"

    s3 = boto3.client("s3")
    result = s3.list_objects(Bucket=data_bucket, Prefix=object_path, Delimiter="/")
    subfolders = [
        re.search(pattern, o.get("Prefix")).group()
        for o in result.get("CommonPrefixes")
    ]
    subfolders.sort(reverse=True)

    return f"s3://{data_bucket}/{object_path}{subfolders[0]}"
