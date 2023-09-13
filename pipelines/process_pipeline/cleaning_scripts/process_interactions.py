import logging
import pandas as pd

from load_data import load_data

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def process_interactions(data_bucket):
    interactions = load_data(data_bucket, "interaction", "csv")

    # rename headers
    cl_interactions = interactions.rename(
        columns={
            "createdAt": "created_at",
            "updatedAt": "updated_at",
        }
    )

    # convert types
    cl_interactions = cl_interactions.astype(
        {
            "like": "int",
            "click": "int",
        }
    )
    cl_interactions["created_at"] = pd.to_datetime(
        cl_interactions["created_at"]
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    cl_interactions["updated_at"] = pd.to_datetime(
        cl_interactions["updated_at"]
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # drop glue columns
    cl_interactions = cl_interactions.drop(
        ["__typename", "_lastChangedAt", "_version"], axis=1
    )

    return cl_interactions
