import boto3
import logging
import sagemaker
import os
from sagemaker.session import Session

from process_users import process_users
from process_cats import process_cats
from process_images import process_images
from process_interactions import process_interactions
from store_feature import store_feature

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    data_bucket = "dynamodbpetfinder"

    # clean data
    cleaned_users = process_users(data_bucket)
    cleaned_cats = process_cats(data_bucket)
    cleaned_cat_images = process_images(cleaned_cats)
    cleaned_interactions = process_interactions(data_bucket)

    # write data to feature store
    bucket = "petfinder6000"
    object = "auxiliary"
    prefix = "sagemaker-featurestore"
    offline_feature_store_bucket = "s3://{}/{}/{}".format(bucket, object, prefix)
    print(f"Offline bucket: {offline_feature_store_bucket}")

    region = os.environ.get("REGION")
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client(service_name="sagemaker", region_name=region)
    featurestore_runtime = boto_session.client(
        service_name="sagemaker-featurestore-runtime", region_name=region
    )

    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime,
    )

    store_feature(cleaned_users, "users", feature_store_session)
    store_feature(cleaned_cats, "cats", feature_store_session)
    store_feature(cleaned_cat_images, "cat-images", feature_store_session)
    store_feature(cleaned_interactions, "interactions", feature_store_session)

    cleaned_interactions.to_csv(
        f"/opt/ml/processing/interactions/interactions.csv", header=True, index=False
    )
    cleaned_users.to_csv(
        f"/opt/ml/processing/users/users.csv", header=True, index=False
    )
