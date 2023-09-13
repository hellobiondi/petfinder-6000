import boto3
import logging
import os
import sagemaker
from sagemaker.session import Session
from sagemaker.feature_store.feature_store import FeatureStore
from sagemaker.feature_store.feature_group import FeatureGroup

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    region = os.environ.get("REGION")
    inference_path = os.environ.get("INFERENCE_PATH")

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name="sagemaker", region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    featurestore_runtime = boto_session.client(
        service_name="sagemaker-featurestore-runtime", region_name=region
    )

    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime,
    )

    users_feature_group_name = "users-feature-group"
    users_feature_group = FeatureGroup(
        name=users_feature_group_name, sagemaker_session=sagemaker_session
    )

    interactions_feature_group_name = "interactions-feature-group"
    interactions_feature_group = FeatureGroup(
        name=interactions_feature_group_name, sagemaker_session=sagemaker_session
    )

    feature_store = FeatureStore(feature_store_session)
    df, query = (
        feature_store.create_dataset(
            base=users_feature_group, output_path=inference_path
        )
        .with_feature_group(
            interactions_feature_group,
            target_feature_name_in_base="id",
            included_feature_names=["userID", "catID"],
            feature_name_in_target="userID",
        )
        .to_dataframe()
    )

    input_data = df.groupby("id")["catID.1"].apply(list).reset_index(name="seen")
    input_data = input_data.rename(columns={"id": "userID"})
    input_data.to_csv(
        "/opt/ml/processing/batch_input/input_data.csv", header=False, index=False
    )
