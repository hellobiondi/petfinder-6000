from sagemaker.feature_store.feature_group import FeatureGroup
import time


def store_feature(data, data_name, feature_store_session):
    feature_group_name = f"{data_name}-feature-group"
    feature_group = FeatureGroup(
        name=feature_group_name, sagemaker_session=feature_store_session
    )

    feature_group.load_feature_definitions(data_frame=data)

    while feature_group.describe().get("FeatureGroupStatus") != "Created":
        time.sleep(30)

    # load data into feature group
    feature_group.ingest(data_frame=data, max_workers=3, wait=True)
