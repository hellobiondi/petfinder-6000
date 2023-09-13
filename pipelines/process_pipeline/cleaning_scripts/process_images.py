import cv2
import os
import boto3
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

new_size = 128

s3 = boto3.client("s3")


def vectorise(cws_id, data_bucket, local_path):
    filename = f"cropped_{cws_id}.jpg"
    s3.download_file(Bucket=data_bucket, Key=filename, Filename=local_path + filename)

    img = cv2.imread(local_path + filename)
    try:
        resize = cv2.resize(img, (new_size, new_size))
        resize = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
        resize = np.array(resize.tolist()) / 255.0
        return resize
    except:
        print(filename)


# defining a function to extract features
def img_feature_extraction(img_vectors, pre_model):
    batch_img = []

    # preprocessing and then using pretrained model to extract features
    for image in img_vectors:
        im_toarray = tf.keras.preprocessing.image.img_to_array(image)
        im_toarray = np.expand_dims(image, axis=0)
        im_toarray = tf.keras.applications.mobilenet.preprocess_input(im_toarray)

        batch_img.append(im_toarray)

    batch_img = np.vstack(batch_img)
    features = pre_model.predict(batch_img, batch_size=64)
    features = features.reshape((len(img_vectors), -1))
    return features


def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == object:
            data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame


def process_images(cl_cats):
    data_bucket = "petfinder6000images"
    local_path = "../data/cat_images/"

    if not os.path.exists(local_path):
        os.makedirs(local_path)
    else:
        shutil.rmtree(local_path)
        os.makedirs(local_path)

    cat_images = cl_cats.reset_index()
    cat_images = cat_images.loc[:, ["id", "cws_id", "updated_at"]]
    cat_images["img_shape"] = str([new_size, new_size, 3])
    cat_images["img_vector"] = cat_images["cws_id"].map(
        lambda cws_id: vectorise(cws_id, data_bucket, local_path)
    )

    img_vectors = cat_images["img_vector"].values

    # Using a pretrained model to generate features
    # Importing pretrained MobileNetV2
    mnetv2_base = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(new_size, new_size, 3), include_top=False, weights="imagenet"
    )

    # Freezing layers
    for layer in mnetv2_base.layers:
        layer.trainable = False

    # Extract features
    features = img_feature_extraction(img_vectors, mnetv2_base)

    # add the feature vectors to dataframe
    feature_vectors = pd.Series(features.tolist(), index=cat_images.index)
    cat_images = cat_images.merge(
        feature_vectors.rename("feature_vectors"), left_index=True, right_index=True
    )

    # stringify vectors
    cat_images["feature_vectors"] = cat_images["feature_vectors"].astype(
        pd.StringDtype()
    )

    # cat_images['img_vector'] = cat_images['img_vector'].apply(lambda x: x.reshape(-1).tolist())
    # cat_images['img_vector'] = cat_images['img_vector'].astype(pd.StringDtype())

    # drop img_vector (max length of 358,400)
    cat_images = cat_images.drop(columns=["img_vector"])

    cat_images = cast_object_to_string(cat_images)

    return cat_images
