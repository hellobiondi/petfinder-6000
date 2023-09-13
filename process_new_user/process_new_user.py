import os
import pandas as pd
import numpy as np
import re
import json
import boto3
from process_users import process_user

region = os.environ.get("REGION")
boto_session = boto3.Session()


# retrieve users from S3, and returns a dataframe
def retrieve_dataframe():
    pattern = r"[0-9-]+"
    s3 = boto3.client("s3")

    data_bucket = "petfinder6000"
    object_prefix = "auxiliary/users/users"

    result = s3.list_objects(Bucket=data_bucket, Prefix=object_prefix, Delimiter="/")
    subfolders = [
        re.search(pattern, o.get("Prefix")).group()
        for o in result.get("CommonPrefixes")
    ]
    subfolders.sort(reverse=True)

    file_path = object_prefix + subfolders[0] + "/users.csv"
    print(file_path)
    obj = s3.get_object(Bucket=data_bucket, Key=file_path)

    df = pd.read_csv(obj["Body"], header=0)
    return df


def calculate_cosine_similarities(matrix):
    # Convert the input matrix to a numpy array
    matrix = np.array(matrix)

    # check if all elements in matrix are numeric
    if not np.issubdtype(matrix.dtype, np.number):
        print("Input matrix contains non-numeric values")

    # convert to float anyway
    matrix = matrix.astype(float)

    # Calculate the norms of each row
    norms = np.linalg.norm(matrix, axis=1)

    # Compute the dot product of each pair of rows
    dot_products = np.matmul(matrix, matrix.T)

    # Divide the dot products by the norms to get cosine similarities
    similarities = dot_products / np.outer(norms, norms)

    return similarities


# find the closest userID with the closest cosineSim and return their ID
def return_closest_similarity_id(df, attributes_dict):
    df = pd.concat([df, attributes_dict], ignore_index=True)
    new_user_idx = len(df) - 1

    # some pre-processing
    df.drop(
        df.columns[df.columns.str.contains("unnamed", case=False)], axis=1, inplace=True
    )
    id_list = df["id"].tolist()
    df.drop(["id", "username", "created_at", "updated_at"], axis=1, inplace=True)
    categorical_columns = [
        "energy_level",
        "attention_need",
        "personality",
        "employment",
        "home_ownership",
        "gender",
    ]
    df = pd.get_dummies(df, columns=categorical_columns)
    df = df * 1  # convert to int 0 or 1

    cosine_sim_matrix = calculate_cosine_similarities(df)
    user_similarity_list = cosine_sim_matrix[new_user_idx]
    user_similarity_tuplelist = []
    for idx, cosSim in enumerate(user_similarity_list):
        user_similarity_tuplelist.append((idx, cosSim))

    # choose person most similar to new user
    user_similarity_tuplelist.sort(key=lambda x: x[1], reverse=True)
    # return second most similar as first most similar will be himself
    most_similar_idx = user_similarity_tuplelist[1][0]
    most_similar_score = user_similarity_tuplelist[1][1]

    # corner case handling if person has exact same preferences, return the other person's ID
    if most_similar_idx == new_user_idx:
        most_similar_idx = user_similarity_tuplelist[0][0]
        most_similar_score = user_similarity_tuplelist[0][1]
    most_similar_id = id_list[most_similar_idx]

    print(
        f"The person that is most similar to this cold start chap is of index {most_similar_idx}"
        f" and ID of {most_similar_id} with a similarity score of {most_similar_score}"
    )

    return most_similar_id


def get_rankings():
    s3_client = boto3.client("s3")

    bucket_name = "petfinder6000"
    prefix = "ranking/"

    # List objects in the S3 bucket with the specified prefix
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # Extract the folder names from the object keys
    folder_names = [obj["Key"].split("/")[1] for obj in response["Contents"]]

    # Sort the folder names in descending order
    folder_names.sort(reverse=True)

    # Retrieve the latest folder name
    latest_folder = folder_names[0]
    print(latest_folder)

    # Read the input_data.csv.out file from the latest folder as a dataframe
    file_key = f"{prefix}{latest_folder}/input_data.csv.out"
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    ranking_df = pd.read_csv(obj["Body"])

    return ranking_df


def handler(event, context):
    if "body" in event:
        input_body = event.get("body")
        attributes_dict = json.loads(input_body.rstrip())
        user_df = pd.read_json(input_body, orient="index").T
    else:
        attributes_dict = {}
        for key, value in event.items():
            attributes_dict[key] = value
        attr_json = json.dumps(attributes_dict)
        user_df = pd.read_json(attr_json, orient="index").T

    print(attributes_dict)
    print(user_df)

    # fetch ranking from S3 bucket, and returning recommended cats as a list
    rankings = get_rankings()

    requestor_id = attributes_dict.get("id")
    # check if incoming ID exists in ranking list, and returns a boolean
    record_exists = (requestor_id in rankings["userID"]) and (
        rankings[rankings["userID"] == requestor_id]["reco"] != ""
    )
    print(f"Record exists: {record_exists}")

    if not record_exists:
        df = retrieve_dataframe()

        # preprocess userid
        # user_df = pd.DataFrame(attributes_dict, index=[0])
        processed_user = process_user(user_df)

        # homemade calculation of cosSim without using sklearn
        closest_similarity_id = return_closest_similarity_id(df, processed_user)
        print(df[df["id"] == closest_similarity_id]["username"])
        user_id = closest_similarity_id
    else:
        user_id = attributes_dict["id"]

    recommendations = rankings[rankings["userID"] == user_id]["reco"].to_list()

    json_output = {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
        },
        "body": json.dumps({"ranking": recommendations}),
        "isBase64Encoded": False,
    }

    return json_output
