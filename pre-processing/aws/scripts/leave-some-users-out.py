import pandas as pd
import os
from sklearn.model_selection import train_test_split

input_data_path = os.path.join("/opt/ml/processing/input", "interactions.csv")
df = pd.read_csv(input_data_path)
print("Shape of data is:", df.shape)

test_size = 0.2
validation_size = 0.5  # of test size
random_state = 2023

users = df["userID"].unique()

train_users, test_users = train_test_split(
    users, test_size=test_size, shuffle=True, random_state=random_state
)
validation_users, test_users = train_test_split(
    test_users, test_size=validation_size, shuffle=True, random_state=random_state
)

train_set = df[df["userID"].isin(train_users)]
validation_set = df[df["userID"].isin(validation_users)]
test_set = df[df["userID"].isin(test_users)]

try:
    os.makedirs("/opt/ml/processing/output/train")
    os.makedirs("/opt/ml/processing/output/validation")
    os.makedirs("/opt/ml/processing/output/test")
    print("Successfully created directories")
except Exception as e:
    # if the Processing call already creates these directories (or directory otherwise cannot be created)
    print(e)
    print("Could not make directories")
    pass

try:
    train_set.to_csv("/opt/ml/processing/train/train.csv")
    validation_set.to_csv("/opt/ml/processing/validation/validation.csv")
    test_set.to_csv("/opt/ml/processing/test/test.csv")
    print("Wrote files successfully")
except Exception as e:
    print("Failed to write the files")
    print(e)
    pass

print("Completed running the processing job")
