import pandas as pd
import os
from sklearn.model_selection import train_test_split

input_data_path = os.path.join("/opt/ml/processing/input", "interactions.csv")
df = pd.read_csv(input_data_path)
print("Shape of data is:", df.shape)

test_size = 0.2
validation_size = 0.5  # of test size
random_state = 2023

train_set = df.groupby("userID").sample(frac=1 - test_size, random_state=random_state)
test_set = df.drop(train_set.index)

validation_set = test_set.groupby("userID").sample(
    frac=validation_size, random_state=random_state
)
test_set = test_set.drop(validation_set.index)

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
