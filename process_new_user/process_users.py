import re
import logging
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def process_user(users):
    # rename headers
    cl_users = users.rename(
        columns={
            "A_gender": "gender",
            "A_primarycolor": "primary_color",
            "A_agegroup": "age_group",
            "A_energy": "energy_level",
            "A_attention": "attention_need",
            "A_sweetspicy": "personality",
            "A_firstcat": "is_first_cat",
            "A_othercats": "has_other_cats",
            "A_otherdogs": "good_with_other_dogs",
            "A_kids": "good_with_kids",
            "A_employment": "employment",
            "A_homeownership": "home_ownership",
            "A_allergies": "has_allergies",
            "A_adoptionfee": "agree_to_fee",
            "createdAt": "created_at",
            "updatedAt": "updated_at",
        }
    )

    # clean multi-select columns with No Preference options (age, color)
    def clean_multi_select(row):
        if isinstance(row, str):
            arr = row.split(",")
            if (len(arr) > 1) and ("No preference" in arr):
                arr.remove("No preference")
            return [s.lower() for s in arr]
        else:
            arr = row
            if isinstance(row, dict):
                arr = [o.get("S") for o in row]
            if (len(arr) > 1) and ("No preference" in arr):
                arr.remove("No preference")
            return [s.lower() for s in arr]

    cl_users["age_group"] = cl_users["age_group"].map(
        lambda choice: clean_multi_select(choice)
    )
    cl_users["primary_color"] = cl_users["primary_color"].map(
        lambda choice: clean_multi_select(choice)
    )

    # split columns with list (age, color)
    age_groups = ["kitten", "juvenile", "no_preference", "adult", "senior"]
    split_age_groups = cl_users["age_group"].map(
        lambda row: ",".join([str(age in row) for age in age_groups])
    )
    new_age_columns = split_age_groups.str.split(",", expand=True)
    new_age_columns = new_age_columns.applymap(lambda val: (val == "True"))
    # new_age_columns = new_age_columns.astype('bool')
    new_age_columns = new_age_columns.astype("int")

    pattern = re.compile(r"\s|/")
    # new_age_columns.columns = [f'age_{pattern.sub("_", age).lower()}' for age in age_groups]
    cl_users[
        [f'age_{pattern.sub("_", age).lower()}' for age in age_groups]
    ] = new_age_columns
    cl_users = cl_users.drop("age_group", axis=1)

    color_groups = [
        "no_preference",
        "black",
        "calico_tortie",
        "tabby",
        "others",
        "ginger",
        "white",
    ]
    split_color_groups = cl_users["primary_color"].map(
        lambda row: ",".join([str(color in row) for color in color_groups])
    )
    new_color_columns = split_color_groups.str.split(",", expand=True)
    new_color_columns = new_color_columns.applymap(lambda val: (val == "True"))
    # new_color_columns = new_color_columns.astype('bool')
    new_color_columns = new_color_columns.astype("int")

    cl_users[
        [f'primary_color_{pattern.sub("_", color).lower()}' for color in color_groups]
    ] = new_color_columns
    cl_users = cl_users.drop("primary_color", axis=1)

    # convert string fields to lower case (gender, energy_level, attention_need, personality, employment, home_ownership)
    cl_users["gender"] = cl_users["gender"].map(lambda val: val.lower())
    cl_users["energy_level"] = cl_users["energy_level"].map(lambda val: val.lower())
    cl_users["attention_need"] = cl_users["attention_need"].map(lambda val: val.lower())
    cl_users["personality"] = cl_users["personality"].map(lambda val: val.lower())
    cl_users["employment"] = cl_users["employment"].map(lambda val: val.lower())
    cl_users["home_ownership"] = cl_users["home_ownership"].map(lambda val: val.lower())

    # # convert int booleans to actual booleans (is_first_cat, has_other_cats, good_with_other_dogs, good_with_kids, has_allergies, agree_to_fee)
    cl_users["is_first_cat"] = cl_users["is_first_cat"].map(lambda val: (val == 1))
    cl_users["has_other_cats"] = cl_users["has_other_cats"].map(lambda val: (val == 1))
    cl_users["good_with_other_dogs"] = cl_users["good_with_other_dogs"].map(
        lambda val: (val == 1)
    )
    cl_users["good_with_kids"] = cl_users["good_with_kids"].map(lambda val: (val == 1))
    cl_users["has_allergies"] = cl_users["has_allergies"].map(lambda val: (val == 1))
    cl_users["agree_to_fee"] = cl_users["agree_to_fee"].map(lambda val: (val == 1))

    cl_users["created_at"] = pd.to_datetime(cl_users["created_at"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    cl_users["updated_at"] = pd.to_datetime(cl_users["updated_at"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    cl_users = cl_users.astype(
        {
            "is_first_cat": "int",
            "has_other_cats": "int",
            "good_with_other_dogs": "int",
            "good_with_kids": "int",
            "has_allergies": "int",
            "agree_to_fee": "int",
            "created_at": "object",
            "updated_at": "object",
        }
    )

    # drop glue columns
    if (
        "__typename" in cl_users.columns
        and "_lastChangedAt" in cl_users.columns
        and "_version" in cl_users.columns
    ):
        cl_users = cl_users.drop(["__typename", "_lastChangedAt", "_version"], axis=1)

    return cl_users
