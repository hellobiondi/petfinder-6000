import logging
import pandas as pd

from load_data import load_data

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# map integers in categorical columns
def map_categorical(val):
    if val == 1:
        return "yes"
    elif val == -1:
        return "no"
    else:
        return "neutral"


def process_cats(data_bucket):
    cats = load_data(data_bucket, "cat", "csv")

    # rename headers
    cl_cats = cats.rename(
        columns={
            "C_gender": "gender",
            "C_primarycolor": "primary_color",
            "C_agegroup": "age_group",
            "C_energy": "energy_level",
            "C_attention": "attention_need",
            "C_sweetspicy": "personality",
            "C_firstcat": "good_first_cat",
            "C_othercats": "good_with_other_cats",
            "C_otherdogs": "good_with_other_dogs",
            "C_kids": "good_with_kids",
            "C_employment": "preferred_employment",
            "C_homeownership": "preferred_home_ownership",
            "C_allergies": "good_with_allergies",
            "C_adoptionfee": "require_fee",
            "createdAt": "created_at",
            "updatedAt": "updated_at",
        }
    )

    # fill attention and personality columns as neutral
    cl_cats["attention_need"] = cl_cats["attention_need"].fillna("neutral")
    cl_cats["personality"] = cl_cats["personality"].fillna("neutral")

    # fill empty description with no description available
    cl_cats["description"] = cl_cats["description"].fillna("no description available")

    cl_cats["good_first_cat"] = cl_cats["good_first_cat"].map(
        lambda val: map_categorical(val)
    )
    cl_cats["good_with_other_cats"] = cl_cats["good_with_other_cats"].map(
        lambda val: map_categorical(val)
    )
    cl_cats["good_with_other_dogs"] = cl_cats["good_with_other_dogs"].map(
        lambda val: map_categorical(val)
    )
    cl_cats["good_with_kids"] = cl_cats["good_with_kids"].map(
        lambda val: map_categorical(val)
    )

    # convert string fields to lower case (gender, breed, primary_color, age_group, energy_level,
    # attention_need, personality, preferred_employment, preferred_home_ownership, require_fee)
    cl_cats["gender"] = cl_cats["gender"].map(lambda val: val.lower())
    cl_cats["breed"] = cl_cats["breed"].map(lambda val: val.lower())
    cl_cats["primary_color"] = cl_cats["primary_color"].map(lambda val: val.lower())
    cl_cats["age_group"] = cl_cats["age_group"].map(lambda val: val.lower())
    cl_cats["energy_level"] = cl_cats["energy_level"].map(lambda val: val.lower())
    cl_cats["attention_need"] = cl_cats["attention_need"].map(lambda val: val.lower())
    cl_cats["personality"] = cl_cats["personality"].map(lambda val: val.lower())
    cl_cats["preferred_employment"] = cl_cats["preferred_employment"].map(
        lambda val: val.lower()
    )
    cl_cats["preferred_home_ownership"] = cl_cats["preferred_home_ownership"].map(
        lambda val: val.lower()
    )
    cl_cats["require_fee"] = cl_cats["require_fee"].map(lambda val: val.lower())

    cl_cats["description"] = cl_cats["description"].map(
        lambda val: val.encode("ascii", "ignore")
        .decode()
        .replace("\n", " ")
        .replace("\r", "")
        .lower()
    )
    cl_cats["details"] = cl_cats["details"].map(
        lambda val: val.encode("ascii", "ignore")
        .decode()
        .replace("\n", " ")
        .replace("\r", "")
        .lower()
    )

    cl_cats["created_at"] = pd.to_datetime(cl_cats["created_at"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    cl_cats["updated_at"] = pd.to_datetime(cl_cats["updated_at"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    cl_cats = cl_cats.astype(
        {
            "playful": "int",
            "active": "int",
            "curious": "int",
            "talkative": "int",
            "quiet": "int",
            "loving": "int",
            "sweet": "int",
            "likes_held": "int",
            "friendly": "int",
            "shy": "int",
            "spicy": "int",
            "loves_attention": "int",
        }
    )

    # drop glue columns
    cl_cats = cl_cats.drop(["__typename", "_lastChangedAt", "_version"], axis=1)

    return cl_cats
