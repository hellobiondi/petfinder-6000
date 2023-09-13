from cornac.metrics import RankingMetric
from recommenders.evaluation.python_evaluation import user_serendipity
import pandas as pd


class Serendipity(RankingMetric):
    def __init__(self, k=-1, *args):
        RankingMetric.__init__(self, name="Serendipity", k=k)

    def compute(self, gt_pos, pd_rank, seen_items, reco_items, **kwargs):
        if len(seen_items) > 0:
            train_df = pd.concat(
                [
                    pd.Series(1, index=range(len(seen_items)), dtype="int64"),
                    pd.Series(seen_items),
                ],
                axis=1,
                keys=["userID", "catID"],
            )
        else:
            train_df = pd.DataFrame(columns=["userID", "catID"])

        rank = [r for r in pd_rank if r not in seen_items]
        reco_df = pd.concat(
            [pd.Series(1, index=range(len(rank)), dtype="int64"), pd.Series(rank)],
            axis=1,
            keys=["userID", "catID"],
        )
        reco_df["relevance"] = reco_df["catID"].map(
            lambda id: reco_items.get(id) if id in reco_items else 0
        )

        # convert all columns to string
        train_df = train_df.astype(int)
        reco_df = reco_df.astype(int)
        # print(train_df)
        # print(reco_df)
        ser = user_serendipity(
            train_df,
            reco_df,
            col_user="userID",
            col_item="catID",
            col_relevance="relevance",
        )
        if len(ser) > 0:
            return ser.loc[0, "user_serendipity"]
        else:
            return 0
