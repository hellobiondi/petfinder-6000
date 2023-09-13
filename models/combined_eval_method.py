import time
import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
from collections import OrderedDict

from cornac.eval_methods import BaseMethod
from cornac.experiment.result import Result


def rating_eval(model, metrics, test_set, user_based=False, verbose=False):
    if len(metrics) == 0:
        return [], []

    avg_results = []
    user_results = []

    (u_indices, i_indices, r_values) = test_set.uir_tuple
    r_preds = np.fromiter(
        tqdm(
            (
                model.rate(user_idx, item_idx).item()
                for user_idx, item_idx in zip(u_indices, i_indices)
            ),
            desc="Rating",
            disable=not verbose,
            miniters=100,
            total=len(u_indices),
        ),
        dtype="float",
    )

    gt_mat = test_set.csr_matrix
    pd_mat = csr_matrix((r_preds, (u_indices, i_indices)), shape=gt_mat.shape)

    for mt in metrics:
        if user_based:  # averaging over users
            user_results.append(
                {
                    user_idx: mt.compute(
                        gt_ratings=gt_mat.getrow(user_idx).data,
                        pd_ratings=pd_mat.getrow(user_idx).data,
                    ).item()
                    for user_idx in test_set.user_indices
                }
            )
            avg_results.append(sum(user_results[-1].values()) / len(user_results[-1]))
        else:  # averaging over ratings
            user_results.append({})
            avg_results.append(mt.compute(gt_ratings=r_values, pd_ratings=r_preds))

    return avg_results, user_results


def ranking_eval(
    model,
    metrics,
    train_set,
    test_set,
    val_set=None,
    rating_threshold=1.0,
    exclude_unknowns=True,
    verbose=False,
):
    if len(metrics) == 0:
        return [], []

    avg_results = []
    user_results = [{} for _ in enumerate(metrics)]

    gt_mat = test_set.csr_matrix
    train_mat = train_set.csr_matrix
    val_mat = None if val_set is None else val_set.csr_matrix

    def pos_items(csr_row):
        return [
            item_idx
            for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
            if rating >= rating_threshold
        ]

    def thresholded_items(csr_row):
        return {
            item_idx: int(rating >= rating_threshold)
            for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
        }

    for user_idx in tqdm(
        test_set.user_indices, desc="Ranking", disable=not verbose, miniters=100
    ):
        test_pos_items = pos_items(gt_mat.getrow(user_idx))
        if len(test_pos_items) == 0:
            continue

        u_gt_pos = np.zeros(test_set.num_items, dtype="int")
        u_gt_pos[test_pos_items] = 1

        val_pos_items = [] if val_mat is None else pos_items(val_mat.getrow(user_idx))
        train_pos_items = (
            []
            if train_set.is_unk_user(user_idx)
            else pos_items(train_mat.getrow(user_idx))
        )

        u_gt_neg = np.ones(test_set.num_items, dtype="int")
        u_gt_neg[test_pos_items + val_pos_items + train_pos_items] = 0

        item_indices = None if exclude_unknowns else np.arange(test_set.num_items)
        item_rank, item_scores = model.rank(user_idx, item_indices)

        try:
            seen_items = train_mat.getrow(user_idx).indices
        except:
            seen_items = []

        for i, mt in enumerate(metrics):
            # extend compute to add training positive for serendipity metric
            mt_score = mt.compute(
                gt_pos=u_gt_pos,
                gt_neg=u_gt_neg,
                pd_rank=item_rank,
                pd_scores=item_scores,
                seen_items=seen_items,
                reco_items=thresholded_items(gt_mat.getrow(user_idx)),
            )
            user_results[i][user_idx] = mt_score

    # avg results of ranking metrics
    for i, mt in enumerate(metrics):
        avg_results.append(sum(user_results[i].values()) / len(user_results[i]))

    return avg_results, user_results


class CombinedBaseMethod(BaseMethod):
    def __init__(self, **kwargs):
        print("initialising Combined Base")
        super().__init__(self, **kwargs)

    def _eval(self, model, test_set, val_set, user_based):
        metric_avg_results = OrderedDict()
        metric_user_results = OrderedDict()

        avg_results, user_results = rating_eval(
            model=model,
            metrics=self.rating_metrics,
            test_set=test_set,
            user_based=user_based,
            verbose=self.verbose,
        )
        for i, mt in enumerate(self.rating_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        avg_results, user_results = ranking_eval(
            model=model,
            metrics=self.ranking_metrics,
            train_set=self.train_set,
            test_set=test_set,
            val_set=val_set,
            rating_threshold=self.rating_threshold,
            exclude_unknowns=self.exclude_unknowns,
            verbose=self.verbose,
        )
        for i, mt in enumerate(self.ranking_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        return Result(model.name, metric_avg_results, metric_user_results)

    def evaluate(self, model, metrics, user_based, show_validation=True):
        if self.train_set is None:
            raise ValueError("train_set is required but None!")
        if self.test_set is None:
            raise ValueError("test_set is required but None!")

        self._reset()
        self._organize_metrics(metrics)

        ###########
        # FITTING #
        ###########
        if self.verbose:
            print("\n[{}] Training started!".format(model.name))

        start = time.time()
        model.fit(self.train_set, self.val_set)
        train_time = time.time() - start

        ##############
        # EVALUATION #
        ##############
        if self.verbose:
            print("\n[{}] Evaluation started!".format(model.name))

        start = time.time()
        test_result = self._eval(
            model=model,
            test_set=self.test_set,
            val_set=self.val_set,
            user_based=user_based,
        )
        test_time = time.time() - start
        test_result.metric_avg_results["Train (s)"] = train_time
        test_result.metric_avg_results["Test (s)"] = test_time

        val_result = None
        if show_validation and self.val_set is not None:
            start = time.time()
            val_result = self._eval(
                model=model, test_set=self.val_set, val_set=None, user_based=user_based
            )
            val_time = time.time() - start
            val_result.metric_avg_results["Time (s)"] = val_time

        return test_result, val_result

    @classmethod
    def from_splits(
        cls,
        train_data,
        test_data,
        val_data=None,
        fmt="UIR",
        rating_threshold=1.0,
        exclude_unknowns=False,
        seed=None,
        verbose=False,
        **kwargs
    ):
        print("creating from splits")
        method = cls(
            fmt=fmt,
            rating_threshold=rating_threshold,
            exclude_unknowns=exclude_unknowns,
            seed=seed,
            verbose=verbose,
            **kwargs
        )

        return method.build(
            train_data=train_data, test_data=test_data, val_data=val_data
        )
