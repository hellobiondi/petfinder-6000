from cornac.hyperopt import BaseSearch
from cornac.models import Recommender
from cornac.metrics import RatingMetric
from combined_eval_method import ranking_eval, rating_eval
import numpy as np


class NewBaseSearch(BaseSearch):
    def __init__(self, model, space, metric, eval_method, name="BaseSearch"):
        super().__init__(model, space, metric, eval_method, name)

    def fit(self, train_set, val_set=None):
        """Doing hyper-parameter search"""
        assert val_set is not None
        Recommender.fit(self, train_set, val_set)

        param_set = self._build_param_set()
        compare_op = np.greater if self.metric.higher_better else np.less
        self.best_score = -np.inf if self.metric.higher_better else np.inf
        self.best_model = None
        self.best_params = None

        # this can be parallelized if needed
        # keep it simple because multimodal algorithms are usually resource-hungry
        for params in param_set:
            if self.verbose:
                print("Evaluating: {}".format(params))

            model = self.model.clone(params).fit(train_set, val_set)

            if isinstance(self.metric, RatingMetric):
                score = rating_eval(model, [self.metric], val_set)[0][0]
            else:
                score = ranking_eval(
                    model,
                    [self.metric],
                    train_set,
                    val_set,
                    rating_threshold=self.eval_method.rating_threshold,
                    exclude_unknowns=self.eval_method.exclude_unknowns,
                    verbose=False,
                )[0][0]

            if compare_op(score, self.best_score):
                self.best_score = score
                self.best_model = model
                self.best_params = params

            del model

        if self.verbose:
            print("Best parameter settings: {}".format(self.best_params))
            print("{} = {:.4f}".format(self.metric.name, self.best_score))

        return self
