from new_base_search import NewBaseSearch
from cornac.hyperopt import get_rng


class NewRandomSearch(NewBaseSearch):
    def __init__(self, model, space, metric, eval_method, n_trails=10):
        super().__init__(
            model, space, metric, eval_method, name="RandomSearch_{}".format(model.name)
        )
        self.n_trails = n_trails

    def _build_param_set(self):
        """Generate searching points"""
        param_set = []
        keys = [d.name for d in self.space]
        rng = get_rng(self.model.seed)
        while len(param_set) < self.n_trails:
            params = [d._sample(rng) for d in self.space]
            param_set.append(dict(zip(keys, params)))
        return param_set
