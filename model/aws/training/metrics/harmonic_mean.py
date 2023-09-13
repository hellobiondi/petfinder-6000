import statistics
from cornac.metrics import RankingMetric


class HarmonicMean(RankingMetric):
    def __init__(self, k=-1, *args):
        RankingMetric.__init__(self, name="HarmonicMean", k=k)
        self.metrics = list(args)

    def compute(self, gt_pos, pd_rank, **kwargs):
        output = []

        for met in self.metrics:
            output.append(met.compute(gt_pos, pd_rank, **kwargs))

        return statistics.harmonic_mean(output)
