from verta.client import ExperimentRun
from nntoolbox.callbacks import Callback
from nntoolbox.metrics import Metric
from verta.integrations.torch import verta_hook
from typing import Dict


__all__ = ['ModelDBCB']


class ModelDBCB(Callback):
    """
    Integration between VertaAI's ModelDB and nn-toolbox.

    Log the model's architecture and final (best) validation metrics
    """
    def __init__(self, run: ExperimentRun, metrics: Dict[str, Metric]):
        super().__init__()
        self.order = 9999
        self.metrics = metrics
        self.run = run

    def on_train_begin(self):
        # automatically log model's topology
        self.learner._model.register_forward_hook(verta_hook(self.run))

    def on_train_end(self):
        # Log the best value of metrics
        final_metrics = {}
        for key in self.metrics:
            final_metrics[key] = self.metrics[key].get_best()
        self.run.log_metrics(final_metrics)
