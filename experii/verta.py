from verta.client import ExperimentRun
from nntoolbox.callbacks import Callback
from nntoolbox.metrics import Metric
from verta.integrations.torch import verta_hook
from typing import Dict, Any


__all__ = ['ModelDBCB']


class ModelDBCB(Callback):
    """
    Integration between VertaAI's ModelDB and nn-toolbox.

    Log the model's architecture and final (best) validation metrics. Also checkpointing model by epoch.
    """
    def __init__(
            self, run: ExperimentRun, metrics: Dict[str, Metric], monitor: str='loss',
            save_best_only: bool=True, mode: str='min', period: int=1
    ):
        super().__init__()
        assert monitor in metrics

        self.order = 9999
        self.metrics = metrics
        self.run = run

        self.monitor = monitor
        self.period = period
        self.mode = mode
        self.save_best_only = save_best_only

    def on_train_begin(self):
        # automatically log model's topology
        self.hook = self.learner._model.register_forward_hook(verta_hook(self.run))

    def on_batch_end(self, logs: Dict[str, Any]):
        if logs["iter_cnt"] == 0:
            # Deregister verta hook to avoid slow downs:
            self.hook.remove()

    def on_epoch_end(self, logs: Dict[str, Any]) -> bool:
        if self.save_best_only:
            epoch_metrics = logs['epoch_metrics']
            best_so_far = self.metrics[self.monitor].get_best()

            if self.mode == "min":
                if epoch_metrics[self.monitor] <= best_so_far:
                    self.run.log_artifact(key="weights_%d" % logs["epoch"], artifact=self.learner._model.state_dict())
            else:
                if epoch_metrics[self.monitor] >= best_so_far:
                    self.run.log_artifact(key="weights_%d" % logs["epoch"], artifact=self.learner._model.state_dict())
        else:
            self.run.log_artifact(key="weights_%d" % logs["epoch"], artifact=self.learner._model.state_dict())

        return False

    def on_train_end(self):
        # Log the best value of metrics
        final_metrics = {}
        for key in self.metrics:
            final_metrics[key] = self.metrics[key].get_best()
        self.run.log_metrics(final_metrics)
