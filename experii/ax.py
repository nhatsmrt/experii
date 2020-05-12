from typing import Callable, Dict, Any, List
from ax.service.managed_loop import optimize
from verta import Client
from verta.client import ExperimentRun
from torch.nn import Module


__all__ = ['AxTuner']


class AxTuner:
    """
    Perform hyperparam optimization via ax-platform, logging hyperparams and result to ModelDB
    """
    def __init__(
            self, client: Client, evaluate_fn: Callable[[Dict[str, Any], Module, ExperimentRun], float],
            model_fn: Callable[[Dict[str, Any]], Module], specifications: List[Dict[str, Any]],
    ):
        self.client, self.evaluate_fn = client, evaluate_fn
        self.model_fn, self.specifications = model_fn, specifications
        self.run_id = 0

    def find_best_parameter(self, total_trials: int=20, arms_per_trial: int=1):
        best_parameters, best_values, experiment, model = optimize(
            parameters=self.specifications,
            evaluation_function=self.evaluate,
            total_trials=total_trials,
            arms_per_trial=arms_per_trial,
            minimize=False,
        )

        print(best_parameters)
        return best_parameters

    def evaluate(self, parameterization: Dict[str, Any]) -> float:
        """
        Wrap around evaluate_fn to support VertaAI's API.

        Note that ModelDBCB should be used to log metrics

        :param parameterization:
        :return: result of the experiment
        """
        run = self.client.set_experiment_run("Run %d" % self.run_id)
        run.log_hyperparameters(parameterization)
        self.run_id += 1
        return self.evaluate_fn(parameterization, self.model_fn(parameterization), run)

