
import torch
import abc

class SamplingStrategy(abc.ABC):
    @abc.abstractmethod
    def update_step(self, x_t, noise_pred, t, y, operator, model_params):
        pass
