import torch

class DiffusionSolver:
    def __init__(self, config, score_model, operator):
        self.score_model = score_model  # 预训练的扩散模型 (如 UNet)
        self.operator = operator        # 测量算子 A (如 Inpainting, Super-resolution)
        self.strategy = self._get_strategy(config.method) # DCPS, DPS, or SMC
        self.num_steps = config.num_steps
        self.device = config.device

    def _get_strategy(self, method):
        if method == 'DPS':
            from sampling_strategies.dps import DPSStrategy
            return DPSStrategy()
        elif method == 'DCPS':
            from sampling_strategies.dcps import DCPSStrategy
            return DCPSStrategy()
        elif method == 'SMC':
            from sampling_strategies.smc import SMCStrategy
            return SMCStrategy()
        else:
            raise NotImplementedError(f"Unknown method: {method}")

    def sample(self, y, shape):
        x_t = torch.randn(shape, device=self.device) # 初始化高斯噪声
        for t in reversed(range(self.num_steps)):
            score = self.score_model(x_t, t)
            guidance = self.strategy.compute_guidance(x_t, y, t, self.operator)
            x_t = self.strategy.update_step(x_t, score, guidance, t)
        return x_t
