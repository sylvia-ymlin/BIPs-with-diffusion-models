class SMCStrategy:
    def compute_guidance(self, x_t, y, t, operator):
        # SMC粒子权重与扭曲函数
        # ...
        return None

    def update_step(self, x_t, score, guidance, t):
        # SMC特有的步进
        # ...
        return x_t
