class DCPSStrategy:
    def compute_guidance(self, x_t, y, t, operator):
        # 局部变分推断，混合高斯近似
        # ...
        return None

    def update_step(self, x_t, score, guidance, t):
        # DCPS特有的步进
        # ...
        return x_t
