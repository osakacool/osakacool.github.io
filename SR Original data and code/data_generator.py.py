from config import CFG

class SyntheticTimberHallDataset(Dataset):
    def __init__(self, duration_hours=24, seed=None):
        self.sr = CFG.SAMPLING_RATE_HZ           # ← 替换硬编码 100
        self.pga_range = (CFG.PGA_MIN_G, CFG.PGA_MAX_G)  # ← 替换 (0.05, 0.6)
        self.seed = seed if seed is not None else CFG.RANDOM_SEED
        # ...