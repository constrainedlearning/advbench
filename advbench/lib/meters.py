import time

try:
    import wandb
    wandb_log=True
except ImportError:
    wandb_log=False

from advbench.lib.plotting import plot_perturbed_wandb
from einops import rearrange

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, avg_mom=0.5):
        self.avg_mom = avg_mom
        self.reset()
        self.print = True

    def reset(self):
        self.val = 0
        self.avg = 0 # running average of whole epoch
        self.smooth_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.smooth_avg = val if self.count == 0 else self.avg*self.avg_mom + val*(1-self.avg_mom)
        self.avg = self.sum / self.count

class TimeMeter:
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.start = time.time()

    def batch_start(self):
        self.data_time.update(time.time() - self.start)

    def batch_end(self):
        self.batch_time.update(time.time() - self.start)
        self.start = time.time()
if wandb:
    class WBHistogramMeter:
        def __init__(self, name):
            self.print = False
            self.name = name

        def reset(self):
            pass

        def update(self, val):
            wandb.log({self.name: wandb.Histogram(val)})
    
    class WBDeltaMeter(WBHistogramMeter):
        def __init__(self, names = [], dims = 0):
            self.print = False
            self.dims = dims
            if isinstance(names, str):
                names = [f"{names} {i}" for i in range(dims)]
            self.meters = [WBHistogramMeter(name) for name in names]

        def reset(self):
            pass

        def update(self, vals):
            for i in range(self.dims):
                self.meters[i].update(vals[:,i])

    
    class WBLinePlotMeter():
        def __init__(self, name):
            self.print = False
            self.name = name
        def reset(self):
            pass
        def update(self, grid, vals):
            plot_perturbed_wandb(grid, vals, name=self.name)
    
    
    
    class WBDualMeter(WBHistogramMeter):
        def __init__(self, grid, names = "dual vs angle", locs = [(0, 0), (-1,-1)], log_every=100):
            self.print = False
            self.locs = []
            for loc in locs:
                self.locs.append((grid[:,1]==loc[0])&(grid[:,2]==loc[1]))
            if isinstance(names, str):
                names = [f"{names} {grid[i[0], 1], grid[i[0], 2]}" for i in locs]
            self.grid = grid
            self.meters = [WBLinePlotMeter(name) for name in names]
            self.log_every = log_every
            self.counter = 0

        def reset(self):
            self.counter=0

        def update(self, vals):
            if self.counter%self.log_every == 0:
                for i in range(len(self.locs)):
                    self.meters[i].update(self.grid[:, 0], vals[self.locs[i]])
            self.counter+=1

else:
    class WBHistogramMeter:
        def __init__(self, name):
            self.print = False

        def reset(self):
            pass

        def update(self, val):
            pass

    class WBDeltaMeter(WBHistogramMeter):
        def __init__(self,names = [], dims = 0):
            self.print = False
