

class PPOCfg:
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 4
        self.gamma = 0.99
        self.K_epochs = 30
        self.eps_clip = 0.05
        self.has_cont_act_sp = True
        self.action_std_init = 0.6
        self.lr_actor = 1e-4
        self.lr_critic = 1e-4
        self.device = 'cuda'
        self.max_timestep = 3_000_000_000
        self.horizon = 500
        self.update_every = self.horizon * 4
        self.action_std_decay_freq = int(2.5e5)  
        self.action_std_decay_rate = 0.05 
        self.min_action_std = 0.1
        self.log_freq = self.horizon *3
        self.save_after = self.horizon * 20
        self.checkpointFile = "/home/tribikram/RL/logs/checkpoints/PPO.pth"