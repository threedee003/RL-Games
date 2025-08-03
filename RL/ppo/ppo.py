import torch
import copy
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from ppo.ppo_config import PPOCfg

class ReplayBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, device):
        super(ActorCritic, self).__init__()
        self.device = device

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self):
        self.cfg = PPOCfg()
        self.has_cont_sp = self.cfg.has_cont_act_sp
        if self.has_cont_sp:
            self.action_std = self.cfg.action_std_init
        self.gamma = self.cfg.gamma
        self.eps_clip = self.cfg.eps_clip
        self.K_epochs = self.cfg.K_epochs
        self.lr_actor = self.cfg.lr_actor
        self.lr_critic = self.cfg.lr_critic
        self.device = self.cfg.device

        self.buffer = ReplayBuffer()
        self.policy = ActorCritic(state_dim=self.cfg.state_dim,
                                      action_dim=self.cfg.action_dim,
                                      has_continuous_action_space=self.cfg.has_cont_act_sp,
                                      action_std_init=self.cfg.action_std_init,
                                      device=self.cfg.device
        )
        self.optimizer = torch.optim.Adam([
            {'params' : self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old = self.policy_old.to(self.cfg.device)
        self.policy = self.policy.to(self.cfg.device)
        self.load(checkpoint_file=self.cfg.checkpointFile)

        self.loss = torch.nn.MSELoss()
        



    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_cont_sp:
            with torch.no_grad():
                # state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()







    def update(self):
        rewards = []
        discounted_rew = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_rew = 0
            discounted_rew = reward + (self.gamma * discounted_rew)
            rewards.insert(0, discounted_rew)


        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim = 0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim = 0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim = 0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim = 0)).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()

        # minibatch
        for i in range(self.K_epochs):
            logProb, stateVal, distEntrop = self.policy.evaluate(old_states, old_actions)

            stateVal = torch.squeeze(stateVal)

            ratio = torch.exp(logProb - old_logprobs.detach())

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) *advantages
            clipped_loss = -torch.min(surr1, surr2)
            value_loss = self.loss(stateVal, rewards)
            # we also take an entropy loss for exploration
            total_loss = clipped_loss + 0.5 * value_loss - 0.01 * distEntrop


            self.optimizer.zero_grad()
            total_loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()



    def save(self, checkpoint_file):
        torch.save(self.policy_old.state_dict(), checkpoint_file)

    def load(self, checkpoint_file):
        self.policy_old.load_state_dict(torch.load(checkpoint_file, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_file, map_location=lambda storage, loc: storage))



