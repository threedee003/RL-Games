import argparse
import torch
import torch.nn as nn
import datetime

# from torch.utils.tensorboard import SummaryWriter
from game.gamev2 import GameEnv
from ppo.ppo_config import PPOCfg
from ppo.ppo import PPO

path = "~/RL/logs"


def train():
    # writer = SummaryWriter(path+f'/runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_PPO')
    checkpoint_path = "/home/tribikram/RL/logs/checkpoints"
    ppo = PPO()
    cfg = PPOCfg()
    running_reward = 0
    running_episode = 0
    log_running_reward = 0
    log_running_episode = 0


    timestep = 0
    i_episode = 0
    env = GameEnv()



    while env.running:
        state = env.reset()
        current_ep_reward = 0

        for t in range(1, cfg.horizon + 1):
            state = torch.FloatTensor(state).to(cfg.device)[None]
            action = ppo.select_action(state)
            state, reward, done, _ = env.step(action)
            env.render()

            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(done)

            timestep += 1
            current_ep_reward += reward

            if timestep % cfg.update_every == 0:
                ppo.update()
            
            if cfg.has_cont_act_sp and timestep % cfg.action_std_decay_freq == 0:
                ppo.decay_action_std(cfg.action_std_decay_rate, cfg.min_action_std)

            if timestep % cfg.log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episode
                log_avg_reward = round(log_avg_reward, 4)

                # writer.add_scalar('avg_reward', log_avg_reward, timestep)
                log_running_episode = 0
                log_running_reward = 0

                print(f"Episode : {i_episode} | Timestep : {timestep} | Avg Reward: {log_avg_reward}")

            if timestep % cfg.save_after == 0:
                print("Saving model checkpt")
                ppo.save(checkpoint_path + f"/PPO.pth")


            if done:
                break
            
            if timestep >= cfg.max_timestep:
                print()


        log_running_reward += current_ep_reward
        log_running_episode += 1
        i_episode += 1


    env.close()

    


