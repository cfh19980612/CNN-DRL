from DRLEnv import FedEnv
from DDPG import Agent
from tqdm import tqdm, trange
import torch
import numpy as np
import pandas as pd
from collections import deque

if __name__ == '__main__':
    print (torch.cuda.is_available())
    epoches, print_every = 200, 100
    env = FedEnv(Client = 3, k = 2)  # env
    agent = Agent(state_size=5184, action_size=9, random_seed=2)  # agent
    scores_deque = deque(maxlen=print_every)
    scores = []
    episode = []
    for i_episode in range(1, 200+1):
        X, Y = [], []  # x and y axis for test_data
        start_time = 0
        state = env.reset()
        agent.reset()
        score = 0
        pbar = tqdm(range(2))
        for i in pbar:
            action = agent.act(state)
            time, accuracy, next_state, reward = env.step(action,i)
            
            # save accuracy
            start_time += time
            X.append(start_time)
            Y.append(accuracy)
            
            agent.step(state, action, reward, next_state)
            state = next_state
            score += reward
            
            # end?
            if accuracy > 0.825:
                env.save_acc(X,Y)
                break
            pbar.set_description("Epoch: %d Accuracy: %.3f" %(i, accuracy))
        scores_deque.append(score)
        scores.append(score)
        episode.append(i_episode)
        print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        
        # save reward
        dataframe = pd.DataFrame(episode, columns=['X'])
        dataframe = pd.concat([dataframe, pd.DataFrame(scores,columns=['Y'])],axis=1)
        dataframe.to_csv("/home/ICDCS/Reward_data/reward.csv",mode='w',header = False,index=False,sep=',')
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))



