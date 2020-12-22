from DRLEnv import FedEnv
from DDPG import Agent
import torch
import numpy as np
from collections import deque

if __name__ == '__main__':
    print (torch.cuda.is_available())
    epoches, print_every = 200, 100
    env = FedEnv(Client = 10, k = 2)  # env
    agent = Agent(state_size=8640, action_size=100, random_seed=2)  # agent
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, 200+1):
        X, Y = [], []  # x and y axis
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(epoches):
            action = agent.act(state)
            time, accuracy, next_state, reward = env.step(action,t)
            agent.step(state, action, reward, next_state)
            state = next_state
            score += reward
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        
        # save reward
        dataframe = pd.DataFrame(np.mean(scores_deque), columns=['X'])
        dataframe = pd.concat([dataframe, pd.DataFrame(i_episode,columns=['Y'])],axis=1)
        dataframe.to_csv("/home/ICDCS/Reward_data/reward.csv",mode='a',header = False,index=False,sep=',')
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))



