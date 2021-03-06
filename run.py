from DRLEnv import FedEnv
from DDPG import Agent
from tqdm import tqdm, trange
import torch
import numpy as np
import pandas as pd
from collections import deque
import os

if __name__ == '__main__':
    print(torch.cuda.is_available())
    if os.path.exists('/home/cifar-gcn-drl/checkpoint/drl_cifar10_actor_local.pth') or\
        os.path.exists('/home/mnist-gcn-drl/checkpoint/drl_mnist_actor_local.pth'):
        checkpoint = True
    else:
        checkpoint = False

    dataset, net = 'FASHION-MNIST', 'MobileNet'

    if dataset == 'MNIST':
        target = 0.99
    elif dataset == 'CIFAR10':
        target = 0.95
    elif dataset == 'FASHION-MNIST':
        target = 0.93

    epoches, print_every = 200, 100
    env = FedEnv(Client = 10, k = 9, dataset = dataset, net = net)  # env
    agent = Agent(state_size=100, action_size=100, random_seed=2)  # agent
    scores_deque = deque(maxlen=print_every)
    scores = []
    episode = []

    # 是否重用训练的模型
    if checkpoint:
        if dataset == 'CIFAR10':
            agent.actor_local.load_state_dict(torch.load('/home/cifar-gcn-drl/checkpoint/drl_cifar10_actor_local.pth'))
            agent.actor_target.load_state_dict(torch.load('/home/cifar-gcn-drl/checkpoint/drl_cifar10_actor_target.pth'))
            agent.critic_local.load_state_dict(torch.load('/home/cifar-gcn-drl/checkpoint/drl_cifar10_critic_local.pth'))
            agent.critic_target.load_state_dict(torch.load('/home/cifar-gcn-drl/checkpoint/drl_cifar10_critic_target.pth'))
        elif dataset == 'MNIST':
            agent.actor_local.load_state_dict(torch.load('/home/mnist-gcn-drl/checkpoint/drl_mnist_actor_local.pth'))
            agent.actor_target.load_state_dict(torch.load('/home/mnist-gcn-drl/checkpoint/drl_mnist_actor_target.pth'))
            agent.critic_local.load_state_dict(torch.load('/home/mnist-gcn-drl/checkpoint/drl_mnist_critic_local.pth'))
            agent.critic_target.load_state_dict(torch.load('/home/mnist-gcn-drl/checkpoint/drl_mnist_critic_target.pth'))
        elif dataset == 'FASHION-MNIST':
            agent.actor_local.load_state_dict(torch.load('/home/fmnist-gcn-drl/checkpoint/drl_fmnist_actor_local.pth'))
            agent.actor_target.load_state_dict(torch.load('/home/fmnist-gcn-drl/checkpoint/drl_fmnist_actor_target.pth'))
            agent.critic_local.load_state_dict(torch.load('/home/fmnist-gcn-drl/checkpoint/drl_fmnist_critic_local.pth'))
            agent.critic_target.load_state_dict(torch.load('/home/fmnist-gcn-drl/checkpoint/drl_fmnist_critic_target.pth'))


    for i_episode in range(1, 200+1):
        X, Y, Z = [], [], []  # x and y axis for test_data
        start_time = 0
        # initialize pca ?
        if i_episode == 0:
            state = env.reset(Tag = True)
        else:
            state = env.reset(Tag = False)
        # initialize agent's noise
        agent.reset()
        score = 0


        reward_y = []
        episode_x = []
        pbar = tqdm(range(100))

        for i in pbar:
            action = agent.act(state)
            time, accuracy, test_loss, next_state, reward = env.step(action,i)
            # save accuracy
            start_time += time
            X.append(start_time)
            Y.append(accuracy)
            Z.append(test_loss)
            agent.step(state, action, reward, next_state)
            state = next_state
            score += reward

            # end?
            if accuracy >= target:
                break
            pbar.set_description("Epoch: %d Accuracy: %.3f Loss: %.3f Reward: %.3f Time: %.3f" %(i, accuracy, test_loss, reward, start_time))

        # save accuracy
        env.save_acc(X,Y,Z,i_episode)
        scores_deque.append(score)
        scores.append(score)
        episode.append(i_episode)
        print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, score), end="\n")

        # save reward
        if dataset == 'MNIST':
            location = '/home/mnist-gcn-drl/Test_data/mnist_reward.csv'
        elif dataset == 'CIFAR10':
            location = '/home/cifar-gcn-drl/Test_data/cifar10_reward.csv'
        elif dataset == 'FASHION-MNIST':
            location = '/home/fmnist-gcn-drl/Test_data/fmnist_reward.csv'
        dataframe = pd.DataFrame(episode, columns=['X'])
        dataframe = pd.concat([dataframe, pd.DataFrame(scores,columns=['Y'])],axis=1)
        dataframe.to_csv(location,mode='w',header = False,index=False,sep=',')

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    # save the drl models
    if dataset == 'CIFAR10':
        torch.save(agent.actor_local.state_dict(),'/home/cifar-gcn-drl/checkpoint/drl_cifar10_actor_local.pth')
        torch.save(agent.actor_target.state_dict(),'/home/cifar-gcn-drl/checkpoint/drl_cifar10_actor_target.pth')
        torch.save(agent.critic_local.state_dict(),'/home/cifar-gcn-drl/checkpoint/drl_cifar10_critic_local.pth')
        torch.save(agent.critic_target.state_dict(),'/home/cifar-gcn-drl/checkpoint/drl_cifar10_critic_target.pth')
    elif dataset == 'MNIST':
        torch.save(agent.actor_local.state_dict(),'/home/mnist-gcn-drl/checkpoint/drl_mnist_actor_local.pth')
        torch.save(agent.actor_target.state_dict(),'/home/mnist-gcn-drl/checkpoint/drl_mnist_actor_target.pth')
        torch.save(agent.critic_local.state_dict(),'/home/mnist-gcn-drl/checkpoint/drl_mnist_critic_local.pth')
        torch.save(agent.critic_target.state_dict(),'/home/mnist-gcn-drl/checkpoint/drl_mnist_critic_target.pth')
    elif dataset == 'FASHION-MNIST':
        torch.save(agent.actor_local.state_dict(),'/home/fmnist-gcn-drl/checkpoint/drl_fmnist_actor_local.pth')
        torch.save(agent.actor_target.state_dict(),'/home/fmnist-gcn-drl/checkpoint/drl_fmnist_actor_target.pth')
        torch.save(agent.critic_local.state_dict(),'/home/fmnist-gcn-drl/checkpoint/drl_fmnist_critic_local.pth')
        torch.save(agent.critic_target.state_dict(),'/home/fmnist-gcn-drl/checkpoint/drl_fmnist_critic_target.pth')


