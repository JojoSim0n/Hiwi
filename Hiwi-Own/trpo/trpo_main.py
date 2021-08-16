import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from trpo.models import *
from trpo.replay_memory import Memory
from trpo.running_state import ZFilter
from torch.autograd import Variable
from trpo.trpo import trpo_step
from trpo.utils import *
from discrim.discrim_net import DiscrimNetGAIL

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
class Trpo():
    
    # change to set parameters
    # parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    # parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
    #                     help='discount factor (default: 0.995)')
    # parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
    #                     help='name of the environment to run')
    # parser.add_argument('--tau', type=float, default=0.97, metavar='G',
    #                     help='gae (default: 0.97)')
    # parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
    #                     help='l2 regularization regression (default: 1e-3)')
    # parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
    #                     help='max kl value (default: 1e-2)')
    # parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
    #                     help='damping (default: 1e-1)')
    # parser.add_argument('--seed', type=int, default=543, metavar='N',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--render', action='store_true',
    #                     help='render the environment')
    # parser.add_argument('--log-interval', type=int, default=1, metavar='N',
    #                     help='interval between training status logs (default: 10)')
    # args = parser.parse_args()
    def __init__(self, disc: DiscrimNetGAIL):


        self.gamma = 0.995
        self.env_name = "CartPole-v0"
        self.tau = 0.97
        self.gae = 0.97
        self.l2_reg = 1e-3
        self.max_kl = 1e-2
        self.damping = 1e-1
        self.seed = 543
        self.batch_size = 15000
        self.render = False
        self.log_interval = 1


        self.env = gym.make(self.env_name)

        self.num_inputs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]

        self.env.seed(self.seed)
        torch.manual_seed(self.seed)

        self.policy_net = Policy(self.num_inputs, self.num_actions)
        self.value_net = Value(self.num_inputs)

        self.disc = disc


    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, _, action_std = self.policy_net(Variable(state))
        action = torch.normal(action_mean, action_std)
        return action

    def update_params(self, batch):
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(np.concatenate(batch.action, 0))
        states = torch.Tensor(batch.state)
        values = self.value_net(Variable(states))

        returns = torch.Tensor(actions.size(0),1)
        deltas = torch.Tensor(actions.size(0),1)
        advantages = torch.Tensor(actions.size(0),1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + self.gamma * self.tau * prev_advantage * masks[i]

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        targets = Variable(returns)

        # Original code uses the same LBFGS to optimize the value loss
        def get_value_loss(flat_params):
            set_flat_params_to(self.value_net, torch.Tensor(flat_params))
            for param in self.value_net.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = self.value_net(Variable(states))

            value_loss = (values_ - targets).pow(2).mean()

            # weight decay
            for param in self.value_net.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            value_loss.backward()
            return (value_loss.data.double().numpy(), get_flat_grad_from(self.value_net).data.double().numpy())

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(self.value_net).double().numpy(), maxiter=25)
        set_flat_params_to(self.value_net, torch.Tensor(flat_params))

        advantages = (advantages - advantages.mean()) / advantages.std()

        action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
            else:
                action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                    
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()


        def get_kl():
            mean1, log_std1, std1 = self.policy_net(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        trpo_step(self.policy_net, get_loss, get_kl, self.max_kl, self.damping)

    def train(self):
        running_state = ZFilter((self.num_inputs,), clip=5)
        running_reward = ZFilter((1,), demean=False, clip=10)

        for i_episode in count(1):
            memory = Memory()

            num_steps = 0
            reward_batch = 0
            num_episodes = 0
            while num_steps < self.batch_size:
                state = self.env.reset()
                state = running_state(state)

                reward_sum = 0
                for t in range(10000): # Don't infinite loop while learning
                    action = self.select_action(state)
                    action = action.data[0].numpy()
                    next_state, reward, done, _ = self.env.step(action)
                    #reward_sum += reward

                    reward_sum += self.disc.reward_train(state, action, next_state, done)

                    next_state = running_state(next_state)

                    mask = 1
                    if done:
                        mask = 0

                    memory.push(state, np.array([action]), mask, next_state, reward)

                    if self.render:
                        self.env.render()
                    if done:
                        break

                    state = next_state
                num_steps += (t-1)
                num_episodes += 1
                reward_batch += reward_sum

            reward_batch /= num_episodes
            batch = memory.sample()
            self.update_params(batch)

            if i_episode % self.log_interval == 0:
                print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                    i_episode, reward_sum, reward_batch))