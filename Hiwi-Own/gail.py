from typing import Mapping, Optional, Type
import gym
import logging
import os
from typing import Dict, Iterable, Mapping, Optional, Type, Union
import tqdm

import torch as th
import torch.utils.tensorboard as thboard

from trpo.trpo_main import *
from discrim.discrim_net import DiscrimNetGAIL
from discrim.common import *
from data.util import endless_iter


import ray._private.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from data.types import *

class AdversarialTrainer():
    # implement step by step the gail algorithm by Ho

    # step one: Input: Expert trajectories E  E, initial policy and discriminator parameters 0;w0
    def __init__(self, env_name,
     init_tensorboard: bool = False,
      init_tensorboard_graph: bool = False,
       disc_opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
       disc_opt_kwargs: Optional[Mapping] = None,
       n_disc_updates_per_round: int = 2,
       *,
       expert_data: Union[Iterable[Mapping], Transitions], 
       expert_batch_size: int,
       ):
        
        env = gym.make(env_name)

        # initialize disc 
        self.discrim = DiscrimNetGAIL(env.observation_space, env.action_space)
        # initializ generator algo
        self.gen = Trpo(self.discrim)
        
        # Create graph for optimising/recording stats on discriminator
        self._disc_opt_cls = disc_opt_cls
        self._disc_opt_kwargs = disc_opt_kwargs or {}
        self._disc_opt = self._disc_opt_cls(
            self.discrim.parameters(), **self._disc_opt_kwargs)

        self._init_tensorboard = init_tensorboard
        self._init_tensorboard_graph = init_tensorboard_graph

        self._log_dir = "output/"


        if self._init_tensorboard:
            logging.info("building summary directory at " + self._log_dir)
            summary_dir = os.path.join(self._log_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            self._summary_writer = thboard.SummaryWriter(summary_dir)

        self.global_steps = 0
        self.disc_step = 0
        self.n_disc_updates_per_round = n_disc_updates_per_round

        # keep things safe
        if expert_batch_size <= 0:
            raise ValueError(f"expert_batch_size={expert_batch_size} must be positive.")
        self.expert_batch_size = expert_batch_size

        self.expert_batch_size = expert_batch_size
        if isinstance(expert_data, Transitions):
            if len(expert_data) < expert_batch_size:
                raise ValueError(
                    "Provided Transitions instance as `expert_data` argument but "
                    "len(expert_data) < expert_batch_size. "
                    f"({len(expert_data)} < {expert_batch_size})."
                )

            self.expert_data_loader = th_data.DataLoader(
                expert_data,
                batch_size=expert_batch_size,
                collate_fn=transitions_collate_fn,
                shuffle=True,
                drop_last=True,
            )
        else:
            self.expert_data_loader = expert_data
        self._endless_expert_iterator = endless_iter(self.expert_data_loader)




    # generate gen sample batch
    def store_gen_samples(self):
        batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
        writer = JsonWriter(
        os.path.join(ray._private.utils.get_user_temp_dir(), "demo-out"))

        # You normally wouldn't want to manually create sample batches if a
        # simulator is available, but let's do it anyways for example purposes:
        env = gym.make("CartPole-v0")

        # RLlib uses preprocessors to implement transforms such as one-hot encoding
        # and flattening of tuple and dict observations. For CartPole a no-op
        # preprocessor is used, but this may be relevant for more complex envs.
        prep = get_preprocessor(env.observation_space)(env.observation_space)
        print("The preprocessor is", prep)

        for eps_id in range(100):
            obs = env.reset()
            prev_action = np.zeros_like(env.action_space.sample())
            prev_reward = 0
            done = False
            t = 0
            while not done:
                action = self.gen.select_action(obs)
                new_obs, rew, done, info = env.step(action)
                action_mean, _, action_std = self.gen.policy_net(Variable(obs))
                batch_builder.add_values(
                    t=t,
                    eps_id=eps_id,
                    agent_index=0,
                    obs=prep.transform(obs),
                    actions=action,
                    action_prob=action_mean,  # put the true action probability here
                    action_logp=action_std,
                    rewards=rew,
                    prev_actions=prev_action,
                    prev_rewards=prev_reward,
                    dones=done,
                    infos=info,
                    new_obs=prep.transform(new_obs))
                obs = new_obs
                prev_action = action
                prev_reward = rew
                t += 1
            writer.write(batch_builder.build_and_reset())
            










    # step 4: update discrim parameters
    def train_disc(self, expert_samples: Optional[Mapping], gen_samples: Optional[Mapping]):

        """Perform a single discriminator update, optionally using provided samples.
        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.expert_batch_size` samples.
                If this argument is not provided, then `self.expert_batch_size` expert
                samples from `self.expert_data_loader` are used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.expert_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.
        Returns:
           dict: Statistics for discriminator (e.g. loss, accuracy).
        """
        
        # optionally write TB summaries for collected ops
        write_summaries = self._init_tensorboard and self._global_step % 20 == 0

        # compute loss
        batch = self.make_disc_train_batch(
            gen_samples=gen_samples, expert_samples=expert_samples
        )
        disc_logits = self.discrim.logits_gen_is_high(
            batch["state"],
            batch["action"],
            batch["next_state"],
            batch["done"],
            batch["log_policy_act_prob"],
        )
        loss = self.discrim.disc_loss(disc_logits, batch["labels_gen_is_one"])

        # do gradient step
        self._disc_opt.zero_grad()
        loss.backward()
        self._disc_opt.step()
        self._disc_step += 1

        # compute/write stats and TensorBoard data
        with th.no_grad():
            train_stats = compute_train_stats(
                disc_logits,
                batch["labels_gen_is_one"],
                loss,
            )
        if write_summaries:
            self._summary_writer.add_histogram("disc_logits", disc_logits.detach())

        return train_stats
        

    def make_disc_train_batch(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> dict:
        """Build and return training batch for the next discriminator update.
        Args:
          gen_samples: Same as in `train_disc_step`.
          expert_samples: Same as in `train_disc_step`.
        """
        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first."
                )
            gen_samples = self._gen_replay_buffer.sample(self.expert_batch_size)
            gen_samples = dataclass_quick_asdict(gen_samples)

        n_gen = len(gen_samples["obs"])
        n_expert = len(expert_samples["obs"])
        if not (n_gen == n_expert == self.expert_batch_size):
            raise ValueError(
                "Need to have exactly self.expert_batch_size number of expert and "
                "generator samples, each. "
                f"(n_gen={n_gen} n_expert={n_expert} "
                f"expert_batch_size={self.expert_batch_size})"
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        n_samples = n_expert + n_gen
        assert n_expert == len(expert_samples["acts"])
        assert n_expert == len(expert_samples["next_obs"])
        assert n_gen == len(gen_samples["acts"])
        assert n_gen == len(gen_samples["next_obs"])

        # Concatenate rollouts, and label each row as expert or generator.
        obs = np.concatenate([expert_samples["obs"], gen_samples["obs"]])
        acts = np.concatenate([expert_samples["acts"], gen_samples["acts"]])
        next_obs = np.concatenate([expert_samples["next_obs"], gen_samples["next_obs"]])
        dones = np.concatenate([expert_samples["dones"], gen_samples["dones"]])
        labels_gen_is_one = np.concatenate(
            [np.zeros(n_expert, dtype=int), np.ones(n_gen, dtype=int)]
        )
        
        # Calculate generator-policy log probabilities.
        with th.no_grad():
            obs_th = th.as_tensor(obs, device=self.gen_algo.device)
            acts_th = th.as_tensor(acts, device=self.gen_algo.device)
            _, log_act_prob_th, _ = self.gen.policy_net(Variable(obs))
            log_act_prob = log_act_prob_th.detach().cpu().numpy()
            del obs_th, acts_th, log_act_prob_th  # unneeded
        assert len(log_act_prob) == n_samples
        log_act_prob = log_act_prob.reshape((n_samples,))

        batch_dict = {
            "state": self._torchify_with_space(obs, self.discrim.observation_space),
            "action": self._torchify_with_space(acts, self.discrim.action_space),
            "next_state": self._torchify_with_space(
                next_obs, self.discrim.observation_space
            ),
            "done": self._torchify_array(dones),
            "labels_gen_is_one": self._torchify_array(labels_gen_is_one),
            "log_policy_act_prob": self._torchify_array(log_act_prob),
        }

        return batch_dict




    # step 5: update generator
    def train_gen(self):
        # single generator update step
        self.gen.train()
        
       



    
    def train_gail(self, total_timesteps: int):
        # alternating between disc and gen update
        n_rounds = total_timesteps // self.gen_batch_size
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_batch_size} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
        )
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.train_gen(self.gen_batch_size)
            for _ in range(self.n_disc_updates_per_round):
                self.train_disc()
            
    def _torchify_array(self, ndarray: np.ndarray, **kwargs) -> th.Tensor:
        return th.as_tensor(ndarray, device=self.discrim.device(), **kwargs)    

    def _torchify_with_space(
        self, ndarray: np.ndarray, space: gym.Space, **kwargs
    ) -> th.Tensor:
        tensor = th.as_tensor(ndarray, device=self.discrim.device(), **kwargs)
        preprocessed = preprocessing.preprocess_obs(
            tensor,
            space,
            
            normalize_images=self.discrim.scale,
        )
        return preprocessed


class GAIL(AdversarialTrainer):
    def __init__(
        self,
        env_name,
        expert_data: Union[Iterable[Mapping], Transitions],
        expert_batch_size: int,
        *,
        discrim_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Generative Adversarial Imitation Learning.
        Most parameters are described in and passed to `AdversarialTrainer.__init__`.
        Additional parameters that `GAIL` adds on top of its superclass initializer are
        as follows:
        Args:
            discrim_kwargs: Optional keyword arguments to use while constructing the
                DiscrimNetGAIL.
        """
        env = gym.make(env_name)
        discrim_kwargs = discrim_kwargs or {}
        discrim = DiscrimNetGAIL(
            env.observation_space, env.action_space, **discrim_kwargs
        )
        super().__init__(
            discrim, expert_data, expert_batch_size, **kwargs
        )