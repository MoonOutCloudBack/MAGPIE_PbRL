import json
import os
import numpy as np
import pandas as pd
import wandb
import torch
from argparse import Namespace
from tensorboardX import SummaryWriter

from offpolicy.utils.rec_buffer import RecReplayBuffer, PrioritizedRecReplayBuffer
from offpolicy.utils.util import DecayThenFlatSchedule
from offpolicy.reward_model.reward_model import RewardModel
from offpolicy.utils.util import namespace_to_dict


class RecRunner(object):
    """Base class for training recurrent policies."""

    def __init__(self, config):
        """
        Base class for training recurrent policies.
        :param config: (dict) Config dictionary containing parameters for training.
        """
        self.args = config["args"]
        self.device = config["device"]
        self.q_learning = ["qmix","vdn"]

        self.share_policy = self.args.share_policy
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps
        self.use_wandb = self.args.use_wandb
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_popart = self.args.use_popart
        self.use_per = self.args.use_per
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval_episode = self.args.hard_update_interval_episode
        self.popart_update_interval_step = self.args.popart_update_interval_step
        self.actor_train_interval_step = self.args.actor_train_interval_step
        self.train_interval_episode = self.args.train_interval_episode
        self.train_interval = self.args.train_interval
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval

        self.total_env_steps = 0  # total environment interactions collected during training
        self.num_episodes_collected = 0  # total episodes collected during training
        self.total_train_steps = 0  # number of gradient updates performed
        self.last_train_episode = 0  # last episode after which a gradient update was performed
        self.last_eval_T = 0  # last episode after which a eval run was conducted
        self.last_save_T = 0  # last epsiode after which the models were saved
        self.last_log_T = 0 # last timestep after which information was logged
        self.last_hard_update_episode = 0 # last episode after which target policy was updated to equal live policy

        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("use_same_share_obs"):
            self.use_same_share_obs = config["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if config.__contains__("use_available_actions"):
            self.use_avail_acts = config["use_available_actions"]
        else:
            self.use_avail_acts = False

        if config.__contains__("buffer_length"):
            self.episode_length = config["buffer_length"]
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = config["buffer_length"]
            else:
                self.data_chunk_length = self.args.data_chunk_length
        else:
            self.episode_length = self.args.episode_length
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = self.args.episode_length
            else:
                self.data_chunk_length = self.args.data_chunk_length

        self.policy_info = config["policy_info"]
        self.policy_ids = sorted(list(self.policy_info.keys()))
        self.policy_mapping_fn = config["policy_mapping_fn"]

        self.num_agents = config["num_agents"]
        self.agent_ids = [i for i in range(self.num_agents)]

        self.env = config["env"]
        self.eval_env = config["eval_env"]
        # no parallel envs
        self.num_envs = 1

        # dir
        self.model_dir = self.args.model_dir
        self.run_dir = config["run_dir"]
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        self.config_path = str(self.run_dir / 'config.json')
        with open(self.config_path, 'w') as f:
            json.dump(namespace_to_dict(config['args']), f, indent=4)
        # log: always use tensorboard + csv; additionally wandb if enabled
        if self.use_wandb:
            log_run_dir = self.run_dir / wandb.run.id
        else:
            log_run_dir = self.run_dir
        if not os.path.exists(str(log_run_dir)):
            os.makedirs(str(log_run_dir))

        self.log_dir = str(log_run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)

        self.log_df = pd.DataFrame(columns=['total_env_steps'])
        self.log_df_dir = str(log_run_dir)

        self.save_dir = str(log_run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # initialize all the policies and organize the agents corresponding to each policy
        if self.algorithm_name == "rmatd3":
            from offpolicy.algorithms.r_matd3.algorithm.rMATD3Policy import R_MATD3Policy as Policy
            from offpolicy.algorithms.r_matd3.r_matd3 import R_MATD3 as TrainAlgo
        elif self.algorithm_name == "rmaddpg":
            assert self.actor_train_interval_step == 1, (
                "rmaddpg only supports actor_train_interval_step=1.")
            from offpolicy.algorithms.r_maddpg.algorithm.rMADDPGPolicy import R_MADDPGPolicy as Policy
            from offpolicy.algorithms.r_maddpg.r_maddpg import R_MADDPG as TrainAlgo
        elif self.algorithm_name == "rmasac":
            assert self.actor_train_interval_step == 1, (
                "rmasac only support actor_train_interval_step=1.")
            raise NotImplementedError("rmasac is not implemented.")
            # from offpolicy.algorithms.r_masac.algorithm.rMASACPolicy import R_MASACPolicy as Policy
            # from offpolicy.algorithms.r_masac.r_masac import R_MASAC as TrainAlgo
        elif self.algorithm_name == "qmix":
            from offpolicy.algorithms.qmix.algorithm.QMixPolicy import QMixPolicy as Policy
            from offpolicy.algorithms.qmix.qmix import QMix as TrainAlgo
        elif self.algorithm_name == "vdn":
            from offpolicy.algorithms.vdn.algorithm.VDNPolicy import VDNPolicy as Policy
            from offpolicy.algorithms.vdn.vdn import VDN as TrainAlgo
        else:
            raise NotImplementedError
        
        self.collecter = self.collect_rollout
        self.saver = self.save_q if self.algorithm_name in self.q_learning else self.save        
        self.restorer = self.restore_q if self.algorithm_name in self.q_learning else self.restore
        self.train = self.batch_train_q if self.algorithm_name in self.q_learning else self.batch_train

        self.policies = {p_id: Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids}

        if self.model_dir is not None:
            self.restorer()

        # initialize trainer class for updating policies
        self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.policy_mapping_fn,
                                 device=self.device, episode_length=self.episode_length)

        # map policy id to agent ids controlled by that policy
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}

        self.policy_obs_dim = {
            policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {
            policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {
            policy_id: self.policies[policy_id].central_obs_dim for policy_id in self.policy_ids}

        num_train_episodes = (self.num_env_steps / self.episode_length) / (self.train_interval_episode)
        self.beta_anneal = DecayThenFlatSchedule(
            self.per_beta_start, 1.0, num_train_episodes, decay="linear")

        if self.use_per:
            self.buffer = PrioritizedRecReplayBuffer(self.per_alpha,
                                                     self.policy_info,
                                                     self.policy_agents,
                                                     self.buffer_size,
                                                     self.episode_length,
                                                     self.use_same_share_obs,
                                                     self.use_avail_acts,
                                                     self.use_reward_normalization)
        else:
            self.buffer = RecReplayBuffer(self.policy_info,
                                          self.policy_agents,
                                          self.buffer_size,
                                          self.episode_length,
                                          self.use_same_share_obs,
                                          self.use_avail_acts,
                                          self.use_reward_normalization)
        
        # preference-based RL
        self.use_PbRL = self.args.use_PbRL
        self.num_interact = self.args.num_interact
        self.max_feedback = self.args.max_feedback
        self.reward_update = self.args.reward_update
        self.total_feedback, self.labeled_feedback = 0, 0
        self.interact_count = 0
        self.agg_fn = np.sum  # default mono aggregation: sum per-agent rewards

        # MARK: init reward model
        if self.use_PbRL:
            rm_type = getattr(self.args, 'reward_model_type', 'magpie')
            rm_kwargs = dict(
                lr=self.args.reward_model_lr,
                mb_size=self.args.reward_model_mb_size,
                size_segment=self.args.size_segment,
                teacher_gamma=self.args.gamma,
                save_query_dir=self.run_dir,
                L_reg_coef=self.args.L_reg_coef,
            )
            if rm_type == 'magpie':
                rm_cls = RewardModel
            else:
                raise ValueError(f"Unknown reward_model_type: {rm_type}")
            self.reward_model = rm_cls(
                self.policy_central_obs_dim['policy_0'],
                self.num_agents, self.policy_act_dim, self.device,
                **rm_kwargs,
            )
    
    def run(self):
        """Collect a training episode and perform appropriate training, saving, logging, and evaluation steps."""
        # MARK: run

        # collect data
        self.trainer.prep_rollout()  # trainer 调到 eval 模式
        collect_results = self.collecter(explore=True, training_episode=True, warmup=False)  # collector 是 separated_collect_rollout 或 shared_collect_rollout
        if isinstance(collect_results, tuple) and len(collect_results) == 2:
            env_info, rollout_info = collect_results
        else:
            env_info, rollout_info = collect_results, {}
        for k, v in env_info.items():
            self.env_infos[k].append(v)
        self.interact_count += self.episode_length * self.num_envs

        # MARK: use_PbRL
        if self.use_PbRL:
            # 把 rollout_info 存到 reward model 里
            self.reward_model.add_rollout_info(rollout_info)

            if self.total_feedback < self.max_feedback and self.interact_count >= self.num_interact:
                reward_train_infos = self.learn_reward()
                for k in reward_train_infos.keys():
                    if k not in self.env_infos.keys():
                        self.env_infos[k] = []
                    self.env_infos[k].append(reward_train_infos[k])
                self.interact_count = 0

            # relabel
            self.buffer.relabel_with_predictor(
                lambda obses, actions: self.reward_model.r_hat_batch(np.concatenate([obses, actions], axis=-1)),
                agg_fn=self.agg_fn,
            )

        # train
        if (self.num_episodes_collected - self.last_train_episode) / self.train_interval_episode >= 1 or self.last_train_episode == 0:
            self.train()
            self.total_train_steps += 1
            self.last_train_episode = self.num_episodes_collected

        # save
        if (self.total_env_steps - self.last_save_T) / self.save_interval >= 1:
            self.saver()
            self.last_save_T = self.total_env_steps

        # log
        if ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
            self.log()
            self.last_log_T = self.total_env_steps

        # eval
        if self.use_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1:
            self.eval()  # evaluate
            self.last_eval_T = self.total_env_steps

        return self.total_env_steps

    def learn_reward(self):
        # get queries
        labeled_queries = self.reward_model.uniform_sampling()
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.reward_update):
                train_acc, reward_loss, loss_BT_mean, loss_reg_mean = self.reward_model.train_reward()
                if train_acc > 0.97:
                    break

        print(f"Reward function is updated!! ACC: {train_acc} (epoch: {epoch})")
        test_acc = self.reward_model.test_reward_accuracy()
        return {
            'reward_train_acc': train_acc,  'reward_eval_acc': test_acc,
            'reward_loss': reward_loss, 'reward_loss_BT': loss_BT_mean, 'reward_loss_reg': loss_reg_mean, 
        }

    def warmup(self, num_warmup_episodes):
        """
        Fill replay buffer with enough episodes to begin training.

        :param: num_warmup_episodes (int): number of warmup episodes to collect.
        """
        self.trainer.prep_rollout()
        warmup_rewards = []
        print("warm up...")
        for _ in range((num_warmup_episodes // self.num_envs) + 1):
            env_info = self.collecter(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(env_info[0]['average_episode_rewards'])
        warmup_reward = np.mean(warmup_rewards)
        print("warmup average episode rewards: {}".format(warmup_reward))

    def batch_train(self):
        """Do a gradient update for all policies."""
        self.trainer.prep_training()

        # gradient updates
        self.train_infos = []
        update_actor = False
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            update_method = self.trainer.shared_train_policy_on_batch if self.use_same_share_obs else self.trainer.cent_train_policy_on_batch
            
            train_info, new_priorities, idxes = update_method(p_id, sample)
            update_actor = train_info['update_actor']

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update and update_actor:
            for pid in self.policy_ids:
                self.policies[pid].soft_target_updates()
        else:
            if ((self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode) >= 1:
                for pid in self.policy_ids:
                    self.policies[pid].hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def batch_train_q(self):
        """Do a q-learning update to policy (used for QMix and VDN)."""
        self.trainer.prep_training()  # trainer 调到 train 模式
        # gradient updates
        self.train_infos = []

        for p_id in self.policy_ids:
            if self.use_per:  # prioritized experience replay
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample)

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def save(self):
        """Save all policies to the path specified by the config."""
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(),
                       critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(),
                       actor_save_path + '/actor.pt')

        if self.use_PbRL:
            self.reward_model.save(self.save_dir, self.total_env_steps)

    def save_q(self):
        """Save all policies to the path specified by the config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            policy_Q = self.policies[pid].q_network
            p_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            torch.save(policy_Q.state_dict(), p_save_path + '/q_network.pt')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.trainer.mixer.state_dict(),
                   self.save_dir + '/mixer.pt')

        if self.use_PbRL:
            self.reward_model.save(self.save_dir, self.total_env_steps)

    def restore(self):
        """Load policies policies from pretrained models specified by path in config."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')

            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    def restore_q(self):
        """Load policies policies from pretrained models specified by path in config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_q_state_dict = torch.load(path + '/q_network.pt')           
            self.policies[pid].q_network.load_state_dict(policy_q_state_dict)
            
        policy_mixer_state_dict = torch.load(str(self.model_dir) + '/mixer.pt')
        self.trainer.mixer.load_state_dict(policy_mixer_state_dict)

    def log(self):
        """Log relevent training and rollout colleciton information.."""
        raise NotImplementedError

    def log_clear(self):
        """Clear logging variables so they do not contain stale information."""
        raise NotImplementedError

    def log_env(self, env_info, suffix=None):
        """
        Log information related to the environment.
        :param env_info: (dict) contains logging information related to the environment.
        :param suffix: (str) optional string to add to end of keys in env_info when logging. 
        """
        for k, v in env_info.items():
            if len(v) > 0:
                v = np.mean(v)
                suffix_k = k if suffix is None else suffix + k 
                print(suffix_k + " is " + str(v))
                self.writter.add_scalars(suffix_k, {suffix_k: v}, self.total_env_steps)
                if self.use_wandb:
                    wandb.log({suffix_k: v}, step=self.total_env_steps)

                # 接下来存 dataframe，存在 self.log_df 里
                # 我希望 log_df 里有一个列是 total_env_steps，但 suffix_k 这个列不一定有
                # 我希望 log_df 中 total_env_steps = total_env_steps 对应的这一行，suffix_k 的值是 v
                # 如果 total_env_steps 这一列里没有 total_env_steps = total_env_steps 这一行，那就新建一行
                if suffix_k not in self.log_df.columns:
                    self.log_df[suffix_k] = np.nan
                if self.total_env_steps not in self.log_df.total_env_steps.values:
                    new_row = pd.DataFrame({'total_env_steps': [self.total_env_steps]})
                    self.log_df = pd.concat([self.log_df, new_row], ignore_index=True)
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                self.log_df.loc[self.log_df.total_env_steps == self.total_env_steps, suffix_k] = v
                # 这里的 self.log_df 是一个 dataframe，里面有一列是 total_env_steps, 还有一列是 suffix_k

    def log_train(self, policy_id, train_info):
        """
        Log information related to training.
        :param policy_id: (str) policy id corresponding to the information contained in train_info.
        :param train_info: (dict) contains logging information related to training.
        """
        for k, v in train_info.items():
            policy_k = str(policy_id) + '/' + k
            self.writter.add_scalars(policy_k, {policy_k: v}, self.total_env_steps)
            if self.use_wandb:
                wandb.log({policy_k: v}, step=self.total_env_steps)

            # 接下来存 dataframe，存在 self.log_df 里
            if policy_k not in self.log_df.columns:
                self.log_df[policy_k] = np.nan
            if self.total_env_steps not in self.log_df.total_env_steps.values:
                new_row = pd.DataFrame({'total_env_steps': [self.total_env_steps]})
                self.log_df = pd.concat([self.log_df, new_row], ignore_index=True)
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            self.log_df.loc[self.log_df.total_env_steps == self.total_env_steps, policy_k] = v

    def collect_rollout(self):
        """Collect a rollout and store it in the buffer."""
        raise NotImplementedError