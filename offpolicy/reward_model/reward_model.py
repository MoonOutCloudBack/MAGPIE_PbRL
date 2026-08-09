import time
from typing import List

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from offpolicy.utils.util import NotVerified


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net


def KCenterGreedy(obs, full_obs, num_new_sample, device):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs, device)
        max_index = torch.argmax(dist)
        max_index = max_index.item()

        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index + 1:]

        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs,
            obs[selected_index]],
            axis=0)
    return selected_index


def compute_smallest_dist(obs, full_obs, device):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)

        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)


class RewardModel:
    def __init__(self, ds, num_agents, da_all, device,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1,
                 save_query_dir='./', 
                 max_size=500, activation='tanh', capacity=5e5,
                 large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1,
                 teacher_eps_mistake=0,
                 teacher_eps_skip=0,
                 teacher_eps_equal=0,
                 # surf
                 data_aug_ratio=0,  # original 20
                 data_aug_window=5,
                 threshold_u=0.99,  # surf use 0.99
                 lambda_u=1,
                 # L_reg_coef
                 L_reg_coef=1e-4,
                 ):

        # train data is trajectories, must process to sa and s..
        self.ds = int(ds)  # shared_obs
        self.num_agents = num_agents
        self.da_all = da_all  # single agent action
        self.da = int(np.sum(list(da_all.values())))  # all agent action
        self.dr = num_agents
        self.de = ensemble_size
        self.lr = lr
        self.ensemble: List[nn.Module] = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = int(size_segment)  # segment length
        self.device = device

        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, self.size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, self.size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, self.dr), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.UCELoss = nn.CrossEntropyLoss(reduction='none')
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch

        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0

        self.label_margin = label_margin
        self.label_target = 1 - 2 * self.label_margin

        # surf
        self.u_buffer_seg1 = np.empty((self.capacity, self.size_segment, self.ds + self.da), dtype=np.float32)
        self.u_buffer_seg2 = np.empty((self.capacity, self.size_segment, self.ds + self.da), dtype=np.float32)
        self.u_buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.u_buffer_index = 0
        self.u_buffer_full = False

        self.data_aug_ratio = data_aug_ratio
        self.data_aug_window = data_aug_window
        self.threshold_u = threshold_u
        self.lambda_u = lambda_u

        # L_reg_coef
        self.L_reg_coef = L_reg_coef
        
        # for recording
        self.save_query_dir = save_query_dir
        self.df = pd.DataFrame(columns=[
            'step', 
            'en0', 'start0', 'seg_R0', 
            'en1', 'start1', 'seg_R1', 
            'label', 
        ])
        self.df_count = 1  # first queries are obtained in step 20000, rather than 0
        self.episode_counter = 0
        self.episode_num_list = []

    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size * new_frac)

    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)

    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip

    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal

    def construct_ensemble(self):
        for ens_idx in range(self.de):
            ensemble_model = []
            for agent_idx in range(self.num_agents):
                model = nn.Sequential(*gen_net(in_size=self.ds + self.da,
                        out_size=1, H=256, n_layers=3,
                        activation=self.activation)).float().to(self.device)
                self.paramlst.extend(model.parameters())
                ensemble_model.append(model)
            self.ensemble.append(ensemble_model)
        
        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def add_data(self, obs, act, rew, done):
        # MARK: add_data
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew

        flat_input = sa_t.reshape(1, self.da + self.ds)  # [s, a]
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, self.num_agents)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            self.episode_num_list.append(self.episode_counter)
            self.episode_counter += 1
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
                self.episode_num_list = self.episode_num_list[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])

    def add_rollout_info(self, rollout_info):
        # MARK: add_rollout_info
        '''
        rollout_info = {
            'episode_obs': episode_obs,
            'episode_share_obs': episode_share_obs, # {pid: (episode_len + 1, 1, 1, self.ds)}
            'episode_acts': episode_acts,  # {pid: (episode_length, 1, 1, self.da_single)}
            'episode_rewards': episode_rewards,
            'episode_personalized_rewards': episode_personalized_rewards,  # {pid: (episode_length, 1, 1, 1)}
            'episode_dones': episode_dones,
            'episode_dones_env': episode_dones_env,  # {pid: (episode_length, 1, 1)}
            'episode_avail_acts': episode_avail_acts
        }
        '''
        episode_share_obs = rollout_info['episode_share_obs']['policy_0'][:-1, 0, 0, :]  # (episode_length, ds)
        episode_acts = np.concatenate(
                [rollout_info['episode_acts']['policy_' + str(agent_idx)][:, 0, 0, :] \
                for agent_idx in range(self.num_agents)], axis=-1)  # (episode_length, num_agents * da_single)
        episode_personalized_rewards = np.concatenate(
                [rollout_info['episode_personalized_rewards']['policy_' + str(agent_idx)][:, 0, 0, :]
                for agent_idx in range(self.num_agents)], axis=-1)  # (episode_length, num_agents)
        episode_dones_env = rollout_info['episode_dones_env']['policy_0'][:, 0, 0]  # (episode_length,)

        assert episode_share_obs.shape[0] == episode_acts.shape[0] == \
                episode_personalized_rewards.shape[0] == episode_dones_env.shape[0]
        episode_len = episode_share_obs.shape[0]
        for i in range(episode_len):
            self.add_data(episode_share_obs[i], episode_acts[i], 
                          episode_personalized_rewards[i], episode_dones_env[i])

    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])

    @NotVerified
    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)

        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        ''' x: (batch_size, segment_len, ds + da), results: (batch_size, dr) '''
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)  # (batch_size, segment_len, dr)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1).unsqueeze(-1)  # (batch_size, dr, 1)
            r_hat2 = r_hat2.sum(axis=1).unsqueeze(-1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # (batch_size, dr, 2)

        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:, 0]

    def p_hat_entropy(self, x_1, x_2, member=-1):
        ''' x: (batch_size, segment_len, ds + da), results: (batch_size, dr) '''
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)  # (batch_size, segment_len, dr)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1).unsqueeze(-1)  # (batch_size, dr, 1)
            r_hat2 = r_hat2.sum(axis=1).unsqueeze(-1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        ''' x: (*, ds + da), results: (*, dr) '''
        results = []
        for rew_model in self.ensemble[member]:
            results.append(rew_model(torch.from_numpy(x).float().to(self.device)))  # (*, 1)
        results = torch.cat(results, dim=-1)  # (*, dr)
        return results

    def r_hat(self, x):
        ''' x: (ds + da, ), results: (dr, ) '''
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())  # (dr, )
        r_hats = np.array(r_hats)  # (de, dr)
        return np.mean(r_hats, axis=0)  # (dr, )

    # MARK: r_hat_batch
    def r_hat_batch(self, x):
        ''' x: (*, ds + da), results: (*, dr) '''
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())  # (*, dr)
        r_hats = np.array(r_hats)  # (de, *, dr)

        return np.mean(r_hats, axis=0)  # (*, dr)

    def save(self, model_dir, step):
        for member in range(self.de):
            for agent_idx, rew_model in enumerate(self.ensemble[member]):
                torch.save(rew_model.state_dict(),
                           '%s/reward_model_%s_mem_%s_a_%s.pt' % (model_dir, step, member, agent_idx))

    def load(self, model_dir, step):
        for member in range(self.de):
            for agent_idx, rew_model in enumerate(self.ensemble[member]):
                rew_model.load_state_dict(
                    torch.load('%s/reward_model_%s_mem_%s_a_%s.pt' % (model_dir, step, member, agent_idx))
                )

    @NotVerified
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len / batch_size))

        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch + 1) * batch_size
            if (epoch + 1) * batch_size > max_len:
                last_index = max_len

            sa_t_1 = self.buffer_seg1[epoch * batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch * batch_size:last_index]
            labels = self.buffer_label[epoch * batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(self.device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)

    def get_queries(self, mb_size=20):
        # MARK: get_queries
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)

        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1

        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])

        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1]  # Batch x T x dim of s&a, T is traj_len
        r_t_1 = train_targets[batch_index_1]  # Batch x T x dr

        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2]  # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2]  # Batch x T x dr

        ep_num_1 = np.array(self.episode_num_list)[batch_index_1].reshape(-1, 1)  # Batch x 1
        ep_num_2 = np.array(self.episode_num_list)[batch_index_2].reshape(-1, 1)

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])  # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])  # (Batch x T) x dr
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])  # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])  # (Batch x T) x dr

        # Generate time index
        time_index = np.array([list(range(i * len_traj,
                                          i * len_traj + self.size_segment)) for i in range(mb_size)])
        ep_start_1 = np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        ep_start_2 = np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        time_index_1 = time_index + ep_start_1
        time_index_2 = time_index + ep_start_2

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)  # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)  # Batch x size_seg x dr
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)  # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0)  # Batch x size_seg x dr

        return sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2

    def put_queries(self, sa_t_1, sa_t_2, labels):
        # MARK: put_queries
        total_sample = sa_t_1.shape[0]  # (batch_size, size_seg, ds + da)
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index

    @NotVerified
    def surf_data_aug_process(self):
        u_sa_t_1, u_sa_t_2, _, _, = self.get_queries(mb_size=self.mb_size * self.large_batch)
        self.put_unlabeled_queries(u_sa_t_1, u_sa_t_2)

    def put_unlabeled_queries(self, sa_t_1, sa_t_2, labels=None):
        # MARK: put_unlabeled_queries
        # r_t_1 = self.r_hat_batch(sa_t_1)
        # r_t_2 = self.r_hat_batch(sa_t_2)
        # sum_r_t_1 = np.sum(r_t_1[:, self.data_aug_window: -self.data_aug_window], axis=1)
        # sum_r_t_2 = np.sum(r_t_2[:, self.data_aug_window: -self.data_aug_window], axis=1)
        # labels = 1*(sum_r_t_1 < sum_r_t_2)
        if labels is None:
            labels = np.zeros((sa_t_1.shape[0], 1))  # (batch_size, 1)

        total_sample = sa_t_1.shape[0]
        next_index = self.u_buffer_index + total_sample
        if next_index >= self.capacity:
            self.u_buffer_full = True
            maximum_index = self.capacity - self.u_buffer_index
            np.copyto(self.u_buffer_seg1[self.u_buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.u_buffer_seg2[self.u_buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.u_buffer_label[self.u_buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.u_buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.u_buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.u_buffer_label[0:remain], labels[maximum_index:])

            self.u_buffer_index = remain
        else:
            np.copyto(self.u_buffer_seg1[self.u_buffer_index:next_index], sa_t_1)
            np.copyto(self.u_buffer_seg2[self.u_buffer_index:next_index], sa_t_2)
            np.copyto(self.u_buffer_label[self.u_buffer_index:next_index], labels)
            self.u_buffer_index = next_index

    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, 
            ep_num_1=None, ep_num_2=None, ep_start_1=None, ep_start_2=None):
        # MARK: get_label
        # r_t_1: (batch_size, size_seg, dr)

        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size - 1):
            temp_r_t_1[:, :index + 1] *= self.teacher_gamma
            temp_r_t_2[:, :index + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)  # (batch_size, dr)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)

        rational_labels = 1 * (sum_r_t_1 < sum_r_t_2)  # (batch_size, dr)
        if self.teacher_beta > 0:  # Bradley-Terry rational model
            r_hat = np.concatenate([np.expand_dims(sum_r_t_1, axis=-1), 
                                    np.expand_dims(sum_r_t_2, axis=-1)], axis=-1)
            r_hat = torch.from_numpy(r_hat).float().to(self.device)  # (batch_size, dr, 2)
            r_hat = r_hat * self.teacher_beta  # -r_hat
            r_hat = r_hat.reshape(-1, 2)  # (batch_size * dr, 2)
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, self.dr)  # (batch_size, dr)
        else:
            labels = rational_labels

        for i in range(min(20, len(sum_r_t_1))):
            self.df.loc[len(self.df)] = {
                'step': self.df_count, 
                'en0': ep_num_1[i][0], 'start0': ep_start_1[i][0], 
                'seg_R0': f'{sum_r_t_1[i][0]:.2f}',
                'en1': ep_num_2[i][0], 'start1': ep_start_2[i][0],
                'seg_R1': f'{sum_r_t_2[i][0]:.2f}', 
                'label': labels[i][0],
            }
        self.df_count += 1
        self.df.to_csv(os.path.join(self.save_query_dir, 'query.csv'), index=False)
        print(f'csv of query returns saved!')

        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels

    @NotVerified
    def kcenter_sampling(self):

        # get queries
        num_init = self.mb_size * self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size, self.device)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    @NotVerified
    def kcenter_disagree_sampling(self):

        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]

        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size, self.device)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    @NotVerified
    def kcenter_entropy_sampling(self):

        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]

        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size, self.device)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def uniform_sampling(self):
        # MARK: uniform_sampling
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2 = self.get_queries(
            mb_size=self.mb_size)

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:
            self.surf_data_aug_process()

        return len(labels)

    @NotVerified
    def disagreement_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        ep_num_1, ep_num_2 = ep_num_1[top_k_index], ep_num_2[top_k_index]
        ep_start_1, ep_start_2 = ep_start_1[top_k_index], ep_start_2[top_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        # surf: get unlabeled samples
        if self.data_aug_ratio:
            self.surf_data_aug_process()

        return len(labels)

    @NotVerified
    def entropy_sampling(self):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)

        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def train_reward(self):
        # MARK: train_reward
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss, loss_BT_mean, loss_reg_mean = 0.0, 0.0, 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]  # (batch size, segment len, ds + da)
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels).long().to(self.device)  # (batch size, dr)

                if member == 0:
                    total += labels.size(0) * self.dr

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch size, segment len, dr)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1).unsqueeze(-1)  # (batch size, dr, 1)
                r_hat2 = r_hat2.sum(axis=1).unsqueeze(-1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # (batch size, dr, 2)

                # compute loss
                labels = labels.reshape(-1)  # (batch size * dr)
                r_hat_flatten = r_hat.reshape(-1, 2)  # (batch size * dr, 2)
                curr_loss = self.CEloss(r_hat_flatten, labels)
                loss += curr_loss
                loss_BT_mean += curr_loss.item()

                # compute L_reg
                # 假设 R0 > R1, diff = R0-R1 > 0. 三个 agent 的 r0 > r1 是 1 1 0, r0 < r1 是 0 0 1
                sum_r_hat = r_hat.sum(axis=1)  # (batch size, 2)
                sum_r_hat_0_minus_1 = sum_r_hat[:, 0] - sum_r_hat[:, 1]  # (batch size, )
                if True:  # 更强的正则化
                    with torch.no_grad():
                        r_hat_0_preferred = (r_hat[:, :, 0] > r_hat[:, :, 1]).float()  # (batch size, dr),  [1 1 0]
                        r_hat_0_preferred_count = r_hat_0_preferred.sum(axis=1) / self.dr  # (batch size, ),  [2/3]
                        r_hat_1_preferred_count = 1.0 - r_hat_0_preferred_count  # (batch size, ),  [1/3]
                    sum_agent_r_hat_mismatch = torch.abs(-r_hat_0_preferred_count * sum_r_hat_0_minus_1) + \
                        torch.abs(r_hat_1_preferred_count * sum_r_hat_0_minus_1)  # (batch size, ),  [1/3*diff]
                    loss_reg = torch.mean(sum_agent_r_hat_mismatch) / self.size_segment  # scalar
                else:
                    with torch.no_grad():
                        r_hat_0_preferred = (r_hat[:, :, 0] > r_hat[:, :, 1]).float()  # (batch size, dr),  [1 1 0]
                        r_hat_0_preferred_count = r_hat_0_preferred.sum(axis=1) / self.dr - 0.5  # (batch size, ),  [1/6]
                    sum_agent_r_hat_mismatch = torch.abs(-r_hat_0_preferred_count * sum_r_hat_0_minus_1)  # (batch size, ),  [ abs(-1/6*diff) = 0]
                    loss_reg = torch.mean(sum_agent_r_hat_mismatch) / self.size_segment  # scalar
                loss_reg_mean += loss_reg.item()
                loss += loss_reg * self.L_reg_coef

                # compute acc
                _, predicted = torch.max(r_hat_flatten.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total
        loss_BT_mean = loss_BT_mean / self.de
        loss_reg_mean = loss_reg_mean / self.de

        return np.mean(ensemble_acc), loss.item(), loss_BT_mean, loss_reg_mean

    def test_reward_accuracy(self):
        # MARK: test_reward_accuracy

        # uniform_sampling
        sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2 = \
            self.get_queries(mb_size=self.mb_size)  # self.mb_size = 100
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, ep_num_1, ep_num_2, ep_start_1, ep_start_2)

        total = labels.shape[0] * self.dr

        # get random batch
        labels = torch.from_numpy(labels).long().to(self.device)  # (batch size, dr)

        # get logits
        r_hat1 = torch.from_numpy(self.r_hat_batch(sa_t_1)).to(self.device)  # (batch size, segment len, dr)
        r_hat2 = torch.from_numpy(self.r_hat_batch(sa_t_2)).to(self.device)
        r_hat1 = r_hat1.sum(axis=1).unsqueeze(-1)  # (batch size, dr, 1)
        r_hat2 = r_hat2.sum(axis=1).unsqueeze(-1)
        r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # (batch size, dr, 2)

        # compute acc
        labels = labels.reshape(-1)  # (batch size * dr)
        r_hat_flatten = r_hat.reshape(-1, 2)  # (batch size * dr, 2)
        _, predicted = torch.max(r_hat_flatten.data, 1)
        correct = (predicted == labels).sum().item()
        test_acc = correct / total

        return test_acc

    def semi_train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)
        u_max_len = self.capacity if self.u_buffer_full else self.u_buffer_index
        u_total_batch_index = self.shuffle_dataset(u_max_len)

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        mu = u_max_len / max_len
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            u_last_index = int((epoch + 1) * self.train_batch_size * mu)
            if u_last_index > u_max_len:
                u_last_index = u_max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                labels = labels.repeat(self.data_aug_ratio)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)  # (batch size, segment len, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                # shifting & cropping time (slightly shift or resize behaviour won't affect the preference)
                mask_1, mask_2 = self.get_cropping_mask(r_hat1, self.data_aug_ratio)
                r_hat1 = r_hat1.repeat(self.data_aug_ratio, 1, 1)  # (batch size * data_aug_ratio, segment len, 1)
                r_hat2 = r_hat2.repeat(self.data_aug_ratio, 1, 1)
                r_hat1 = (mask_1 * r_hat1).sum(axis=1)
                r_hat2 = (mask_2 * r_hat2).sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)

                # get random unlabeled batch
                u_idxs = u_total_batch_index[member][int(epoch * self.train_batch_size * mu): u_last_index]
                u_sa_t_1 = self.u_buffer_seg1[u_idxs]
                u_sa_t_2 = self.u_buffer_seg2[u_idxs]

                # get logits
                u_r_hat1 = self.r_hat_member(u_sa_t_1, member=member)
                u_r_hat2 = self.r_hat_member(u_sa_t_2, member=member)

                # pseudo-labeling
                u_r_hat1_noaug = u_r_hat1[:, self.data_aug_window:-self.data_aug_window]
                u_r_hat2_noaug = u_r_hat2[:, self.data_aug_window:-self.data_aug_window]
                with torch.no_grad():
                    u_r_hat1_noaug = u_r_hat1_noaug.sum(axis=1)
                    u_r_hat2_noaug = u_r_hat2_noaug.sum(axis=1)
                    u_r_hat_noaug = torch.cat([u_r_hat1_noaug, u_r_hat2_noaug], axis=-1)

                    pred = torch.softmax(u_r_hat_noaug, dim=1)
                    pred_max = pred.max(1)
                    # use unlabeled data only if preference prob exceed some threshold
                    mask = (pred_max[0] >= self.threshold_u)
                    pseudo_labels = pred_max[1].detach()
                pseudo_labels = pseudo_labels.repeat(self.data_aug_ratio)
                mask = mask.repeat(self.data_aug_ratio)

                # shifting & cropping time
                u_mask_1, u_mask_2 = self.get_cropping_mask(u_r_hat1, self.data_aug_ratio)
                u_r_hat1 = u_r_hat1.repeat(self.data_aug_ratio, 1, 1)
                u_r_hat2 = u_r_hat2.repeat(self.data_aug_ratio, 1, 1)
                u_r_hat1 = (u_mask_1 * u_r_hat1).sum(axis=1)
                u_r_hat2 = (u_mask_2 * u_r_hat2).sum(axis=1)
                u_r_hat = torch.cat([u_r_hat1, u_r_hat2], axis=-1)

                curr_loss += torch.mean(self.UCELoss(u_r_hat, pseudo_labels) * mask) * self.lambda_u

                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)  # supervised acc
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc

    def shuffle_dataset(self, max_len):
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        return total_batch_index

    def get_cropping_mask(self, r_hat1, w):
        mask_1_, mask_2_ = [], []
        B, S, _ = r_hat1.shape  # batch size, segment length
        mask_1 = torch.zeros((B * w, S, 1)).to(self.device)
        mask_2 = torch.zeros((B * w, S, 1)).to(self.device)

        length = np.random.randint(S - 15, S - 5 + 1, size=(B * w, ))  # 60 -> [45, 55]
        start_index_1 = np.random.randint(0, S + 1 - length)  # (B*w, )
        start_index_2 = np.random.randint(0, S + 1 - length)
        end_index_1 = np.array(start_index_1 + length, dtype=int)
        end_index_2 = np.array(start_index_2 + length, dtype=int)
        # print(f'start_index_1: {start_index_1[5:]}, end_index_1: {end_index_1[5:]}')
        indices = np.arange(S).reshape(1, -1)  # (1, segment_length)
        mask_1[(indices >= start_index_1[:, None]) & (indices <= end_index_1[:, None])] = 1
        mask_2[(indices >= start_index_2[:, None]) & (indices <= end_index_2[:, None])] = 1

        return mask_1, mask_2  # (batch_size * w, segment_length, 1)

    @NotVerified
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc
