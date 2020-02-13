import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import joblib
import tensorflow as tf
import os
from sac.misc import utils
from sac.policies.hierarchical_policy import FixedOptionPolicy
from sac.misc.sampler import rollouts
import pdb
# from sac.value_functions import NNQFunction, NNVFunction, NNDiscriminatorFunction

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--max-path-length', '-l', type=int, default=100)
    parser.add_argument('--n_paths', type=int, default=1)
    parser.add_argument('--dim_0', type=int, default=0)
    parser.add_argument('--dim_1', type=int, default=1)
    parser.add_argument('--use_qpos', type=bool, default=False)
    parser.add_argument('--use_action', type=bool, default=False)
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
                        action='store_false')
    parser.add_argument('--plot-one-dim',dest='plot_one_dim',type=int, default=0)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()
    
    # filename = '{}_{}_{}_trace.png'.format(os.path.splitext(args.file)[0][0:12]
                                            # +"plot_trace_imgs/"+os.path.splitext(args.file)[0][12:], 
                                            # args.dim_0, args.dim_1)

    checkpoint_dir = args.file.split("itr")[0]
    checkpoint_file = args.file[len(checkpoint_dir):]
    
    if args.plot_one_dim:
        save_plot_path = os.path.join(os.getcwd(), checkpoint_dir, 'plot_trace_imgs_{}'.format(args.dim_0))
        filename = '{}/{}_{}_trace.png'.format(save_plot_path,checkpoint_file.split(".")[0],args.dim_0)
    
    else:
        save_plot_path = os.path.join(os.getcwd(), checkpoint_dir, 'plot_trace_imgs_{}_{}'.format(args.dim_0,args.dim_1))
        filename = '{}/{}_{}_{}_trace.png'.format(save_plot_path,checkpoint_file.split(".")[0],args.dim_0,args.dim_1)
    
    # filename = '{}_{}_{}_trace.png'.format(os.path.splitext(args.file)[0],
                                           # args.dim_0, args.dim_1)
    
    # save_plot_path = os.path.join(os.getcwd(),args.file.split("itr")[0],"plot_trace_imgs_{}_{}".format(args.dim_0, args.dim_1))


    if not os.path.exists(save_plot_path):
        os.mkdir(save_plot_path)


    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        disc = data['discriminator']
        env = data['env']
        num_skills = data['policy'].observation_space.flat_dim - data['env'].spec.observation_space.flat_dim
        
        # discriminator = NNDiscriminatorFunction(
                            # env_spec=env.spec,
                            # hidden_layer_size=disc._layer_sizes[0:2],
                            # num_skills=num_skills
                            # )
        top_rewards = np.zeros((num_skills,))
        # top_rewards = np.zeros((num_skills,))
        # plt.figure(figsize=(6, 6))
        # fig, ax = plt.subplots(2,1,figsize=(6,12))
        fig, ax = plt.subplots(3,1,figsize=(6,18))
        palette = sns.color_palette('hls', num_skills)
        with policy.deterministic(args.deterministic):
            for z in range(num_skills):
                fixed_z_policy = FixedOptionPolicy(policy, num_skills, z)
                cum_skill_task_reward = []
                cum_reward = 0
                for path_index in range(args.n_paths):
                    obs = env.reset()
                    if args.use_qpos:
                        qpos = env.wrapped_env.env.model.data.qpos[:, 0]
                        obs_vec = [qpos]
                    else:
                        obs_vec = [obs]
                    for t in range(args.max_path_length):
                        action, _ = fixed_z_policy.get_action(obs)
                        # div_rew = disc(obs)
                        (obs, reward, _, _) = env.step(action)
                        cum_reward += reward*1.0
                        cum_skill_task_reward.append(cum_reward*1.0)
                        if args.use_qpos:
                            qpos = env.wrapped_env.env.model.data.qpos[:, 0]
                            obs_vec.append(qpos)
                        elif args.use_action:
                            obs_vec.append(action)
                        else:
                            obs_vec.append(obs)

                    obs_vec = np.array(obs_vec)
                    if args.plot_one_dim:
                        x = obs_vec[:,args.dim_0]
                        ax[0].plot(x,c=palette[z])
                    
                    else:
                        x = obs_vec[:, args.dim_0]
                        y = obs_vec[:, args.dim_1]

                        # x = obs_vec[:,1]
                        # y = obs_vec[:,2]
                        # plt.plot(x, y, c=palette[z])
                        ax[0].plot(x,y,c=palette[z])
                    ax[1].plot(cum_skill_task_reward,c=palette[z])
                    top_rewards[z] = cum_reward*1.0 

        sorted_args = np.argsort(top_rewards,axis=0)
        ax[2].plot(top_rewards)
        plt.savefig(filename)
        plt.close()
