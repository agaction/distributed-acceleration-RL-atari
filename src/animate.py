import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
_path = os.path.abspath(os.path.join(current_path, '../cule/examples'))
if not _path in sys.path:
    sys.path = [_path] + sys.path

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from torchcule.atari import Env, Rom
from utils.openai.envs import create_vectorize_atari_env

from a2c.model import ActorCritic
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CuLE')
    parser.add_argument('--debug', action='store_true', help='Single step through frames for debugging')
    parser.add_argument('--env-name', type=str, help='Atari Game')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (default: 0)')
    parser.add_argument('--initial-steps', type=int, default=1000, help='Number of steps used to initialize the environment')
    parser.add_argument('--num-envs', type=int, default=5, help='Number of atari environments')
    parser.add_argument('--rescale', action='store_true', help='Resize output frames to 84x84 using bilinear interpolation')
    parser.add_argument('--training', action='store_true', help='Set environment to training mode')
    parser.add_argument('--use-cuda', action='store_true', help='Execute ALEs on GPU')
    parser.add_argument('--use-openai', action='store_true', default=False, help='Use OpenAI Gym environment')
    parser.add_argument('--model-filename', type=str, default=None, help='filename of the model to load')
    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu) if args.use_cuda else 'cpu')
    debug  = args.debug
    num_actions = 4
    num_envs = args.num_envs

    if args.use_openai:
        env = create_vectorize_atari_env(args.env_name, seed=0, num_envs=args.num_envs,
                                         episode_life=False, clip_rewards=False)
        observation = env.reset()
    else:
        env = Env(args.env_name, args.num_envs, 'gray', device='cpu',
                  rescale=True, episodic_life=False, repeat_prob=0.0, frameskip=4)
        print(env.cart)

        if args.training:
            env.train()
        observation = env.reset(initial_steps=args.initial_steps)



    model = ActorCritic(4, env.action_space, normalize=True, name=args.env_name)
    model.load_state_dict(torch.load(args.model_filename))
    #print(model)
    model = model.to(device)
    #model.eval()    

    fig = plt.figure()
    img = plt.imshow(np.squeeze(np.hstack(observation.cpu().numpy())), animated=True, cmap=None)
    ax = fig.add_subplot(111)

    states = torch.zeros((num_envs, 4, 84, 84), device=device, dtype=torch.float32)
    states[:, -1] = observation.squeeze(-1).to(device=device, dtype=torch.float32)
    frame = 0

    if debug:
        ax.set_title('frame: {}, rewards: {}, done: {}'.format(frame, [], []))
    else:
        fig.suptitle(frame)

    def updatefig(*args):
        global ax, debug, device, env, frame, img, num_envs, model, states, fire_reset, lives

        if debug:
            input('Press Enter to continue...')

        policy = model(states)[1]
        actions = F.softmax(policy, dim=1).multinomial(1).cpu()
        #actions = env.sample_random_actions()



        observation, reward, done, info = env.step(actions)

        obs = np.squeeze(np.hstack(observation.cpu().numpy()))
        img.set_array(np.dstack([obs]*3))
        
        observation = observation.to(device=device, dtype=torch.float32)
        
        states[:, :-1].copy_(states[:, 1:].clone())
        states *= (1.0 - done.to(device=device, dtype=torch.float32)).view(-1, *[1] * (observation.dim() - 1))
        states[:, -1].copy_(observation.view(-1, *states.size()[-2:]))
        
        if debug:
            ax.title.set_text('{}) rewards: {}, done: {}'.format(frame, reward, done))
        else:
            fig.suptitle(frame)

        frame += 1

        return img,

    ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=False)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

