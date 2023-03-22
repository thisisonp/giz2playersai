import json
import torch
import copy
import random
from torch.distributions import Categorical

import visdom
vis = visdom.Visdom(env='giz_lsarsa')
first = True
total_round = 0
total_score = [0, 0]
if True:
    import sys
    import os
    sys.path.append(os.path.realpath('..'))
    from ai_2p.IDGenerator import IDGenerator
    # from ai_2p.Critic import Critic
    from ai_2p.QLearner import QLearner
    from ai_2p.PPOModel import PPOModel
    from ai_2p.SARSA import SARSA

    from env.types import ActionType, Observation, Action
    from env.common import Stage
    from env.GizmosEnv import GizmosEnv

def sqr(x):
    return x * x
env = GizmosEnv(player_num=2, log=False, check=False)
debug_round = 1000
eps_clip = 0.1

idg = IDGenerator(path='d.json')
models = [SARSA(idg, path='../SARSA-1p.pkl'), SARSA(idg, path='../SARSA-2p.pkl')
          ]
# models = [PPOModel(idg, path='ppo-1p.pkl'),PPOModel(idg, path='ppo-2p.pkl')
#            ]
optimizers = [torch.optim.SGD(model.parameters(), lr=0.003)
              for model in models]
# optimizers = torch.optim.Adam(critic.parameters(), lr=0.01)
# optimizers = torch.optim.RMSprop(critic.parameters())

best_turn: int = 25

def log_replay(replay: list[Observation | Action],
               observation: Observation | None = None,
               action: Action | None = None):
    if observation == None:
        if action == None:
            return
        replay.append(copy.deepcopy(action))
    else:
        ob = copy.deepcopy(observation)
        del ob['gizmos']
        replay.append(ob)


log_file = open('log.txt', 'a+')

w0 = 0
w1 = 0
max_can_num = 0
ppo_actor_loss = 0
ppo_critic_loss = 0
batch_actor_loss = [0.0, 0.0]
batch_other_loss = [0.0, 0.0]
batch_critic_loss = [0.0, 0.0]
batch_count = [0.0, 0.0]
for i in range(1000000):
    # if i % 100 == 0:
    #     env.reset()
    #     temp_env = copy.deepcopy(env)
    # else:
    #     env = copy.deepcopy(temp_env)
    env.reset()
    last_score = 0
    ret = 0
    input = [[], []]
    output = [[], []]
    random_a = [[], []]
    action = [[], []]
    traj = []
    replay: list[Observation | Action] = []

    models[1].ppo_total_loss = []

    last_score = [0, 0]
    while True:
        np = env.state['curr_player_index']
        model = models[np]
        ob = env.observation(np)
        log_replay(replay, observation=ob)
        action_space = ob['action_space']
        if ob['curr_stage'] == Stage.GAME_OVER or ob['curr_turn'] > 25:
            break
        act = None
        feature, dense_feature = model.get_context_feature(ob)
        ti = []
        end_act = None
        can_num = 0
        debug = []
        for action in action_space:
            if action['type'] == ActionType.END:
                end_act = action
                # print("end", can_num)
                continue
            ids = "None"
            if "id" in action.keys():
                ids = action['id']
            debug.append(action['type'] + str(ids))
            act_feature = model.gen_action_feature(action)
            ti.append(copy.copy(feature + dense_feature + act_feature))
            can_num += 1
        # print("all", can_num)
        # if can_num > 30:
        #     print("all", debug)
        max_can_num = max(max_can_num, can_num)
        if model.model_name == "PPO":
            if can_num == 0:
                act = end_act
                model.ppo_total_loss.append(torch.tensor(0.0))
            else:
                if can_num == 1:
                    yhat = model.actor_forward(torch.Tensor(ti)).view(-1, )
                    model.ppo_total_loss.append(torch.tensor(0.0))
                    act = action_space[0]
                else:
                    yhat = model.actor_forward(torch.Tensor(ti)).view(-1, )
                    batch_other_loss[np] += -0.000001 * torch.sum(torch.log(torch.clamp(yhat, 1e-9, 1.0)))
                    best_action = yhat  # / sum(yhat)
                    # if i % 100 == 0:
                    #     print("!!", best_action, debug)
                    # print("?", ti)
                    dist = Categorical(best_action)
                    idx = dist.sample()
                    # print("meet", yhat[idx])
                    model.ppo_total_loss.append(yhat[idx] / yhat[idx].detach())
                    act = action_space[idx]

        elif model.model_name == "QLearner":
            yhat = model.forward(torch.Tensor(ti)).view(-1,)
            # def sample_gumbel(shape, eps=1e-20):
            #     U = torch.rand(shape)
            #     return -torch.log(-torch.log(U + eps) + eps)
            if random.random() < 0.01:
                best_action = torch.rand(yhat.shape) / 1.0
            else:
                best_action = yhat  # + sample_gumbel(yhat.shape) / 10000.0
            # print(yhat, best_action)
            if best_action.numel() == 0:
                act = end_act
            else:
                idx = torch.argmax(best_action)
                act = action_space[idx]
        elif model.model_name == "SARSA":
            yhat = model.forward(torch.Tensor(ti)).view(-1,)
            # def sample_gumbel(shape, eps=1e-20):
            #     U = torch.rand(shape)
            #     return -torch.log(-torch.log(U + eps) + eps)
            if random.random() < 0.03:
                best_action = torch.rand(yhat.shape) / 1.0
                random_a[np].append(True)
            else:
                best_action = yhat  # + sample_gumbel(yhat.shape) / 10000.0
                random_a[np].append(False)
            # print(yhat, best_action)
            if best_action.numel() == 0:
                act = end_act
                random_a[np][-1] = False
            else:
                idx = torch.argmax(best_action)
                act = action_space[idx]
        else:
            print("unknown model")
            exit(0)

        traj.append(str(ob['curr_turn']) + ": " + str(act))
        act_feature = model.gen_action_feature(act)
        input[np].append(list(map(int, feature + act_feature)))
        input_dense[np].append(list(map(float, dense_feature)))
        output[np].append((ob['players'][np]['score'] - last_score[np]) / 100.0)
        # output[np].append(0)
        last_score[np] = ob['players'][np]['score']
        env.step(np, act)
        log_replay(replay, action=act)

    ob = env.observation(0)
    p0 = ob['players'][0]
    p1 = ob['players'][1]
    if p0['score'] != p1['score']:
        winner = 0 if p0['score'] > p1['score'] else 1
    elif len(p0['gizmos']) != len(p1['gizmos']):
        winner = 0 if len(p0['gizmos']) > len(p1['gizmos']) else 1
    elif p0['total_energy_num'] != p1['total_energy_num']:
        winner = 0 if p0['total_energy_num'] > p1['total_energy_num'] else 1
    else:
        winner = 1
    if winner == 0:
        w0 += 1
    else:
        w1 += 1
    total_round += ob['curr_turn']
    total_score[0] += ob['players'][0]['score']
    total_score[1] += ob['players'][1]['score']
    for np in range(2):
        last = len(output[np]) - 1
        v = 0 # + (ob['players'][np]['score']) / 100.0 - ob['curr_turn'] * 0.1
        v = 0 ## - ob[
            #'curr_turn'] * 10) / 100.0
        if np == winner:
            # v += ob['players'][np]['score'] / ob['curr_turn']
            #v += 1
            v += (27 - ob['curr_turn']) / 2 * (1 + ob['players'][np]['score'] / 100.0 - ob['players'][1 - np]['score'] / 100.0)
        output[np].append(v)
        random_a[np].append(False)
        # print(v)
        # while last >= 0:
        #     output[np][last] += v
        #     # v *= 0.999
        #     last -= 1
    for np in range(2):
        # input[np] = input[np][:-1]
        # output[np] = output[np][1:]
        model = models[np]
        if model.model_name == "PPO":
            # optimizers[np].zero_grad()
            vhat = model.critic_forward(torch.Tensor(input[np]))
            for j in range(len(input[np]) - 1):
                batch_actor_loss[np] += -model.ppo_total_loss[j] * (output[np][j + 1] - vhat[j].detach())
                vt = output[np][j]
                if j == len(input[np]) - 1:
                    vt = vhat[j + 1].detach()
                batch_critic_loss[np] += sqr(vt - vhat[j])#append(output[np][j])
                # batch_critic_loss_vhat[np].append(vhat[j])
            batch_count[np] += len(input[np]) - 1
            # batch_out[np] += output[np][1:]
            # batch_out_all[np] += output[np]
            # batch_ppo_total_loss[np] += model.ppo_total_loss[:-1]
            if i % 10 == 9:
                # yhat = model.actor_forward(torch.Tensor(input[np][:-1]))
                # print("!!!", model.ppo_total_loss[:-1])
                # print(model.ppo_total_loss[:-1])
                # print((torch.Tensor(output[np][1:]) - vhat[:-1].detach()))
                # A = torch.stack(batch_ppo_total_loss[np]) * (torch.Tensor(batch_out[np]) - vhat[:-1].detach())
                # A = torch.log(yhat / torch.sum(yhat.detach())) * (torch.Tensor(output[np][1:]))
                # ppo_actor_loss = -torch.mean(torch.stack) * 3
                # ppo_actor_loss = torch.sum(model.ppo_total_loss[:-1])
                # ppo_critic_loss = torch.mean(model.critic_loss(torch.Tensor(output[np]), vhat))
                # loss = ppo_actor_loss + ppo_critic_loss
                loss = (batch_actor_loss[np] * 10 + batch_critic_loss[np] + batch_other_loss[np]) / batch_count[np]
                ppo_actor_loss = batch_actor_loss[np] / batch_count[np]
                ppo_critic_loss = batch_critic_loss[np] / batch_count[np]
                ppo_other_loss = batch_other_loss[np] / batch_count[np]
                batch_actor_loss[np] = 0
                batch_other_loss[np] = 0
                batch_critic_loss[np] = 0
                batch_count[np] = 0
                # loss += 0.00001 * torch.sum(torch.log(torch.clamp(yhat, 1e-9, 1.0)))
                loss.backward()
                # print("?", model.ppo_total_loss[-1], model.ppo_total_loss[-1].grad)
                # print("?", torch.sum(model.base_embedding.grad))
                optimizers[np].step()
                optimizers[np].zero_grad()
        elif model.model_name == "QLearner":
            # pass
            exit()
            pass
            # input[np] = input[np][:-1]
            # output[np] = output[np][1:]
            # yhat = model.forward(torch.Tensor(input[np]))
            # for j in range(len(input[np])):
            #     batch_critic_loss[np] += sqr(output[np][j] - yhat[j])
            # # optimizers[np].zero_grad()
            # batch_count[np] += len(input[np])
            # if i % 10 == 9:
            #     loss = batch_critic_loss[np] / batch_count[np]
            #     loss.backward()
            #     optimizers[np].step()
            #     optimizers[np].zero_grad()
            #     batch_critic_loss[np] = 0
            #     batch_count[np] = 0
            #     qlloss = loss
        elif model.model_name == "SARSA":
            # input[np] = input[np][:-1]
            # output[np] = output[np][1:]
            yhat = model.forward(torch.Tensor(input[np]), torch.Tensor(input_dense[np]))
            # for j in range(len(input[np]) - 1):
            #     batch_sarsa[np].append([input[np], ])
            for j in range(len(input[np])):
                if j == len(input[np]) - 1:
                    batch_critic_loss[np] += sqr(output[np][j + 1] - yhat[j])
                else:
                    if not random_a[np][j + 1]:
                        # batch_critic_loss[np] += sqr(yhat[j + 1].detach() - yhat[j] - output[np][j + 1])
                        jj = j
                        ret = yhat[j]
                        while jj < len(input[np]) - 1 and not random_a[np][jj + 2]:
                            ret += output[np][jj + 1]
                            jj += 1
                        if jj == len(input[np]) - 1:
                            ret = sqr(ret - output[np][jj + 1])
                        else:
                            ret = sqr(ret - yhat[jj + 1].detach())
                        batch_critic_loss[np] += ret
                        # print(ret)
                    else:
                        batch_count[np] -= 1
            # optimizers[np].zero_grad()
            batch_count[np] += len(input[np])
            if i % 5 == 4 and batch_count[np] > 0:
                loss = batch_critic_loss[np] / batch_count[np]
                loss.backward()
                optimizers[np].step()
                optimizers[np].zero_grad()
                batch_critic_loss[np] = 0
                batch_count[np] = 0
                qlloss = loss
        else:
            print("incomplete model")
            exit(0)

    if i == 0:
        continue
    if i % 100 == 0:
        print(traj)
        print("step", i)
    if i % 10 == 0:
        # raw_log = "Games played:", i, "; token seen:", idg.cnt, "; loss:", float(ppo_critic_loss), float(ppo_actor_loss), float(ppo_other_loss), "; end turn", ob[
        #     'curr_turn'], "; final score",  p0['score'],  p1['score'], '; maxcan:', max_can_num
        raw_log = "Games played:", i, "; token seen:", idg.cnt, "; loss:", float(qlloss), "; end turn", ob[
            'curr_turn'], "; final score",  p0['score'],  p1['score'], '; maxcan:', max_can_num
        train_log = ' '.join(map(lambda x: str(x), raw_log))
        print(train_log)
        log_file.write(train_log + '\n')

        vis.line(X=torch.tensor([i, ]), Y=torch.tensor([w0 / (w0 + w1 + 0.000000001), ]), win='p0 win percent',
                 update='append' if not first else None, opts = {'title':"p0 win percent"})
        vis.line(X=torch.tensor([i, ]), Y=torch.tensor([total_score[0] / (total_round + 0.000000001), ]), win='scores/turns', name="p0",
                 update='append' if not first else None, opts = {'showlegend':True,'title':"scores/turns"})
        vis.line(X=torch.tensor([i, ]), Y=torch.tensor([total_score[1] / (total_round + 0.000000001), ]), win='scores/turns', name="p1",
                 update='append' if not first else None)
        vis.line(X=torch.tensor([i, ]), Y=torch.tensor([total_round / 10.0, ]), win='turns',
                 update='append' if not first else None, opts = {'title':"average end turn"})
        first = False
        total_score = [0, 0]
        total_round = 0
        w0 = 0
        w1 = 0

    if i % 5000 == 0:
        for np in range(2):
            models[np].save(models[np].model_name + '-{}p.pkl'.format(np + 1))
    # print("???", torch.sum(models[1].base_embedding))
    if ob['curr_turn'] < best_turn:
        best_turn = ob['curr_turn']
        json_replay = json.dumps(replay)
        with open('pro_play.json', 'w+') as f:
            f.write(json_replay)
