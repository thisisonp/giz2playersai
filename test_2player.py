from GizmosEnv import GizmosEnv
from common import Stage
import torch
import copy
import random
import os
from GizmosEnv import ActionType
env = GizmosEnv(player_num=2,log=False)
debug_round = 1000
class IDGenerator(object):
    def __init__(self):
        self.d = {}
        self.cnt = 0
    def gen_unique_id(self, px, _data, i=-1, energe_flag=False):
        if i == -1:
            data = str(_data)
            if (px + data) not in self.d.keys():
                self.d[(px + data)] = self.cnt
                self.cnt += 1
            return self.d[(px + data)]
        else:
            data = _data
            name = px
            if len(data) <= i:
                name += "none"
            else:
                if energe_flag:
                    name += str(data[i])
                else:
                    name += str(data[i]['id']) + str(data[i]['used'])
            if (name) not in self.d.keys():
                self.d[name] = self.cnt
                self.cnt += 1
            return self.d[name]
    def restore(self, _d):
        self.d = {}
        self.cnt = 0
        for k, v in _d:
            self.d[k] = v
            self.cnt += 1

class Critic(torch.nn.Module):
    def __init__(self, feature_len, idg):
        super(Critic, self).__init__()
        self.idg = idg
        self.feature_len = feature_len
        self.embedding_len = 32
        # self.device = 'cpu'
        self.base_embedding = torch.nn.Parameter(torch.randn(10000, self.embedding_len),
                                           requires_grad=True)

        self.loss_op = torch.nn.MSELoss(reduce=False)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.feature_len * self.embedding_len, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
    def forward(self, x):
        ids = x.view(-1, 1).to(torch.long)
        batch_input = self.base_embedding[ids, :].view(-1, self.feature_len * self.embedding_len)
        return self.model(batch_input)
    def best_action(self, env):
        np = env.state['curr_player_index']
        info = env.observation(np)
        feature = self.get_context_feature(info, np)
        ti = []
        for j in env.action_space:
            self.add_action_feature(feature, j)
            ti.append(copy.copy(feature))
            feature = feature[:-4]
        yhat = critic[np].forward(torch.Tensor(ti)).view(-1,)
        return env.action_space[torch.argmax(yhat)]
    def loss(self, y, yhat):
        return self.loss_op(y.view(-1, 1), yhat)
    
    def get_context_feature(self, info, np):
        return [self.idg.gen_unique_id('curr_stage', str(info['curr_stage'])),
                self.idg.gen_unique_id('curr_turn', info['curr_turn']),
                self.idg.gen_unique_id('energy_board', info['energy_board'], 0, True),
                self.idg.gen_unique_id('energy_board', info['energy_board'], 1, True),
                self.idg.gen_unique_id('energy_board', info['energy_board'], 2, True),
                self.idg.gen_unique_id('energy_board', info['energy_board'], 3, True),
                self.idg.gen_unique_id('energy_board', info['energy_board'], 4, True),
                self.idg.gen_unique_id('energy_board', info['energy_board'], 5, True),
                self.idg.gen_unique_id('gizmos_board', info['gizmos_board'][1], 0),
                self.idg.gen_unique_id('gizmos_board', info['gizmos_board'][1], 1),
                self.idg.gen_unique_id('gizmos_board', info['gizmos_board'][1], 2),
                self.idg.gen_unique_id('gizmos_board', info['gizmos_board'][1], 3),
                self.idg.gen_unique_id('gizmos_board', info['gizmos_board'][2], 0),
                self.idg.gen_unique_id('gizmos_board', info['gizmos_board'][2], 1),
                self.idg.gen_unique_id('gizmos_board', info['gizmos_board'][2], 2),
                self.idg.gen_unique_id('gizmos_board', info['gizmos_board'][3][0]),
                self.idg.gen_unique_id('gizmos_board', info['gizmos_board'][3][1]),
                self.idg.gen_unique_id('free_pick_num', info['free_pick_num']),
                self.idg.gen_unique_id('players_giz', info['players'][np]['file_gizmos'], 0),
                self.idg.gen_unique_id('players_giz', info['players'][np]['file_gizmos'], 1),
                self.idg.gen_unique_id('players_giz', info['players'][np]['file_gizmos'], 2),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 0),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 1),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 2),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 3),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 4),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 5),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 6),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 7),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 8),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 9),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 10),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 11),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 12),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 13),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 14),
                self.idg.gen_unique_id('players_giz', info['players'][np]['gizmos'], 15),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['file_gizmos'], 0),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['file_gizmos'], 1),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['file_gizmos'], 2),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 0),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 1),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 2),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 3),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 4),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 5),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 6),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 7),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 8),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 9),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 10),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 11),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 12),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 13),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 14),
                self.idg.gen_unique_id('players_giz', info['players'][1 - np]['gizmos'], 15),
                self.idg.gen_unique_id('ball', info['players'][np]['energy_num']['red']),
                self.idg.gen_unique_id('ball', info['players'][np]['energy_num']['black']),
                self.idg.gen_unique_id('ball', info['players'][np]['energy_num']['blue']),
                self.idg.gen_unique_id('ball', info['players'][np]['energy_num']['yellow']),
                self.idg.gen_unique_id('ball', info['players'][1 - np]['energy_num']['red']),
                self.idg.gen_unique_id('ball', info['players'][1 - np]['energy_num']['black']),
                self.idg.gen_unique_id('ball', info['players'][1 - np]['energy_num']['blue']),
                self.idg.gen_unique_id('ball', info['players'][1 - np]['energy_num']['yellow'])
                ]
    def add_action_feature(self, feature, j):
        feature.append(self.idg.gen_unique_id("actiontype", str(j['type'])))
        if j['type'] == ActionType.RESEARCH:
            feature.append(self.idg.gen_unique_id("level", str(j['level'])))
            feature.append(self.idg.gen_unique_id("res", "-1"))
            feature.append(self.idg.gen_unique_id("res", "-1"))
        elif j['type'] == ActionType.PICK:
            feature.append(self.idg.gen_unique_id("energy", str(j['energy'])))
            feature.append(self.idg.gen_unique_id("pic", "-1"))
            feature.append(self.idg.gen_unique_id("pic", "-1"))
        elif j['type'] == ActionType.END:
            feature.append(self.idg.gen_unique_id("en", "-1"))
            feature.append(self.idg.gen_unique_id("en", "-1"))
            feature.append(self.idg.gen_unique_id("en", "-1"))
        elif j['type'] == ActionType.GIVE_UP:
            feature.append(self.idg.gen_unique_id("gi", "-1"))
            feature.append(self.idg.gen_unique_id("gi", "-1"))
            feature.append(self.idg.gen_unique_id("gi", "-1"))
        elif j['type'] == ActionType.USE_GIZMO or j['type'] == ActionType.FILE or j[
            'type'] == ActionType.FILE_FROM_RESEARCH:
            feature.append(self.idg.gen_unique_id("ffr", str(j['id'])))
            feature.append(self.idg.gen_unique_id("ffr", "-1"))
            feature.append(self.idg.gen_unique_id("ffr", "-1"))
        else:
            if 'id' not in j.keys():
                print("??", str(j['type']))
            feature.append(self.idg.gen_unique_id("id", str(j['id'])))
            if j['type'] == ActionType.BUILD_FOR_FREE:
                feature.append(self.idg.gen_unique_id("cos", "-1"))
                feature.append(self.idg.gen_unique_id("ccg", "-1"))
            else:
                feature.append(self.idg.gen_unique_id("cost", str(j['cost_energy_num'])))
                feature.append(self.idg.gen_unique_id("cost_converter_gizmos_id", str(len(j['cost_converter_gizmos_id']))))
times = 0

critic = [None, None]
optimizer = [None, None]
idg = IDGenerator()
critic[0] = Critic(64 + 4, idg)
critic[1] = Critic(64 + 4, idg)
# optimizer = torch.optim.Adam(critic.parameters(), lr=0.01)
# optimizer = torch.optim.RMSprop(critic.parameters())
optimizer[0] = torch.optim.SGD(critic[0].parameters(), lr=0.01)
optimizer[1] = torch.optim.SGD(critic[1].parameters(), lr=0.01)

f = open("two_player", "w")

for i in range(1000000):
    env.reset()
    last_score = 0
    ret = 0
    input = [[], []]
    output = [[], []]
    action = [[], []]
    traj = []
    last_ball = 0
    np = 0
    while env.state['curr_stage'] != Stage.GAME_OVER:
        np = env.state['curr_player_index']
        info = env.observation(np)
        if info['curr_turn'] > 70:
            break
        at = None

        feature = critic[0].get_context_feature(info, np)
        at = None
        # if random.random() > 0.01:
        ti = []
        endact = None
        for j in env.action_space:
            if j['type'] == ActionType.END:
                endact = j
                continue
            critic[0].add_action_feature(feature, j)
            ti.append(copy.copy(feature))
            feature = feature[:-4]
        # if times % debug_round == 0:
        #     print(ti)
        tmp = torch.Tensor(ti)
        yhat = critic[np].forward(torch.Tensor(ti)).view(-1,)

        def sample_gumbel(shape, eps=1e-20):
            U = torch.rand(shape)
            return -torch.log(-torch.log(U + eps) + eps)
        if random.random() < 0.05:
            best_action = sample_gumbel(yhat.shape) / 1.0
        else:
            best_action = yhat # + sample_gumbel(yhat.shape) / 10000.0
        # print(yhat, best_action)
        if best_action.numel() == 0:
            at = endact
        else:
            at = env.action_space[torch.argmax(best_action)]
        critic[0].add_action_feature(feature, at)

        traj.append(str(at))
        # feature.append(self.idg.gen_unique_id("action", str(at)))
        input[np].append(list(map(int,feature)))
        output[np].append(0)
        # last_ball = now_ball
        env.step(np, at)

    winner = 0
    info = env.observation(0)
    if info['players'][0]['score'] < info['players'][1]['score']:
        winner = 1
    elif info['players'][0]['score'] == info['players'][1]['score']:
        if len(info['players'][0]['gizmos']) < len(info['players'][1]['gizmos']):
            winner = 1
        elif len(info['players'][0]['gizmos']) == len(info['players'][1]['gizmos']):
            if info['players'][0]['total_energy_num'] < info['players'][1]['total_energy_num']:
                winner = 1
            elif info['players'][0]['total_energy_num'] == info['players'][1]['total_energy_num']:
                winner = 1

    for pl in range(2):
        last = len(output[pl]) - 1
        v = 0 + info['players'][pl]['score'] / 100.0
        if pl == winner:
            v += 1
        while last >= 0:
            output[pl][last] = v
            last -= 1
    if times % debug_round == 0:
        print(traj)
    for pl in range(2):
        input[pl] = input[pl][:-1]
        output[pl] = output[pl][1:]
        yhat = critic[pl].forward(torch.Tensor(input[pl]))
        loss = torch.mean(critic[pl].loss(torch.Tensor(output[pl]), yhat))
        optimizer[pl].zero_grad()
        loss.backward()
        optimizer[pl].step()

    times += 1
    if times % debug_round == 0:
        print("step", times)
    if times % 10 == 0:
        print("Games played:", times, "; token seen:", idg.cnt, "; loss:",float(loss), "; end turn", env.observation(0)['curr_turn'], "; final score",  env.observation(0)['players'][0]['score'],  env.observation(0)['players'][1]['score'])

    if times == 20000:
        ff = open("d.txt", "w")
        for dk in d.keys():
            ff.write(dk + "\n")
        ff.close()
    f.write(str(winner) + "," + str(env.observation(0)['players'][0]['score'])  + "," + str(env.observation(0)['players'][1]['score']) + "," + str(env.observation(0)['curr_turn']) + "\n")
