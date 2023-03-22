import copy
from typing import Optional
import torch
from .IDGenerator import IDGenerator

if True:
    import sys
    import os
    sys.path.append(os.path.realpath('..'))
    from env.types import ActionType, Observation, Action


def lget(l: list, i: int):
    return l[i] if i < len(l) else None


def lget_id(l: list, i: int):
    x = lget(l, i)
    return x['id'] if x is not None else None


class SARSA(torch.nn.Module):
    def __init__(self, idg: Optional[IDGenerator] = None, path='sarsa.pkl'):
        super(SARSA, self).__init__()
        self.model_name = "SARSA"
        self.idg = idg or IDGenerator()
        self.context_feature_id_len = 68
        self.context_feature_dense_len = 18
        self.action_feature_id_len = 29
        self.action_feature_dense_len = 0
        self.all_feature_len = self.context_feature_id_len + self.context_feature_dense_len + self.action_feature_id_len + self.action_feature_dense_len
        self.embedding_len = 8
        self.input_len = self.context_feature_id_len * self.embedding_len + self.context_feature_dense_len + self.action_feature_id_len * self.embedding_len + self.action_feature_dense_len
        # self.device = 'cpu'
        self.base_embedding = torch.nn.Parameter(torch.randn(2000, self.embedding_len),
                                                 requires_grad=True)

        self.loss_op = torch.nn.MSELoss(reduce=False)
        # self.loss_op = torch.nn.BCELoss(reduce=False) # torch.nn.MSELoss(reduce=False)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_len * self.embedding_len, 512),
            torch.nn.ReLU(),
            # torch.nn.Linear(512, 512),
            # torch.nn.ReLU(),
            # torch.nn.Linear(512, 512),
            # torch.nn.ReLU(),
            # torch.nn.Linear(1024, 512),
            # torch.nn.ReLU(),
            # torch.nn.Linear(512, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 64),
            # torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            # torch.nn.Sigmoid()
        )

        if not os.path.exists(path):
            print('[SARSA.__init__] init model')
        else:
            print('[SARSA.__init__] load model from {}'.format(path))
            self.load_state_dict(torch.load(path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x.view(-1, self.all_feature_len)
        context_ids = torch.reshape(features[:, :self.context_feature_id_len], (-1, 1)).to(torch.long)
        context_dense = features[:, self.context_feature_id_len:self.context_feature_id_len + self.context_feature_dense_len]
        action_ids = torch.reshape(features[:, -(self.action_feature_id_len+self.action_feature_dense_len) : -self.action_feature_dense_len], (-1, 1)).to(torch.long)
        print("1", context_ids)
        print("2", -(self.action_feature_id_len+self.action_feature_dense_len), -self.action_feature_dense_len, action_ids)
        batch_input = torch.concat([self.base_embedding[context_ids, :].view(-1, self.context_feature_id_len * self.embedding_len),
                                self.base_embedding[action_ids, :].view(-1, self.action_feature_id_len * self.embedding_len),
                                context_dense], dim=1)
        return self.model(batch_input)

    def best_action(self, ob: Observation):
        feature, dense = self.get_context_feature(ob)
        feature = feature + dense
        ti = []
        for action in ob['action_space']:
            self.add_action_feature(feature, action)
            ti.append(copy.copy(feature))
            feature = feature[:-29]
        yhat = self.forward(torch.Tensor(ti)).view(-1,)
        return ob['action_space'][torch.argmax(yhat)]

    def loss(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        return self.loss_op(yhat.view(-1, ), y.view(-1, ))

    def get_context_feature(self, ob: Observation):
        gen = self.idg.gen_unique_id
        np = ob['curr_player_index']
        me = ob['players'][np]
        rival = ob['players'][1 - np]

        board_en = ob['energy_board']
        board_giz = ob['gizmos_board']
        board_giz_l1 = board_giz.get(1) or board_giz.get('1')
        board_giz_l2 = board_giz.get(2) or board_giz.get('2')
        board_giz_l3 = board_giz.get(3) or board_giz.get('3')
        my_en = me['energy_num']
        rival_en = rival['energy_num']

        dense_part = [board_en.count('red'), board_en.count('black'), board_en.count('blue'), board_en.count('yellow'),
                      my_en['red'], my_en['black'], my_en['blue'], my_en['yellow'],
                      rival_en['red'], rival_en['black'], rival_en['blue'], rival_en['yellow'],
                      len(me['gizmos']), len(me['filed']), me['score'],
                      len(rival['gizmos']), len(rival['filed']), rival['score'],]
        # a = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # b = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # all_giz = []
        # tmp_giz = [0 for i in range(12)]
        # for i in range(len(me['gizmos'])):
        #     tmp_giz = [0 for i in range(12)]
        #     if me['gizmos'][i].type == 'BUILD':
        #         a[0][me['gizmos'][i].effect.color] += 1
        #         tmp_giz[me['gizmos'][i].effect.color] += 1
        #     if me['gizmos'][i].type == 'PICK':
        #         a[1][me['gizmos'][i].effect.color] += 1
        #         tmp_giz[4 + me['gizmos'][i].effect.color] += 1
        #     if me['gizmos'][i].type == 'CONVERTER':
        #         a[2][me['gizmos'][i].effect.color] += 1
        #         tmp_giz[8 + me['gizmos'][i].effect.color] += 1
        #     all_giz.extend(tmp_giz)
        # tmp_giz = [0 for i in range(12)]
        # for i in range(12 - len(me['gizmos'])):
        #     all_giz.extend(tmp_giz)
        #
        # for i in range(len(rival['gizmos'])):
        #     tmp_giz = [0 for i in range(12)]
        #     if rival['gizmos'][i].type == 'BUILD':
        #         b[0][rival['gizmos'][i].effect.color] += 1
        #         tmp_giz[rival['gizmos'][i].effect.color] += 1
        #     if rival['gizmos'][i].type == 'PICK':
        #         b[1][rival['gizmos'][i].effect.color] += 1
        #         tmp_giz[4 + rival['gizmos'][i].effect.color] += 1
        #     if rival['gizmos'][i].type == 'CONVERTER':
        #         b[2][rival['gizmos'][i].effect.color] += 1
        #         tmp_giz[8 + rival['gizmos'][i].effect.color] += 1
        #     all_giz.extend(tmp_giz)
        # tmp_giz = [0 for i in range(12)]
        # for i in range(12 - len(rival['gizmos'])):
        #     all_giz.extend(tmp_giz)
        dense_part = dense_part #+ a[0] + a[1] + a[2] + b[0] + b[1] + b[2] + all_giz
        return [
            gen('stage', ob['curr_stage']),
            gen('turn', ob['curr_turn']),
            gen('is_last_turn', ob['is_last_turn']),
            gen('board_red', board_en.count('red')),
            gen('board_black', board_en.count('black')),
            gen('board_blue', board_en.count('blue')),
            gen('board_yellow', board_en.count('yellow')),
            gen('board_giz', lget_id(board_giz_l1, 0)),
            gen('board_giz', lget_id(board_giz_l1, 1)),
            gen('board_giz', lget_id(board_giz_l1, 2)),
            gen('board_giz', lget_id(board_giz_l1, 3)),
            gen('board_giz', lget_id(board_giz_l2, 0)),
            gen('board_giz', lget_id(board_giz_l2, 1)),
            gen('board_giz', lget_id(board_giz_l2, 2)),
            gen('board_giz', lget_id(board_giz_l3, 0)),
            gen('board_giz', lget_id(board_giz_l3, 1)),
            gen('free_pick_num', ob['free_pick_num']),
            gen('free_build', ob['free_build']),
            gen('my_filed', lget_id(me['filed'], 0)),
            gen('my_filed', lget_id(me['filed'], 1)),
            gen('my_filed', lget_id(me['filed'], 2)),
            gen('my_filed', lget_id(me['filed'], 3)),
            gen('my_filed', lget_id(me['filed'], 4)),
            gen('my_giz', lget_id(me['gizmos'], 0)),
            gen('my_giz', lget_id(me['gizmos'], 1)),
            gen('my_giz', lget_id(me['gizmos'], 2)),
            gen('my_giz', lget_id(me['gizmos'], 3)),
            gen('my_giz', lget_id(me['gizmos'], 4)),
            gen('my_giz', lget_id(me['gizmos'], 5)),
            gen('my_giz', lget_id(me['gizmos'], 6)),
            gen('my_giz', lget_id(me['gizmos'], 7)),
            gen('my_giz', lget_id(me['gizmos'], 8)),
            gen('my_giz', lget_id(me['gizmos'], 9)),
            gen('my_giz', lget_id(me['gizmos'], 10)),
            gen('my_giz', lget_id(me['gizmos'], 11)),
            gen('my_giz', lget_id(me['gizmos'], 12)),
            gen('my_giz', lget_id(me['gizmos'], 13)),
            gen('my_giz', lget_id(me['gizmos'], 14)),
            gen('my_giz', lget_id(me['gizmos'], 15)),
            gen('my_red', my_en['red']),
            gen('my_black', my_en['black']),
            gen('my_blue', my_en['blue']),
            gen('my_yellow', my_en['yellow']),
            gen('rival_filed', lget_id(rival['filed'], 0)),
            gen('rival_filed', lget_id(rival['filed'], 1)),
            gen('rival_filed', lget_id(rival['filed'], 2)),
            gen('rival_filed', lget_id(rival['filed'], 3)),
            gen('rival_filed', lget_id(rival['filed'], 4)),
            gen('rival_giz', lget_id(rival['gizmos'], 0)),
            gen('rival_giz', lget_id(rival['gizmos'], 1)),
            gen('rival_giz', lget_id(rival['gizmos'], 2)),
            gen('rival_giz', lget_id(rival['gizmos'], 3)),
            gen('rival_giz', lget_id(rival['gizmos'], 4)),
            gen('rival_giz', lget_id(rival['gizmos'], 5)),
            gen('rival_giz', lget_id(rival['gizmos'], 6)),
            gen('rival_giz', lget_id(rival['gizmos'], 7)),
            gen('rival_giz', lget_id(rival['gizmos'], 8)),
            gen('rival_giz', lget_id(rival['gizmos'], 9)),
            gen('rival_giz', lget_id(rival['gizmos'], 10)),
            gen('rival_giz', lget_id(rival['gizmos'], 11)),
            gen('rival_giz', lget_id(rival['gizmos'], 12)),
            gen('rival_giz', lget_id(rival['gizmos'], 13)),
            gen('rival_giz', lget_id(rival['gizmos'], 14)),
            gen('rival_giz', lget_id(rival['gizmos'], 15)),
            gen('rival_red', rival_en['red']),
            gen('rival_black', rival_en['black']),
            gen('rival_blue', rival_en['blue']),
            gen('rival_yellow', rival_en['yellow']),
        ], dense_part

    def gen_action_feature(self, action: Action):
        feature: list[int] = []
        gen = self.idg.gen_unique_id
        act_type = action['type']
        # feature.append(gen("act_type", act_type))
        if act_type == ActionType.RESEARCH:
            feature.append(gen(act_type, action['level']))
        else:
            feature.append(gen(ActionType.RESEARCH))

        if act_type == ActionType.PICK:
            feature.append(gen(act_type, action['energy']))
        else:
            feature.append(gen(ActionType.PICK))

        if act_type == ActionType.END:
            feature.append(gen(act_type, True))
        else:
            feature.append(gen(ActionType.END))

        if act_type == ActionType.GIVE_UP:
            feature.append(gen(act_type, True))
        else:
            feature.append(gen(ActionType.GIVE_UP))

        if act_type == ActionType.USE_GIZMO:
            feature.append(gen(act_type, action['id']))
        else:
            feature.append(gen(ActionType.USE_GIZMO))

        if act_type == ActionType.FILE:
            feature.append(gen(act_type, action['id']))
        else:
            feature.append(gen(ActionType.FILE))

        if act_type == ActionType.FILE_FROM_RESEARCH:
            feature.append(gen(act_type, action['id']))
        else:
            feature.append(gen(ActionType.FILE_FROM_RESEARCH))

        if act_type == ActionType.BUILD_FOR_FREE:
            feature.append(gen(act_type, action['id']))
        else:
            feature.append(gen(ActionType.BUILD_FOR_FREE))

        if act_type == ActionType.BUILD or \
                act_type == ActionType.BUILD_FROM_FILED or \
                act_type == ActionType.BUILD_FROM_RESEARCH:
            feature.append(gen(act_type, action['id']))
            feature.append(gen('cost_red', action['cost_energy_num']['red']))
            feature.append(
                gen('cost_black', action['cost_energy_num']['black']))
            feature.append(gen('cost_blue', action['cost_energy_num']['blue']))
            feature.append(
                gen('cost_yellow', action['cost_energy_num']['yellow']))
            cost_giz = action['cost_converter_gizmos_id']
            feature.append(gen("cost_giz", lget(cost_giz, 0)))
            feature.append(gen("cost_giz", lget(cost_giz, 1)))
            feature.append(gen("cost_giz", lget(cost_giz, 2)))
            feature.append(gen("cost_giz", lget(cost_giz, 3)))
            feature.append(gen("cost_giz", lget(cost_giz, 4)))
            feature.append(gen("cost_giz", lget(cost_giz, 5)))
            feature.append(gen("cost_giz", lget(cost_giz, 6)))
            feature.append(gen("cost_giz", lget(cost_giz, 7)))
            feature.append(gen("cost_giz", lget(cost_giz, 8)))
            feature.append(gen("cost_giz", lget(cost_giz, 9)))
            feature.append(gen("cost_giz", lget(cost_giz, 10)))
            feature.append(gen("cost_giz", lget(cost_giz, 11)))
            feature.append(gen("cost_giz", lget(cost_giz, 12)))
            feature.append(gen("cost_giz", lget(cost_giz, 13)))
            feature.append(gen("cost_giz", lget(cost_giz, 14)))
            feature.append(gen("cost_giz", lget(cost_giz, 15)))
        else:
            feature.append(gen(ActionType.BUILD))
            feature.append(gen('cost_red'))
            feature.append(gen('cost_black'))
            feature.append(gen('cost_blue'))
            feature.append(gen('cost_yellow'))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
            feature.append(gen("cost_giz"))
        return feature

    def save(self, name='ql.pkl'):
        torch.save(self.state_dict(), name)
