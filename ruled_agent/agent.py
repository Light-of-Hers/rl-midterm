from pysc2.agents import base_agent

from .rules import *


class FeatureUnitAgent(base_agent.BaseAgent):
    def __init__(self):
        super(FeatureUnitAgent, self).__init__()
        self.last_reward = 0
        self.max_reward = 0

    def setup(self, obs_spec, action_spec):
        super(FeatureUnitAgent, self).setup(obs_spec, action_spec)
        if "feature_units" not in obs_spec:
            raise Exception("This agent needs feature_units observation.")

    def reset(self):
        if self.episodes:
            self.max_reward = max(self.max_reward,
                                  self.reward - self.last_reward)
            self.last_reward = self.reward
            print("average reward: {}, max reward: {}".format(
                self.reward / self.episodes, self.max_reward))

        super(FeatureUnitAgent, self).reset()


class RuledAgent(FeatureUnitAgent):
    def __init__(self):
        super(RuledAgent, self).__init__()
        self._rules = [default_rule]
        self._act_que = []

    def reset(self):
        super(RuledAgent, self).reset()
        self._act_que = []
        [r.reset() for r in self._rules]

    def step(self, obs):
        super(RuledAgent, self).step(obs)
        if not self._act_que:
            self._act_que = [default_act]
            for r in self._rules:
                if r.match(obs):
                    self._act_que = r.exec()
                    break
        return self._act_que.pop(0)(obs)

    def set_rules(self, *rules):
        self._rules = rules


class MoveToBeacon(RuledAgent):
    def __init__(self):
        super(MoveToBeacon, self).__init__()
        self.set_rules(
            select_once_rule,
            Rule(default_cond,
                 lambda obs: FUNCTIONS.Move_screen(
                     "now", unit2pt(first(obs_units(obs, PLAYER_NEUTRAL))))),
        )


class CollectMineralShards(RuledAgent):
    def __init__(self):
        super(CollectMineralShards, self).__init__()
        self._cur_shard_dst = (-1, -1)

        def select_act(obs):
            marines = obs_units(obs, PLAYER_SELF)
            target = sorted(marines, key=lambda m: int(m.is_selected))[0]
            return FUNCTIONS.select_point("select", unit2pt(target))

        def move_act(obs):
            marines = obs_units(obs, PLAYER_SELF)
            target = next(m for m in marines if m.is_selected)
            shard_pts = [unit2pt(u) for u in obs_units(obs, PLAYER_NEUTRAL)]
            if self._cur_shard_dst in shard_pts:
                shard_pts.remove(self._cur_shard_dst)
            if shard_pts:
                dists = distance(shard_pts, unit2pt(target))
                dst_shard_pt = shard_pts[int(np.argmin(dists))]
                self._cur_shard_dst = dst_shard_pt
                return FUNCTIONS.Move_screen("now", dst_shard_pt)
            else:
                return FUNCTIONS.no_op()

        self.set_rules(Rule(default_cond, [select_act, move_act]))

    def reset(self):
        super(CollectMineralShards, self).reset()
        self._cur_shard_dst = (-1, -1)


class FindAndDefeatZerglings(RuledAgent):
    def __init__(self):
        super(FindAndDefeatZerglings, self).__init__()
        self._cur_scout_dst = (-1, -1)
        self._scout_path = []
        self._cur_scout_que = []
        self._margin_rate = 0.2

        def zerg_in_view_cond(obs):
            return xy_locs(obs_mini(obs, PLAYER_RELATIVE) == PLAYER_ENEMY)

        def attack_closest_zerg_act(obs):
            relative = obs_mini(obs, PLAYER_RELATIVE)
            zerg_pts = xy_locs(relative == PLAYER_ENEMY)
            mrn_pts = xy_locs(relative == PLAYER_SELF)
            if zerg_pts and mrn_pts:
                mrn_pt = np.mean(mrn_pts, axis=0).round()
                dists = distance(zerg_pts, mrn_pt)
                dst_zerg_pt = zerg_pts[int(np.argmin(dists))]
                return FUNCTIONS.Attack_minimap("now", dst_zerg_pt)
            return FUNCTIONS.no_op()

        def scout_act(obs):
            marines = xy_locs(obs_mini(obs, PLAYER_RELATIVE) == PLAYER_SELF)
            marine = center(marines)

            if not self._scout_path:
                valid = xy_locs(obs_mini(obs, PATHABLE) == 1)
                xs = [x for (x, _) in valid]
                ys = [y for (_, y) in valid]
                left, right, bottom, top = min(xs), max(xs), max(ys), min(ys)
                x_margin = round((right - left) * self._margin_rate)
                y_margin = round((bottom - top) * self._margin_rate)
                x1, x2 = left + x_margin, right - x_margin
                y1, y2 = top + y_margin, bottom - y_margin
                self._scout_path = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
            if not self._cur_scout_que:
                self._cur_scout_que = self._scout_path[:]

            if self._cur_scout_dst == (-1, -1) or distance(
                    self._cur_scout_dst, marine) < 3:
                self._cur_scout_dst = self._cur_scout_que.pop(0)

            return FUNCTIONS.Move_minimap("now", self._cur_scout_dst)

        self.set_rules(
            select_once_rule,
            Rule(zerg_in_view_cond, attack_closest_zerg_act),
            Rule(default_cond, scout_act),
        )

    def reset(self):
        super(FindAndDefeatZerglings, self).reset()
        self._cur_scout_dst = (-1, -1)
        self._scout_path = []
        self._cur_scout_que = []


class DefeatRoaches(RuledAgent):
    def __init__(self):
        super(DefeatRoaches, self).__init__()

        self.set_rules(
            Rule(unit_selected_cond,
                 attack_unit_act(ROACH, key=lambda r: (r.health, r.y))),
            Rule(default_cond, select_all_act),
        )


class DefeatZerglingsAndBanelings(RuledAgent):
    def __init__(self):
        super(DefeatZerglingsAndBanelings, self).__init__()

        self.set_rules(
            Rule(and_cond(exist_unit_cond(MARINE, busy=False),
                          exist_unit_cond(BANEL)),
                 [select_unit_act(MARINE, busy=False),
                  attack_unit_act(ZERG)]),
            select_once_rule,
            Rule(unit_selected_cond,
                 attack_unit_act(ZERG, key=lambda z: z.health)),
        )


class CollectMineralsAndGas(RuledAgent):
    def __init__(self):
        super(CollectMineralsAndGas, self).__init__()

        def new_cc_pt(obs):
            cc = first(obs_units(obs, COMMAND_CENTER))
            x, y = unit2pt(cc)
            if x < 42:
                x = x + 2 * cc.radius
            else:
                x = x - 2 * cc.radius
            return x, y

        self.set_rules(
            Rule(and_cond(build_cond(COMMAND_CENTER),
                          unit_cnt_cond(None, 1, COMMAND_CENTER)),
                 scv_op_acts(
                     new_cc_pt, FUNCTIONS.Build_CommandCenter_screen)),
            Rule(and_cond(build_cond(SUPPLY_DEPOT),
                          unit_cnt_cond(None, 2, SUPPLY_DEPOT),
                          unit_cnt_cond(2, None, COMMAND_CENTER),
                          food_balanced),
                 build_supply_depot_acts),
            Rule(exist_unit_cond(SCV, busy=False), harvest_mineral_acts),
            Rule(train_cond(SCV), train_scv_acts),
        )


class BuildMarines(RuledAgent):
    def __init__(self):
        super(BuildMarines, self).__init__()

        build_barracks_acts = scv_op_acts(
            lambda obs: list(product((62, 75), (7, 22, 37, 52)))[
                unit_cnt(obs, BARRACKS)],
            FUNCTIONS.Build_Barracks_screen)

        train_marine_acts = train_acts(
            lambda obs: unit2pt(
                min(obs_units(obs, BARRACKS), default=None,
                    key=lambda br: br.order_length)),
            FUNCTIONS.Train_Marine_quick)

        self.set_rules(
            Rule(exist_unit_cond(SCV, busy=False), harvest_mineral_acts),
            Rule(and_cond(build_cond(SUPPLY_DEPOT),
                          unit_cnt_cond(None, 24, SUPPLY_DEPOT),
                          food_balanced),
                 build_supply_depot_acts),
            Rule(and_cond(build_cond(BARRACKS),
                          unit_cnt_cond(None, 6, BARRACKS)),
                 build_barracks_acts),
            Rule(and_cond(train_cond(MARINE),
                          unit_cnt_cond(7, None, BARRACKS)),
                 train_marine_acts),
            Rule(and_cond(train_cond(SCV),
                          unit_cnt_cond(None, 19, SCV)),
                 train_scv_acts),
        )
