from ..utils import *
from itertools import product


class Rule:
    def __init__(self, condition, actions, max_times=-1):
        self._condition = condition
        if not isinstance(actions, (list, tuple)):
            actions = [actions]
        self._actions = actions
        self._max_times = max_times
        self._cur_times = self._max_times

    def reset(self):
        self._cur_times = self._max_times

    def match(self, obs):
        if self._cur_times == 0:
            return False
        return self._condition(obs)

    def exec(self):
        self._cur_times -= 1
        return self._actions[:]


def default_cond(_):
    return True


def default_act(_):
    return FUNCTIONS.no_op()


default_rule = Rule(default_cond, [default_act])


def and_cond(*conds):
    def _cond(obs):
        for c in conds:
            if not c(obs):
                return False
        return True

    return _cond


def or_cond(*conds):
    def _cond(obs):
        for c in conds:
            if c(obs):
                return True
        return False

    return _cond


def not_cond(cond):
    def _cond(obs):
        return not cond(obs)

    return _cond


def unit_selected_cond(obs):
    us = obs_units(obs, PLAYER_SELF)
    return us and us[0].is_selected


def select_all_act(_):
    return FUNCTIONS.select_army("select")


select_once_rule = Rule(default_cond, select_all_act, max_times=1)


def exist_unit_cond(*feats, **kwargs):
    def _cond(obs):
        return obs_units(obs, *feats, **kwargs)

    return _cond


def unit_cnt_cond(fr, to, *feats, **kwargs):
    def _cond(obs):
        cnt = unit_cnt(obs, *feats, **kwargs)
        res = True
        if fr is not None:
            res = res and (cnt >= fr)
        if to is not None:
            res = res and (cnt <= to)
        return res

    return _cond


def select_unit_act(*feats, **kwargs):
    def _act(obs):
        us = obs_units(obs, *feats, **kwargs)
        if not us:
            return FUNCTIONS.no_op()
        return FUNCTIONS.select_point(
            "select", unit2pt(first(obs_units(obs, *feats, **kwargs))))

    return _act


def attack_unit_act(unit_type, key=None):
    def _act(obs):
        us = obs_units(obs, unit_type)
        if not us:
            return FUNCTIONS.no_op()
        us = us if key is None else sorted(us, key=key)
        return FUNCTIONS.Attack_screen("now", unit2pt(first(us)))

    return _act


def build_cond(building):
    minerals_needed, gas_needed = COSTS[building]
    return lambda obs: all([
        player_info(obs, MINERALS) >= minerals_needed,
        player_info(obs, GAS) >= gas_needed
    ])


def scv_op_acts(target_getter, op_func,
                scv_getter=lambda obs: unit2pt(
                    first(obs_units(obs, SCV)))):
    return [
        lambda obs: (
            FUNCTIONS.select_point(
                "select", round_pt(obs, scv_getter(obs)))),
        lambda obs: (
            op_func("now", round_pt(obs, target_getter(obs)))
            if is_available(obs, op_func) else FUNCTIONS.no_op())
    ]


def train_cond(unit):
    mineral_needed, gas_needed = COSTS[unit]
    building = TRAINING_CAMP[unit]
    return lambda obs: all([
        food_enough(obs),
        obs_units(obs, building),
        player_info(obs, MINERALS) >= mineral_needed,
        player_info(obs, GAS) >= gas_needed
    ])


def train_acts(building_getter, train_func):
    return [
        lambda obs: (
            FUNCTIONS.select_point(
                "select", round_pt(obs, building_getter(obs)))),
        lambda obs: (
            train_func("now")
            if is_available(obs, train_func) else FUNCTIONS.no_op())
    ]


build_supply_depot_acts = scv_op_acts(
    lambda _: list(product((20, 28, 36, 42, 50), (6, 12, 20, 48, 56)))[
        unit_cnt(_, SUPPLY_DEPOT)],
    FUNCTIONS.Build_SupplyDepot_screen)

harvest_mineral_acts = scv_op_acts(
    lambda _: unit2pt(first(obs_units(_, MINERAL_FIELD))),
    FUNCTIONS.Harvest_Gather_screen,
    lambda _: unit2pt(first(obs_units(_, SCV, busy=False))))

train_scv_acts = train_acts(
    lambda _: unit2pt(min(obs_units(_, COMMAND_CENTER), default=None,
                          key=lambda _: _.order_length)),
    FUNCTIONS.Train_SCV_quick)
