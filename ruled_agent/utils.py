import numpy as np

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

PLAYER_RELATIVE = "player_relative"
PATHABLE = "pathable"

PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
PLAYER_SELF = features.PlayerRelative.SELF
PLAYER_ENEMY = features.PlayerRelative.ENEMY
FOOD_CAP = features.Player.food_cap
FOOD_USED = features.Player.food_used

MARINE = units.Terran.Marine
SCV = units.Terran.SCV
ZERG = units.Zerg.Zergling
BANEL = units.Zerg.Baneling
ROACH = units.Zerg.Roach
COMMAND_CENTER = units.Terran.CommandCenter
BARRACKS = units.Terran.Barracks
SUPPLY_DEPOT = units.Terran.SupplyDepot
MINERAL_FIELD = units.Neutral.MineralField

MINERALS = features.Player.minerals
GAS = features.Player.vespene

FUNCTIONS = actions.FUNCTIONS

COSTS = {
    COMMAND_CENTER: (450, 0),
    SUPPLY_DEPOT: (100, 0),
    BARRACKS: (150, 0),
    SCV: (50, 0),
    MARINE: (50, 0),
}

TRAINING_CAMP = {
    SCV: COMMAND_CENTER,
    MARINE: BARRACKS,
}


def xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


def first(gen):
    return next(iter(gen), None)


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b), axis=-1)


def center(xs):
    return np.mean(xs, axis=0).round()


def obs_mini(obs, attr=None):
    mini = obs.observation.feature_minimap
    return mini if attr is None else getattr(mini, attr)


def obs_screen(obs, attr=None):
    screen = obs.observation.feature_screen
    return screen if attr is None else getattr(screen, attr)


def obs_units(obs, *feats, **kwargs):
    us = obs.observation.feature_units
    for feat in feats:
        if isinstance(feat, features.PlayerRelative):
            us = [u for u in us if u.alliance == feat]
        else:
            us = [u for u in us if u.unit_type == feat]
    if "busy" in kwargs:
        us = [u for u in us if (int(u.order_length) > 0) == kwargs["busy"]]
    return us


def unit_cnt(obs, *feats, **kwargs):
    return len(obs_units(obs, *feats, **kwargs))


def unit2pt(unit):
    return (unit.x, unit.y) if unit is not None else None


def player_info(obs, info):
    return obs.observation.player[info]


def round_pt(obs, pt):
    if pt is None:
        return 1, 1
    x, y = pt
    max_x, max_y = obs_screen(obs, "shape")[1:]
    return max(min(x, max_x - 1), 0), max(min(y, max_y - 1), 0)


def is_available(obs, func):
    return func.id in obs.observation.available_actions


def food_enough(obs, food_needed=1):
    return (player_info(obs, FOOD_CAP) >=
            player_info(obs, FOOD_USED) + food_needed)


def food_balanced(obs):
    return player_info(obs, FOOD_CAP) == player_info(obs, FOOD_USED)


def call_func(obs, func, *args):
    return func(*args) if is_available(obs, func) else FUNCTIONS.no_op()
