from .common import *


class FindAndDefeatZerglings(RLAgent):
    def __init__(self):
        super().__init__()
        self.file_name = "q_table_3.txt"

        def state1(obs):
            return len(xy_locs(obs_mini(
                obs, PLAYER_RELATIVE) == PLAYER_ENEMY))

        def state2(obs):
            marines = xy_locs(obs_mini(obs, PLAYER_RELATIVE) == PLAYER_SELF)
            xs = np.array([x for (x, _) in marines])
            return np.mean(xs)

        def state3(obs):
            marines = xy_locs(obs_mini(obs, PLAYER_RELATIVE) == PLAYER_SELF)
            ys = np.array([y for (_, y) in marines])
            return np.mean(ys)

        # self.curr_state_func = [state1, state2, state3]
        self.curr_state_func = [state1]
        self.num_state = len(self.curr_state_func)
        # self.arg_bin = [[0, 3, 3], [11, 53, 4], [20, 54, 4]]
        self.arg_bin = [[0, 4, 4]]
        self._scout_path = []
        self._margin_rate = 0.2
        self._cur_scout_dst = (-1, -1)
        self._scout_cnt = 0

        def set_path(obs):
            valid = xy_locs(obs_mini(obs, PATHABLE) == 1)
            xs = [x for (x, _) in valid]
            ys = [y for (_, y) in valid]
            left, right, bottom, top = min(xs), max(xs), max(ys), min(ys)
            x_margin = round((right - left) * self._margin_rate)
            y_margin = round((bottom - top) * self._margin_rate)
            x1, x2 = left + x_margin, right - x_margin
            y1, y2 = top + y_margin, bottom - y_margin
            self._scout_path = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]

        def scout_lt(obs):
            if not self._scout_path:
                set_path(obs)
            return FUNCTIONS.Move_minimap("now", self._scout_path[0])

        def scout_lb(obs):
            if not self._scout_path:
                set_path(obs)
            return FUNCTIONS.Move_minimap("now", self._scout_path[1])

        def scout_rt(obs):
            if not self._scout_path:
                set_path(obs)
            return FUNCTIONS.Move_minimap("now", self._scout_path[2])

        def scout_rb(obs):
            if not self._scout_path:
                set_path(obs)
            return FUNCTIONS.Move_minimap("now", self._scout_path[3])

        def scout(obs):
            marines = xy_locs(obs_mini(obs, PLAYER_RELATIVE) == PLAYER_SELF)
            marine = center(marines)
            if not self._scout_path:
                set_path(obs)
            if self._cur_scout_dst == (-1, -1) or \
                    distance(self._cur_scout_dst, marine) < 3:
                self._scout_cnt = (self._scout_cnt + 1) % 4
                self._cur_scout_dst = self._scout_path[self._scout_cnt]
            return FUNCTIONS.Move_minimap("now", self._cur_scout_dst)

        def attack(obs):
            relative = obs_mini(obs, PLAYER_RELATIVE)
            zerg_pts = xy_locs(relative == PLAYER_ENEMY)
            mrn_pts = xy_locs(relative == PLAYER_SELF)
            if zerg_pts and mrn_pts:
                mrn_pt = np.mean(mrn_pts, axis=0).round()
                dists = distance(zerg_pts, mrn_pt)
                dst_zerg_pt = zerg_pts[int(np.argmin(dists))]
                return FUNCTIONS.Attack_minimap("now", dst_zerg_pt)
            return FUNCTIONS.no_op()

        # self.actions = [attack, scout_lb, scout_lt, scout_rb, scout_rt]
        self.actions = [attack, scout]
        self.num_action = len(self.actions)

        if os.path.exists(self.file_name):
            self.q_table = np.loadtxt(self.file_name)
        else:
            num_bin = 1
            for i in range(self.num_state):
                num_bin *= self.arg_bin[i][2]
            self.q_table = np.random.uniform(
                0, 1, size=(num_bin, len(self.actions)))
            print(self.q_table)
