from .common import *


class DefeatZerglingsAndBanelings(RLAgent):
    def __init__(self):
        super().__init__()
        self.file_name = "q_table_5.txt"

        def state1(obs):
            return len(obs_units(obs, ZERG))

        def state2(obs):
            return len(obs_units(obs, BANEL))
        self.curr_state_func = [state1, state2]
        self.num_state = len(self.curr_state_func)
        self.arg_bin = [[0, 6, 4], [0, 4, 4]]
        
        def marine_attack(obs):
            relative = obs_screen(obs, PLAYER_RELATIVE)
            zerg_pts = xy_locs(relative == PLAYER_ENEMY)
            mrn_pts = xy_locs(relative == PLAYER_SELF)
            if mrn_pts and not is_available(obs, FUNCTIONS.Attack_screen):
                return FUNCTIONS.select_point("select", mrn_pts[0])
            if zerg_pts and mrn_pts:
                mrn_pt = np.mean(mrn_pts, axis=0).round()
                dists = distance(zerg_pts, mrn_pt)
                dst_zerg_pt = zerg_pts[int(np.argmin(dists))]
                return FUNCTIONS.Attack_screen("now", dst_zerg_pt)
            return FUNCTIONS.no_op()

        def army_attack(obs):
            if not is_available(obs, FUNCTIONS.Attack_screen):
                return FUNCTIONS.select_army("select")
            relative = obs_screen(obs, PLAYER_RELATIVE)
            zerg_pts = xy_locs(relative == PLAYER_ENEMY)
            mrn_pts = xy_locs(relative == PLAYER_SELF)
            if zerg_pts and mrn_pts:
                mrn_pt = np.mean(mrn_pts, axis=0).round()
                dists = distance(zerg_pts, mrn_pt)
                dst_zerg_pt = zerg_pts[int(np.argmin(dists))]
                return FUNCTIONS.Attack_screen("now", dst_zerg_pt)
            return FUNCTIONS.no_op()

        self.actions = [army_attack, marine_attack]
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

