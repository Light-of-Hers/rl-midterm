from .common import *


class DefeatRoaches(RLAgent):
    def __init__(self):
        super().__init__()
        self.file_name = "q_table_4.txt"

        def state1(obs):
            enemy = obs_units(obs, PLAYER_ENEMY)[:]
            if len(enemy) < 1:
                return 0
            return enemy[0].health

        def state2(obs):
            enemy = obs_units(obs, PLAYER_ENEMY)[:]
            if len(enemy) < 2:
                return 0
            enemy.sort(key=lambda t: t.health)
            return enemy[1].health

        def state3(obs):
            enemy = obs_units(obs, PLAYER_ENEMY)[:]
            if len(enemy) < 3:
                return 0
            enemy.sort(key=lambda t: t.health)
            return enemy[2].health

        def state4(obs):
            enemy = obs_units(obs, PLAYER_ENEMY)[:]
            if len(enemy) < 4:
                return 0
            enemy.sort(key=lambda t: t.health)
            return enemy[3].health

        self.curr_state_func = [state1, state2, state3, state4]
        self.num_state = len(self.curr_state_func)
        self.arg_bin = [[0, 150, 4], [0, 150, 4], [0, 150, 4], [0, 150, 4]]

        def attack1(obs):
            if not is_available(obs, FUNCTIONS.Attack_screen):
                return FUNCTIONS.select_army("select")
            enemy = obs_units(obs, PLAYER_ENEMY)[:]
            if len(enemy) < 1:
                return FUNCTIONS.no_op()
            enemy.sort(key=lambda t: t.health)
            x, y = enemy[0].x, enemy[0].y
            return FUNCTIONS.Attack_screen("now", (x, y))

        def attack2(obs):
            if not is_available(obs, FUNCTIONS.Attack_screen):
                return FUNCTIONS.select_army("select")
            enemy = obs_units(obs, PLAYER_ENEMY)[:]
            if len(enemy) < 2:
                return attack1(obs)
            enemy.sort(key=lambda t: t.health)
            x, y = enemy[1].x, enemy[1].y
            return FUNCTIONS.Attack_screen("now", (x, y))

        def attack3(obs):
            if not is_available(obs, FUNCTIONS.Attack_screen):
                return FUNCTIONS.select_army("select")
            enemy = obs_units(obs, PLAYER_ENEMY)[:]
            if len(enemy) < 3:
                return attack2(obs)
            enemy.sort(key=lambda t: t.health)
            x, y = enemy[2].x, enemy[2].y
            return FUNCTIONS.Attack_screen("now", (x, y))

        def attack4(obs):
            if not is_available(obs, FUNCTIONS.Attack_screen):
                return FUNCTIONS.select_army("select")
            enemy = obs_units(obs, PLAYER_ENEMY)[:]
            if len(enemy) < 4:
                return attack3(obs)
            enemy.sort(key=lambda t: t.health)
            x, y = enemy[3].x, enemy[3].y
            return FUNCTIONS.Attack_screen("now", (x, y))

        self.actions = [attack1, attack2, attack3, attack4]
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





