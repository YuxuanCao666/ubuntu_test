import logging
import numpy
import random
from gym import spaces
import gym


logger = logging.getLogger(__name__)


class PuzzleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        # 状态空间
        self.states = [1, 2, 3, 4, 5,
                       6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20,
                       21, 22, 23, 24, 25]
        self.x = [75, 125, 175, 225, 275,
                  75, 125, 175, 225, 275,
                  75, 125, 175, 225, 275,
                  75, 125, 175, 225, 275,
                  75, 125, 175, 225, 275]
        self.y = [275, 275, 275, 275, 275,
                  225, 225, 225, 225, 225,
                  175, 175, 175, 175, 175,
                  125, 125, 125, 125, 125,
                  75, 75, 75, 75, 75]

        self.terminate_states = dict()  # 终止状态为字典格式
        self.terminate_states[4] = 1
        self.terminate_states[9] = 1
        self.terminate_states[11] = 1
        self.terminate_states[12] = 1
        self.terminate_states[15] = 1
        self.terminate_states[23] = 1
        self.terminate_states[24] = 1
        self.terminate_states[25] = 1

        self.actions = ['n', 'e', 's', 'w']

        self.rewards = dict();  # 回报的数据结构为字典
        self.rewards['3_e'] = -1.0
        self.rewards['8_e'] = -1.0
        self.rewards['6_s'] = -1.0
        self.rewards['7_s'] = -1.0
        self.rewards['16_n'] = -1.0
        self.rewards['17_n'] = -1.0
        self.rewards['13_w'] = -1.0
        self.rewards['14_n'] = -1.0
        self.rewards['18_s'] = -1.0
        self.rewards['19_s'] = -1.0
        self.rewards['20_s'] = -1.0
        self.rewards['5_w'] = -1.0
        self.rewards['10_w'] = -1.0
        self.rewards['14_e'] = 1.0
        self.rewards['20_n'] = 1.0
        self.rewards['10_s'] = 1.0

        self.t = dict();  # 状态转移的数据格式为字典
        self.t['1_s'] = 6
        self.t['1_e'] = 2
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['2_n'] = 7
        self.t['3_s'] = 7
        self.t['3_w'] = 2
        self.t['3_e'] = 4
        self.t['3_n'] = 8
        self.t['5_s'] = 10
        self.t['5_w'] = 4
        self.t['6_e'] = 7
        self.t['6_n'] = 1
        self.t['6_s'] = 11
        self.t['7_e'] = 8
        self.t['7_n'] = 2
        self.t['7_s'] = 12
        self.t['7_w'] = 6
        self.t['8_e'] = 9
        self.t['8_n'] = 3
        self.t['8_s'] = 13
        self.t['8_w'] = 7
        self.t['10_n'] = 5
        self.t['10_s'] = 15
        self.t['10_w'] = 9
        self.t['13_e'] = 14
        self.t['13_n'] = 8
        self.t['13_s'] = 18
        self.t['13_w'] = 12
        self.t['14_e'] = 15
        self.t['14_n'] = 9
        self.t['14_s'] = 19
        self.t['14_w'] = 13
        self.t['16_e'] = 17
        self.t['16_n'] = 11
        self.t['16_s'] = 21
        self.t['17_e'] = 18
        self.t['17_n'] = 12
        self.t['17_s'] = 22
        self.t['17_w'] = 16
        self.t['18_e'] = 19
        self.t['18_n'] = 13
        self.t['18_s'] = 23
        self.t['18_w'] = 17
        self.t['19_e'] = 20
        self.t['19_n'] = 14
        self.t['19_n'] = 24
        self.t['19_w'] = 18
        self.t['20_n'] = 15
        self.t['20_s'] = 25
        self.t['20_w'] = 24
        self.t['21_e'] = 22
        self.t['21_n'] = 16
        self.t['22_e'] = 23
        self.t['22_n'] = 17
        self.t['22_w'] = 21
        self.gamma = 0.8  # 折扣因子
        self.viewer = None
        self.state = None

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setAction(self, s):
        self.state = s

    def step(self, action):
        # 系统当前状态
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}
        key = "%d_%s" % (state, action)  # 将状态和动作组成字典的键值

        # 状态转移
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state

        is_terminal = False

        if next_state in self.terminate_states:
            is_terminal = True

        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]

        return next_state, r, is_terminal, {}

    def reset(self):
        self.state = self.states[int(random.random() * len(self.states))]
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 300
        screen_height = 300

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # 创建网格世界,共14条线
            self.line1 = rendering.Line((50, 300), (300, 300))
            self.line2 = rendering.Line((50, 250), (200, 250))
            self.line3 = rendering.Line((250, 250), (300, 250))
            self.line4 = rendering.Line((50, 200), (300, 200))
            self.line5 = rendering.Line((50, 150), (300, 150))
            self.line6 = rendering.Line((50, 100), (300, 100))
            self.line7 = rendering.Line((50, 50), (300, 50))
            self.line8 = rendering.Line((50, 50), (50, 300))
            self.line9 = rendering.Line((100, 50), (100, 150))
            self.line10 = rendering.Line((100, 200), (100, 300))
            self.line11 = rendering.Line((150, 50), (150, 300))
            self.line12 = rendering.Line((200, 100), (200, 300))
            self.line13 = rendering.Line((250, 100), (250, 300))
            self.line14 = rendering.Line((300, 100), (300, 300))

            # 创建第一个骷髅
            self.kulo1 = rendering.make_polygon([(200, 200), (250, 200), (250, 300), (200, 300)])
            self.polygontrans = rendering.Transform()
            self.kulo1.add_attr(self.polygontrans)
            self.kulo1.set_color(0, 0, 0)
            # 创建第二个骷髅
            self.kulo2 = rendering.make_polygon([(50, 150), (150, 150), (150, 200), (50, 200)])
            self.polygontrans = rendering.Transform()
            self.kulo2.add_attr(self.polygontrans)
            self.kulo2.set_color(0, 0, 0)
            # 创建第三个骷髅
            self.kulo3 = rendering.make_polygon([(150, 50), (300, 50), (300, 100), (150, 100)])
            self.polygontrans = rendering.Transform()
            self.kulo3.add_attr(self.polygontrans)
            self.kulo3.set_color(0, 0, 0)

            # 创建金条
            self.gold = rendering.make_circle(25)
            self.circletrans = rendering.Transform(translation=(275, 175))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1, 0.9, 0)
            # 创建机器人
            self.robot = rendering.make_circle(25)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)
            self.line12.set_color(0, 0, 0)
            self.line13.set_color(0, 0, 0)
            self.line14.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.line12)
            self.viewer.add_geom(self.line13)
            self.viewer.add_geom(self.line14)

            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.kulo3)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.state is None: return None
        # self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
        self.robotrans.set_translation(self.x[self.state - 1], self.y[self.state - 1])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
