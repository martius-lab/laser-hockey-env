import math
import numpy as np

import Box2D
# noinspection PyUnresolvedReferences
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding, EzPickle

# import pyglet
# from pyglet import gl

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well (Don't touch)

VIEWPORT_W = 600
VIEWPORT_H = 400
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
CENTER_X = W / 2
CENTER_Y = H / 2

RACKETPOLY = [(-5, 20), (+5, 20), (+5, -20), (-5, -20), (-13, -10), (-15, 0), (-13, 10)]

FORCEMULTIPLIER = 5000
TORQUEMULTIPLIER = 200


def dist_positions(p1, p2):
    return np.sqrt(np.sum(np.asarray(p1 - p2) ** 2, axis=-1))


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.goal_player_2 == contact.fixtureA.body or self.env.goal_player_2 == contact.fixtureB.body:
            if self.env.puck == contact.fixtureA.body or self.env.puck == contact.fixtureB.body:
                print('Player 1 scored')
                self.env.done = True
                self.env.winner = 1
        if self.env.goal_player_1 == contact.fixtureA.body or self.env.goal_player_1 == contact.fixtureB.body:
            if self.env.puck == contact.fixtureA.body or self.env.puck == contact.fixtureB.body:
                print('Player 2 scored')
                self.env.done = True
                self.env.winner = -1
        if (contact.fixtureA.body == self.env.player1 or contact.fixtureB.body == self.env.player1) \
            and (contact.fixtureA.body == self.env.puck or contact.fixtureB.body == self.env.puck):
            # print("player 1 contacted the puck")
            self.env.player1_contact_puck = True

    def EndContact(self, contact):
        pass


class LaserHockeyEnv(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'render_fps': FPS
    }

    continuous = False
    NORMAL = 0
    TRAIN_SHOOTING = 1
    TRAIN_DEFENSE = 2

    def __init__(self, mode=NORMAL):
        """ mode is the game mode: NORMAL, TRAIN_SHOOTING, TRAIN_DEFENSE,
        it can be changed later using the reset function
        """
        EzPickle.__init__(self)
        self.seed()
        self.screen = None
        self.clock = None
        self.surf = None
        self.isopen = True
        self.mode = mode

        self.world = Box2D.b2World([0, 0])
        self.player1 = None
        self.player2 = None
        self.puck = None
        self.goal_player_1 = None
        self.goal_player_2 = None
        self.world_objects = []
        self.drawlist = []
        self.done = False
        self.winner = 0
        self.one_starts = True  # player one starts the game (alternating)

        self.max_puck_speed = 20

        self.timeStep = 1.0 / FPS
        self.time = 0
        self.max_timesteps = 500

        self.closest_to_goal_dist = 1000

        # 0  x pos player one
        # 1  y pos player one
        # 2  sin(angle player one)
        # 3  cos(angle player one)
        # 4  x vel player one
        # 5  y vel player one
        # 6  angular vel player one
        # 7  x player two
        # 8  y player two
        # 9  sin(angle player two)
        # 10 cos(angle player two)
        # 11 y vel player two
        # 12 y vel player two
        # 13 angular vel player two
        # 14 x pos puck
        # 15 y pos puck
        # 16 x vel puck
        # 17 y vel puck
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)

        # linear force in (x,y)-direction and torque
        self.action_space = spaces.Box(-1, +1, (3 * 2,), dtype=np.float32)

        # see discrete_to_continous_action()
        self.discrete_action_space = spaces.Discrete(7)

        self.reset(self.one_starts)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self._seed = seed
        return [seed]

    def _destroy(self):
        if self.player1 is None: return
        self.world.contactListener = None
        self.world.DestroyBody(self.player1)
        self.player1 = None
        self.world.DestroyBody(self.player2)
        self.player2 = None
        self.world.DestroyBody(self.puck)
        self.puck = None
        self.world.DestroyBody(self.goal_player_1)
        self.goal_player_1 = None
        self.world.DestroyBody(self.goal_player_2)
        self.goal_player_2 = None
        for obj in self.world_objects:
            self.world.DestroyBody(obj)
        self.world_objects = []
        self.drawlist = []

    def r_uniform(self, mini, maxi):
        return self.np_random.uniform(mini, maxi, 1)[0]

    def _create_player(self, position, color, is_player_two):
        player = self.world.CreateDynamicBody(
            position=position,
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(-x / SCALE if is_player_two else x / SCALE, y / SCALE)
                                             for x, y in RACKETPOLY]),
                density=200.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x011,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        player.color1 = color
        player.color2 = color
        player.linearDamping = 1.0
        player.anguarDamping = 1.0

        return player

    def _create_puck(self, position, color):
        puck = self.world.CreateDynamicBody(
            position=position,
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=10 / SCALE, pos=(0, 0)),
                density=10.0,
                friction=0.1,
                categoryBits=0x001,
                maskBits=0x0010,  # collide only with ground
                restitution=0.95)  # 0.99 bouncy
        )
        puck.color1 = color
        puck.color2 = color
        puck.linearDamping = 0.05

        return puck

    def _create_world(self):
        def _create_wall(position, poly):
            wall = self.world.CreateStaticBody(
                position=position,
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in poly]),
                    density=0,
                    friction=0.1,
                    categoryBits=0x011,
                    maskBits=0x0011)
            )
            wall.color1 = (0, 0, 0)
            wall.color2 = (0, 0, 0)

            return wall

        def _create_decoration():
            objs = []
            objs.append(self.world.CreateStaticBody(
                position=(W / 2, H / 2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=100 / SCALE, pos=(0, 0)),
                    categoryBits=0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (204, 204, 204)
            objs[-1].color2 = (204, 204, 204)

            objs.append(self.world.CreateStaticBody(
                position=(W / 2, H / 2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=100 / SCALE, pos=(0, 0)),
                    categoryBits=0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (204, 204, 204)
            objs[-1].color2 = (204, 204, 204)

            objs.append(self.world.CreateStaticBody(
                position=(W / 2 - 250 / SCALE, H / 2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=70 / SCALE, pos=(0, 0)),
                    categoryBits=0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (255, 204, 191)
            objs[-1].color2 = (255, 204, 191)

            poly = [(0, 100), (100, 100), (100, -100), (0, -100)]
            objs.append(self.world.CreateStaticBody(
                position=(W / 2 - 240 / SCALE, H / 2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in poly]),
                    categoryBits=0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (255, 255, 255)
            objs[-1].color2 = (255, 255, 255)

            objs.append(self.world.CreateStaticBody(
                position=(W / 2 + 250 / SCALE, H / 2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=70 / SCALE, pos=(0, 0)),
                    categoryBits=0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (255, 204, 191)
            objs[-1].color2 = (255, 204, 191)

            poly = [(100, 100), (0, 100), (0, -100), (100, -100)]
            objs.append(self.world.CreateStaticBody(
                position=(W / 2 + 140 / SCALE, H / 2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in poly]),
                    categoryBits=0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (255, 255, 255)
            objs[-1].color2 = (255, 255, 255)

            return objs

        self.world_objects = []

        self.world_objects.extend(_create_decoration())

        poly = [(-250, 5), (-250, -5), (250, -5), (250, 5)]
        self.world_objects.append(_create_wall((W / 2, H - 1), poly))
        self.world_objects.append(_create_wall((W / 2, 1), poly))

        poly = [(-5, 50), (5, 50), (5, -50), (-5, -50)]
        self.world_objects.append(_create_wall((W / 2 - 245 / SCALE, H - 52.5 / SCALE - 1), poly))
        self.world_objects.append(_create_wall((W / 2 - 245 / SCALE, 52.5 / SCALE + 1), poly))

        self.world_objects.append(_create_wall((W / 2 + 245 / SCALE, H - 52.5 / SCALE - 1), poly))
        self.world_objects.append(_create_wall((W / 2 + 245 / SCALE, 52.5 / SCALE + 1), poly))

        self.drawlist.extend(self.world_objects)

    def _create_goal(self, position, poly):
        goal = self.world.CreateStaticBody(
            position=position,
            angle=0.0,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in poly]),
                    density=0,
                    friction=0.1,
                    categoryBits=0x0010,
                    maskBits=0x001,
                    isSensor=True),
                fixtureDef(
                    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in poly]),
                    density=0,
                    friction=0.1,
                    categoryBits=0x010,
                    maskBits=0x0010)]
        )
        goal.color1 = (255, 255, 255)
        goal.color2 = (255, 255, 255)

        return goal

    def reset(self, one_starting=None, mode=None):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.done = False
        self.winner = 0
        self.prev_shaping = None
        self.time = 0
        if mode is not None and mode in [self.NORMAL, self.TRAIN_SHOOTING, self.TRAIN_DEFENSE]:
            self.mode = mode

        if self.mode == self.NORMAL:
            if one_starting is not None:
                self.one_starts = one_starting
            else:
                self.one_starts = not self.one_starts
        self.closest_to_goal_dist = 1000
        self.player1_contact_puck = False

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create world
        self._create_world()

        poly = [(-5, 66), (5, 66), (5, -66), (-5, -66)]
        self.goal_player_1 = self._create_goal((W / 2 - 245 / SCALE, H / 2), poly)
        self.goal_player_2 = self._create_goal((W / 2 + 245 / SCALE, H / 2), poly)

        # Create players
        self.player1 = self._create_player(
            (W / 5, H / 2),
            (204, 0, 0),
            False
        )
        if self.mode != self.NORMAL:
            self.player2 = self._create_player(
                (4 * W / 5 + self.r_uniform(-W / 3, W / 6), H / 2 + self.r_uniform(-H / 4, H / 4)),
                (0, 0, 204),
                True
            )
        else:
            self.player2 = self._create_player(
                (4 * W / 5, H / 2),
                (0, 0, 204),
                True
            )
        if self.mode == self.NORMAL or self.mode == self.TRAIN_SHOOTING:
            if self.one_starts or self.mode == self.TRAIN_SHOOTING:
                self.puck = self._create_puck((W / 2 - self.r_uniform(H / 8, H / 4),
                                               H / 2 + self.r_uniform(-H / 8, H / 8)), (0, 0, 0))
            else:
                self.puck = self._create_puck((W / 2 + self.r_uniform(H / 8, H / 4),
                                               H / 2 + self.r_uniform(-H / 8, H / 8)), (0, 0, 0))
        elif self.mode == self.TRAIN_DEFENSE:
            self.puck = self._create_puck((W / 2 + self.r_uniform(0, W / 3),
                                           H / 2 + 0.9 * self.r_uniform(-H / 2, H / 2)), (0, 0, 0))
            force = -(self.puck.position - (
            0, H / 2 + self.r_uniform(-66 / SCALE, 66 / SCALE))) * self.puck.mass / self.timeStep
            self.puck.ApplyForceToCenter(force, True)

        self.drawlist.extend([self.player1, self.player2, self.puck])

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _apply_action_with_max_speed(self, player, action, max_speed, is_player_one):
        velocity = np.asarray(player.linearVelocity)
        speed = np.sqrt(np.sum((velocity) ** 2))
        if is_player_one:
            force = action * FORCEMULTIPLIER
        else:
            force = -action * FORCEMULTIPLIER

        if (is_player_one and player.position[0] > CENTER_X) \
            or (not is_player_one and player.position[0] < CENTER_X):  # bounce at the center line
            force[0] = 0
            if is_player_one:
                if player.linearVelocity[0] > 0:
                    force[0] = -2 * player.linearVelocity[0] * player.mass / self.timeStep
                force[0] += -1 * (player.position[0] - CENTER_X) * player.linearVelocity[
                    0] * player.mass / self.timeStep
            else:
                if player.linearVelocity[0] < 0:
                    force[0] = -2 * player.linearVelocity[0] * player.mass / self.timeStep
                force[0] += 1 * (player.position[0] - CENTER_X) * player.linearVelocity[0] * player.mass / self.timeStep

            player.linearDamping = 10.0
            player.ApplyForceToCenter(force.tolist(), True)
            return

        if speed < max_speed:
            player.linearDamping = 1.0
            player.ApplyForceToCenter(force.tolist(), True)
        else:
            player.linearDamping = 10.0
            deltaVelocity = self.timeStep * force / player.mass
            if np.sqrt(np.sum((velocity + deltaVelocity) ** 2)) < speed:
                player.ApplyForceToCenter(force.tolist(), True)
            else:
                pass

    def _get_obs(self):
        obs = np.hstack([
            self.player1.position - [CENTER_X, CENTER_Y],
            [math.sin(self.player1.angle), math.cos(self.player1.angle)],
            self.player1.linearVelocity,
            [self.player1.angularVelocity],
            self.player2.position - [CENTER_X, CENTER_Y],
            [math.sin(self.player2.angle), math.cos(self.player2.angle)],
            self.player2.linearVelocity,
            [self.player2.angularVelocity],
            self.puck.position - [CENTER_X, CENTER_Y],
            self.puck.linearVelocity
        ])

        return obs

    def obs_agent_two(self):
        """ returns the observations for agent two (symmetric mirrored version of agent one)
        """
        obs = np.hstack([
            -(self.player2.position - [CENTER_X, CENTER_Y]),
            [math.sin(-self.player2.angle), math.cos(-self.player2.angle)],
            -self.player2.linearVelocity,
            [-self.player2.angularVelocity],
            -(self.player1.position - [CENTER_X, CENTER_Y]),
            [math.sin(-self.player1.angle), math.cos(-self.player1.angle)],
            -self.player1.linearVelocity,
            [-self.player1.angularVelocity],
            -(self.puck.position - [CENTER_X, CENTER_Y]),
            -self.puck.linearVelocity
        ])

        return obs

    def _compute_reward(self):
        r = 0

        if self.done:
            if self.winner == 0:  # tie
                r += 0
            elif self.winner == 1:  # you won
                r += 10
            else:  # opponent won
                r -= 10

        return r

    def _get_info(self):
        # different proxy rewards:
        # Proxy reward for being close to puck in the own half
        reward_closeness_to_puck = 0
        if self.puck.position[0] < CENTER_X:
            dist_to_puck = dist_positions(self.player1.position, self.puck.position)
            max_dist = 10.
            max_reward = -5.  # max (negative) reward through this proxy
            factor = max_reward / (max_dist * self.max_timesteps / 2)
            reward_closeness_to_puck += dist_to_puck * factor  # Proxy reward for being close to puck in the own half
        # Proxy reward: touch puck
        reward_touch_puck = 0.
        if self.player1_contact_puck:
            reward_touch_puck = 1.

        # puck is flying in the right direction
        reward_puck_direction = 0
        max_reward = 1.
        factor = max_reward / (self.max_timesteps * self.max_puck_speed)
        reward_puck_direction = self.puck.linearVelocity[0] * factor  # Puck flies right is good and left not

        return {"winner": self.winner,
                "reward_closeness_to_puck": reward_closeness_to_puck,
                "reward_touch_puck": reward_touch_puck,
                "reward_puck_direction": reward_puck_direction,
                }

    def set_state(self, state):
        """ function to revert the state of the environment to a previous state (observation)"""
        self.player1.position = (state[[0, 1]] + [CENTER_X, CENTER_Y]).tolist()
        self.player1.angle = math.atan2(state[2], state[3])
        self.player1.linearVelocity = [state[4], state[5]]
        self.player1.angularVelocity = state[6]
        self.player2.position = (state[[7, 8]] + [CENTER_X, CENTER_Y]).tolist()
        self.player2.angle = math.atan2(state[9], state[10])
        self.player2.linearVelocity = [state[11], state[12]]
        self.player2.angularVelocity = state[13]
        self.puck.position = (state[[14, 15]] + [CENTER_X, CENTER_Y]).tolist()
        self.puck.linearVelocity = [state[16], state[17]]

    def _limit_puck_speed(self):
        puck_speed = np.sqrt(self.puck.linearVelocity[0] ** 2 + self.puck.linearVelocity[1] ** 2)
        if puck_speed > self.max_puck_speed:
            self.puck.linearDamping = 10.0
        else:
            self.puck.linearDamping = 0.05
        self.puck.angularSpeed = 0

    def discrete_to_continous_action(self, discrete_action):
        ''' converts discrete actions into continuous ones (for each player)
        The actions allow only one operation each timestep, e.g. X or Y or angle change.
        This is surely limiting. Other discrete actions are possible
        Action 0: do nothing
        Action 1: -1 in x
        Action 2: 1 in x
        Action 3: -1 in y
        Action 4: 1 in y
        Action 5: -1 in angle
        Action 6: 1 in angle
        '''
        action_cont = [(discrete_action == 1) * -1 + (discrete_action == 2) * 1,  # player x
                       (discrete_action == 3) * -1 + (discrete_action == 4) * 1,  # player y
                       (discrete_action == 5) * -1 + (discrete_action == 6) * 1]  # player angle

        return action_cont

    def step(self, action):
        action = np.clip(action, -1, +1).astype(np.float32)

        self._apply_action_with_max_speed(self.player1, action[:2], 10, True)
        # add her func to limit angle
        self.player1.ApplyTorque(action[2] * TORQUEMULTIPLIER, True)
        self._apply_action_with_max_speed(self.player2, action[3:5], 10, False)
        self.player2.ApplyTorque(-action[5] * TORQUEMULTIPLIER, True)

        self._limit_puck_speed()
        self.player1_contact_puck = False

        self.world.Step(self.timeStep, 6 * 30, 2 * 30)

        obs = self._get_obs()
        if self.time >= self.max_timesteps:
            self.done = True

        reward = self._compute_reward()
        info = self._get_info()

        self.closest_to_goal_dist = min(self.closest_to_goal_dist,
                                        dist_positions(self.puck.position, (W, H / 2)))
        self.time += 1
        return obs, reward, self.done, False, info

    def render(self, mode='human'):
        if mode is None:
            gym.logger.warn(
                "the render method needs a rendering mode"
            )
            return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(self.surf, radius=f.shape.radius * SCALE, width=0,
                                       center=trans * f.shape.pos * SCALE, color=obj.color1)
                    pygame.draw.circle(self.surf, radius=f.shape.radius * SCALE, width=2,
                                       center=trans * f.shape.pos * SCALE, color=obj.color2)
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, points=path, color=obj.color1, width=0)
                    path.append(path[0])
                    pygame.draw.polygon(self.surf, points=path, color=obj.color2, width=2)

        # self.score_label.draw()
        self.surf = pygame.transform.flip(self.surf, False, True)

        if mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

class BasicOpponent():
    def __init__(self):
        self.mode = 0

    def act(self, obs, verbose=False):
        alpha = math.atan2(obs[2], obs[3])
        p1 = np.asarray([obs[0], obs[1], alpha])
        v1 = np.asarray(obs[4:7])
        puck = np.asarray(obs[14:16])
        puckv = np.asarray(obs[16:18])
        # print(p1,v1,puck,puckv)
        target_pos = p1[0:2]
        target_angle = p1[2]

        time_to_break = 0.1
        kp = 10
        kd = 0.5

        # if ball flies towards our goal or very slowly away: try to catch it
        if puckv[0] < 1.0:
            dist = np.sqrt(np.sum((p1[0:2] - puck) ** 2))
            # Am I behind the ball?
            if p1[0] < puck[0] and abs(p1[1] - puck[1]) < 1.0:
                # Go and kick
                target_pos = [puck[0] + 0.2, puck[1] + puckv[1] * dist * 0.1]
                target_angle = np.random.uniform(-0.5, 0.5)  # calc proper angle here
            else:
                # get behind the ball first
                target_pos = [-7, puck[1]]
                target_angle = 0
        else:  # go in front of the goal
            target_pos = [-7, 0]
            target_angle = 0

        target = np.asarray([target_pos[0], target_pos[1], target_angle])
        # use PD control to get to target
        error = target - p1
        need_break = abs((error / (v1 + 0.01))) < [time_to_break, time_to_break, time_to_break * 10]
        if verbose:
            print(error, abs(error / (v1 + 0.01)), need_break)

        return np.clip(error * [kp, kp, kp / 2] - v1 * need_break * [kd, kd, kd], -1, 1)


# class HumanOpponent()
# use  HumanOpponent of HockeyEnv

from gymnasium.envs.registration import register

try:
    register(
        id='LaserHockey-v0',
        entry_point='laser_hockey_env.laser_hockey_env:LaserHockeyEnv',
    )
except Exception as e:
    print(e)
