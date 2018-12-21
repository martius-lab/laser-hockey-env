import sys, math
import numpy as np
import random

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import pyglet
from pyglet import gl

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

VIEWPORT_W = 600
VIEWPORT_H = 400
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
CENTER_X = W/2
CENTER_Y = H/2

RACKETPOLY = [(-5,20),(+5,20),(+5,-20),(-5,-20)]

FORCEMULIPLAYER = 5000
TORQUEMULTIPLAYER = 100

def r_uniform(mini,maxi):
    return random.random()*(maxi-mini) + mini


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
                self.env.winner = 2

    def EndContact(self, contact):
        pass


class LaserHockeyEnv(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    continuous = False
    NORMAL = 0
    TRAIN_SHOOTING = 1
    TRAIN_DEFENCE = 2


    def __init__(self, mode = NORMAL):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.mode = mode

        self.world = Box2D.b2World([0,0])
        self.player1 = None
        self.player2 = None
        self.puck = None
        self.goal_player_1 = None
        self.goal_player_2 = None
        self.world_objects = []
        self.drawlist = []
        self.done = False
        self.winner = 0

        self.timeStep = 1.0 / FPS
        self.time = 0
        self.max_timesteps = 500

        # x pos player one
        # y pos player one
        # angle player one
        # x vel player one
        # y vel player one
        # angular vel player one
        # x player two
        # y player two
        # angle player two
        # y vel player two
        # y vel player two
        # angular vel player two
        # x pos puck
        # y pos puck
        # x vel puck
        # y vel puck
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)

        if self.continuous:
            # linear force in (x,y)-direction and torque
            self.action_space = spaces.Box(-1, +1, (3*2,), dtype=np.float32)
        else:

            self.action_space = spaces.Discrete(6*2)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
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

    def _create_player(self, position, color):
        player = self.world.CreateDynamicBody(
            position=position,
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in RACKETPOLY ]),
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
                shape=circleShape(radius=10/SCALE, pos=(0,0)),
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
                    shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in poly ]),
                    density=0,
                    friction=0.1,
                    categoryBits=0x011,
                    maskBits=0x0011)
            )
            wall.color1 = (0,0,0)
            wall.color2 = (0,0,0)

            return wall

        def _create_decoration():
            objs = []
            objs.append(self.world.CreateStaticBody(
                position=(W/2, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=100/SCALE, pos=(0,0)),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (0.8,0.8,0.8)
            objs[-1].color2 = (0.8,0.8,0.8)

            objs.append(self.world.CreateStaticBody(
                position=(W/2, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=100/SCALE, pos=(0,0)),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (0.8,0.8,0.8)
            objs[-1].color2 = (0.8,0.8,0.8)

            objs.append(self.world.CreateStaticBody(
                position=(W/2-250/SCALE, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=70/SCALE, pos=(0,0)),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (255./255,204./255,191./255)
            objs[-1].color2 = (255./255,204./255,191./255)

            poly = [(0,100),(100,100),(100,-100),(0,-100)]
            objs.append(self.world.CreateStaticBody(
                position=(W/2-240/SCALE, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[ (x/SCALE, y/SCALE) for x, y in poly]),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (1,1,1)
            objs[-1].color2 = (1,1,1)

            objs.append(self.world.CreateStaticBody(
                position=(W/2+250/SCALE, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=70/SCALE, pos=(0,0)),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (255./255,204./255,191./255)
            objs[-1].color2 = (255./255,204./255,191./255)

            poly = [(100,100),(0,100),(0,-100),(100,-100)]
            objs.append(self.world.CreateStaticBody(
                position=(W/2+140/SCALE, H/2),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[ (x/SCALE, y/SCALE) for x, y in poly]),
                    categoryBits = 0x0,
                    maskBits=0x0)
            ))
            objs[-1].color1 = (1,1,1)
            objs[-1].color2 = (1,1,1)

            return objs

        self.world_objects = []

        self.world_objects.extend(_create_decoration())

        poly = [(-250,5), (-250,-5), (250,-5), (250,5)]
        self.world_objects.append(_create_wall((W/2,H - 1), poly))
        self.world_objects.append(_create_wall((W/2,1), poly))

        poly = [(-5,50), (5,50), (5,-50), (-5,-50)]
        self.world_objects.append(_create_wall((W/2-245/SCALE,H-52.5/SCALE-1), poly))
        self.world_objects.append(_create_wall((W/2-245/SCALE,52.5/SCALE+1), poly))

        self.world_objects.append(_create_wall((W/2+245/SCALE,H-52.5/SCALE-1), poly))
        self.world_objects.append(_create_wall((W/2+245/SCALE,52.5/SCALE+1), poly))

        self.drawlist.extend(self.world_objects)

    def _create_goal(self, position, poly):
        goal = self.world.CreateStaticBody(
            position=position,
            angle=0.0,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in poly ]),
                    density=0,
                    friction=0.1,
                    categoryBits=0x0010,
                    maskBits=0x001,
                    isSensor=True),
                fixtureDef(
                    shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in poly ]),
                    density=0,
                    friction=0.1,
                    categoryBits=0x010,
                    maskBits=0x0010)]
        )
        goal.color1 = (1,1,1)
        goal.color2 = (1,1,1)

        return goal



    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.done = False
        self.winner = 0
        self.prev_shaping = None
        self.time = 0

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create world
        self._create_world()

        poly = [(-5,66), (5,66), (5,-66), (-5,-66)]
        self.goal_player_1 = self._create_goal((W/2-245/SCALE,H/2), poly)
        self.goal_player_2 = self._create_goal((W/2+245/SCALE,H/2), poly)

        # Create players
        self.player1 = self._create_player(
            (W / 3, H / 2),
            (1,0,0)
        )
        if self.mode != self.NORMAL:
            self.player2 = self._create_player(
                (2* W / 3 + r_uniform(-W/6, W/6), H/2 + r_uniform(-H/4, H/4)),
                (0,0,1)
            )
        else:
            self.player2 = self._create_player(
                (2* W / 3, H / 2),
                (0,0,1)
            )
        if self.mode == self.NORMAL:
            self.puck = self._create_puck( (W / 2, H / 2 + r_uniform(-H/4, H/4)), (0,0,0) )
        elif self.mode == self.TRAIN_SHOOTING:
            self.puck = self._create_puck((W / 2 - r_uniform(0, W/3),
                                          H / 2 + r_uniform(-H/4, H/4)),  (0,0,0) )
        elif self.mode == self.TRAIN_DEFENCE:
            self.puck = self._create_puck((W / 2 + r_uniform(0, W/3),
                                           H / 2 + 0.9*r_uniform(-H/2, H/2)),  (0,0,0) )
            force = -(self.puck.position - (0, H/2 + r_uniform(-66/SCALE, 66/SCALE)))*self.puck.mass/self.timeStep
            self.puck.ApplyForceToCenter(force, True)


        self.drawlist.extend([self.player1, self.player2, self.puck])

        obs = self._get_obs()

        return obs

    def _apply_action_with_max_speed(self, player, action, max_speed, is_player_one):
        velocity = np.asarray(player.linearVelocity)
        speed = np.sqrt(np.sum((velocity)**2))
        if is_player_one:
            force = action * FORCEMULIPLAYER
        else:
            force = -action * FORCEMULIPLAYER

        if (is_player_one and player.position[0] > CENTER_X) \
           or (not is_player_one and player.position[0] < CENTER_X): # bounce at the center line
            force[0] = 0
            if is_player_one:
                if player.linearVelocity[0] > 0:
                    force[0] = -2*player.linearVelocity[0] * player.mass / self.timeStep
                force[0] += -1*(player.position[0] - CENTER_X) * player.linearVelocity[0] * player.mass / self.timeStep
            else:
                if player.linearVelocity[0] < 0:
                    force[0] = -2*player.linearVelocity[0] * player.mass / self.timeStep
                force[0] += 1*(player.position[0] - CENTER_X) * player.linearVelocity[0] * player.mass / self.timeStep

            player.linearDamping = 10.0
            player.ApplyForceToCenter(force.tolist(), True)
            return

        if (speed < max_speed):
            player.linearDamping = 1.0
            player.ApplyForceToCenter(force.tolist(), True)
        else:
            player.linearDamping = 10.0
            deltaVelocity = self.timeStep * force / player.mass
            if (np.sqrt(np.sum((velocity + deltaVelocity)**2)) < speed):
                player.ApplyForceToCenter(force.tolist(), True)
            else:
                pass

    def _get_obs(self):
        obs = np.hstack([
            self.player1.position-[CENTER_X,CENTER_Y],
            [self.player1.angle],
            self.player1.linearVelocity,
            [self.player1.angularVelocity],
            self.player2.position-[CENTER_X,CENTER_Y],
            [self.player2.angle],
            self.player2.linearVelocity,
            [self.player2.angularVelocity],
            self.puck.position-[CENTER_X,CENTER_Y],
            self.puck.linearVelocity
            ])

        return obs

    def obs_agent_two(self):
        ''' returns the observations for agent two (symmetric mirrored version of agent one)
        '''
        obs = np.hstack([
            -(self.player2.position-[CENTER_X,CENTER_Y]),
            [-self.player2.angle],
            -self.player2.linearVelocity,
            [-self.player2.angularVelocity],
            -(self.player1.position-[CENTER_X,CENTER_Y]),
            [-self.player1.angle],
            -self.player1.linearVelocity,
            [-self.player1.angularVelocity],
            -(self.puck.position-[CENTER_X,CENTER_Y]),
            -self.puck.linearVelocity
            ])

        return obs


    def _compute_reward(self):
        r = 0
        if self.puck.position[0] < CENTER_X + 0.1:
            dist_to_puck = np.sqrt(np.sum(np.asarray(self.player1.position - self.puck.position)**2))
            r -= dist_to_puck*0.001

        if self.winner == 0: # tie
            r += 5
        elif self.winner == 1: # you won
            r += 10
        else: # opponent won
            r -= 10

        return r

    def _get_info(self):
        return dict(
            winner=self.winner
        )

    def step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
        pass

        self._apply_action_with_max_speed(self.player1, action[:2], 10, True)
        self.player1.ApplyTorque(action[2] * TORQUEMULTIPLAYER, True)
        self._apply_action_with_max_speed(self.player2, action[3:5], 10, False)
        self.player2.ApplyTorque(action[5] * TORQUEMULTIPLAYER, True)

        self.world.Step(self.timeStep, 6 * 30, 2 * 30)

        obs = self._get_obs()
        reward = self._compute_reward()
        info = self._get_info()
        if self.time >=self.max_timesteps:
            self.done = True
        self.time += 1

        return obs, reward, self.done, info

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)
            # self.score_label = pyglet.text.Label('0000', font_size=50,
            #                                      x=VIEWPORT_W/2, y=VIEWPORT_H/2, anchor_x='center', anchor_y='center',
            #                                      color=(0, 0, 0, 255))

        # arr = None
        # win = self.viewer.window
        # win.clear()
        # gl.glViewport(0, 0, VIEWPORT_W, VIEWPORT_H)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        # self.score_label.draw()

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class LaserHockeyEnvContinuous(LaserHockeyEnv):
    continuous  = True
