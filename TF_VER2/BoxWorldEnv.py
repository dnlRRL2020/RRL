import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pyglet

'''
action: 0, 1, ..., num_box*(num_box-1)-1
Box index: 0, 1, ..., num_box-1

Example

    If you want to move box 3 onto box 2, the action 
    is 3 * num_box + 2.

'''

class BoxWorldEnvImage(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    #goal types:  stack, stack top blue, stack buttom blue
    def __init__(self, num_box,max_episode,goal_type='stack',reward=1.0, penalty=-.1,error_penalty=-.1):
        num_box-=1
        self.num_box = num_box
        
        self.max_episode=max_episode
        #self.action_space = spaces.Discrete(  (self.num_box+1) * (self.num_box+1) )
        #self.observation_space = spaces.Discrete( (self.num_box+1) * (self.num_box+1))
        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (64, 64, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete( (num_box+1)**2)
        self.goal_type=goal_type
        self.seed()
        self.viewer = None
        self.boxtrans_set = []
        
        # init: all boxes lying on the floor
        
        self.init_state = np.zeros( [2,num_box],dtype=int)
        for i in range(num_box):
            self.init_state[1,i]=1
            self.init_state[0,i]=i+1

        self.state = self.init_state.copy()
        
        self.color_rgb = [[255,255,255], [0,0,255], [0,255,0],  [255,0,0], [0,255,255], [0,0,0], [128,128,128]]
        self.penalty=penalty
        self.error_penalty=error_penalty
        self.reward=reward
        self.episode=0

    def getOn(self):
        on = np.zeros( (self.num_box+1,self.num_box+1) )
        for i in range(self.num_box):
            if self.state[1,i]==1:
                on[i+1,0]=1
            for j in range(self.num_box):
                if self.state[0,i]==self.state[0,j] and self.state[1,i]==self.state[1,j]+1:
                    on[i+1,j+1]=1
        return on

    def is_goal(self):

        if self.goal_type=="stack":
            
            no_on_the_floor = np.sum( self.state[1,:]==1 )
            if no_on_the_floor == 1:
                return True
            else:
                return False
    
        if self.goal_type=="stack top blue":
            no_on_the_floor = np.sum( self.state[1,:]==1 )
            if no_on_the_floor == 1:
                if self.state[1,0] == self.num_box:
                    return True
                else:
                    return False
            else:
                return False
        
        if self.goal_type=="stack buttom blue":
            no_on_the_floor = np.sum( self.state[1,:]==1 )
            if no_on_the_floor == 1:
                if self.state[1,0] == 1:
                    return True
                else:
                    return False
            else:
                return False
        
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.episode+=1
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        from_box = int(action/(self.num_box+1))
        to_box = action % (self.num_box+1)
        # print('from:{}, to:{}'.format(from_box,to_box))
        # if  

        onstate = self.getOn()
        to_covered = ( np.sum( onstate[:,to_box])>0 )
        from_covered = ( np.sum( onstate[:,from_box])>0 )
        
        
        if from_box == 0  or  from_box==to_box or  ( to_box!=0 and to_covered) or from_covered or onstate[from_box,to_box]>0:
            return self.get_obs(),self.error_penalty, self.is_goal() or  self.episode>=self.max_episode, {}


        if to_box==0:
            notfound=True
            for i in range(self.num_box):
                if np.sum( self.state[0,:]==i+1) == 0: #empty space
                    self.state[0,from_box-1]=i+1
                    self.state[1,from_box-1]=1
                    notfound=False
                    break
            if notfound:
                print('weird error')
                return self.get_obs(),self.error_penalty, self.is_goal() or  self.episode>=self.max_episode, {}
   
        else:
            self.state[0,from_box-1]=self.state[0,to_box-1]
            self.state[1,from_box-1]=self.state[1,to_box-1]+1

        
        if self.is_goal():
            return self.get_obs(),self.reward, True, {}
        else:
           
            return self.get_obs(),self.penalty,  self.episode>=self.max_episode, {}
 
    def get_random_state(self):
        random_state = np.zeros_like( self.state )
        
        xs=np.random.permutation(self.num_box)
        for i in range(self.num_box):
            #x = np.random.randint(low=1,high=5)
            #y = sum(random_state[0,:]==x)
            random_state[0,i] = xs[i]+1
            random_state[1,i] = 1
            
        return random_state
            
            
            
            
        
         

    def reset(self,random=True,allow_goal=False):
        self.episode=0
        if not random:
            self.state = self.init_state.copy()
            return self.get_obs()
        
        else:
            
            self.state = self.get_random_state()
            return self.get_obs()
        
            print('not implemented')
            exit(0)


        self.state = self.get_random_state()
        
        if not allow_goal:
            while self.is_goal():
                self.state = self.get_random_state()
        
        return self.get_obs()
        

    def render(self, mode='human', show_img=False):
        screen_width = 128
        screen_height = 128

        #carty = 100 # TOP OF CART
        boxwidth = 15.0
        boxheight = 15.0
        leftmost_x = 20
        lowmost_y = 20
        x_space =20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.boxtrans_set = []
            self.floor = rendering.Line((0, lowmost_y), (screen_width, lowmost_y))
            self.floor.set_color(.5,.5,.8)
            self.viewer.add_geom(self.floor)
            
            
            for k in range(self.num_box):
                l, r, t, b = 0, boxwidth, boxheight, 0
                box = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                box.set_color(self.color_rgb[k+1][0]/255.,self.color_rgb[k+1][1]/255.,self.color_rgb[k+1][2]/255.)
               
                self.boxtrans_set.append(rendering.Transform())
                box.add_attr(self.boxtrans_set[-1])
                self.viewer.add_geom(box)

            # self.score_label = pyglet.text.Label('AAA', font_size=36,
            # x=20, y=screen_height*2.5/40.00, anchor_x='left', anchor_y='center',
            # color=(255,255,255,255))
            # self.score_label.draw
        
        if self.state is None: return None
        
        
        for k in range(self.num_box):
            self.boxtrans_set[k].set_translation( (self.state[0,k]-1) * x_space + leftmost_x, (self.state[1,k]-1) * boxheight + lowmost_y)

        #pyglet.app.run()
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def get_obs(self):
        return self.render(mode='rgb_array')[::2,::2,:] 
