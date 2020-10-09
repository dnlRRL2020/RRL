import colorsys
import enum
import numpy as np
import gym

# _WALL_COLOR = 0.0
# _SPACE_COLOR = 0.75
# _AGENT_COLOR = 0.5
# _GEM_COLOR = 1.0

from gym import spaces, logger
id_bk=10
id_wall=11
id_agent=12
id_gem=13
MAX_COLOR=6
class GridWorld(gym.Env):

    '''A simple implementation of Deepmind Box-World. '''

    def __init__(self, max_episode,max_length=4, max_branch_num=4, branch_length=1,rel=False):
        self.max_episode=max_episode
        self.max_length = max_length
        self.max_branch_num = max_branch_num
        self.branch_length = branch_length
        self.rel=rel
        self.room = None
        self.branches = None
        

        self.agent_pos = None
        self.agent_hold = None

        self.viewer=None
        self.relational=None
        self.set_rewards(0,.1,1)

        self.colormap = {}


        self.colormap[id_bk] = (255,255,255)
        self.colormap[id_wall] = (255,255,255)
        self.colormap[id_agent] = (0,0,0)
        self.colormap[id_gem] = (0,255,0)
        
        self.colormap[1] = (255,0,0)
        self.colormap[2] = (128,128,128)
        self.colormap[3] = (0,0,255)
        self.colormap[4] = (255,0,255)
        self.colormap[5] = (255,255,0)
        self.colormap[6] = (0,255,255)
        
        if self.rel:
            self.colormap[id_bk] = (0,255,255)
            self.colormap[id_wall] = (0,254,254)
            self.colormap[id_agent] = (1,0,0)
            self.colormap[id_gem] = (2,128,128)
            
            self.colormap[1] = (3,0,0)
            self.colormap[2] = (4,255,0)
            self.colormap[3] = (5,0,255)
            self.colormap[4] = (6,0,255)
            self.colormap[5] = (7,255,0)
            self.colormap[6] = (8,255,255)
        


        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (14, 14, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete( 144)
        self.set_rewards(0.,1,10.,-.01)
        self.episode=0
    def set_rewards(self, r1,r2,r3,r4=None):
        if r4 is None:
            r4=-r2
        self.r1,self.r2,self.r3,self.r4 = r1, r2, r3,r4
    def step(self, action):
    
        self.episode+=1         
        agent_pos = self.agent_pos.copy()
        next_pos = self.agent_pos.copy()
        a = int(action)
        x = a//12
        y=a%12
        next_pos[0] = x+1
        next_pos[1] = y+1

        reward = self.r1
        done = False

        # if action == Action.U:
        #     next_pos[0] -= 1
        # elif action == Action.L:
        #     next_pos[1] -= 1
        # elif action == Action.D:
        #     next_pos[0] += 1
        # elif action == Action.R:
        #     next_pos[1] += 1

        def valid(r, c):
            if self.isWall(r, c):
                return False
            elif self.isLockedKey(r, c):
                return False
            elif self.isLock(r, c):
                return self.agent_hold is not None and  self.room[r, c]==self.agent_hold

            return True

        
        if valid(*next_pos):
            # reward = 0
            if self.isLock(*next_pos):
                
                if self.agent_hold is not None and  self.room[next_pos[0], next_pos[1]]==self.agent_hold:
                    reward = self.r2
                    self.agent_hold = None
                
            elif self.isGem(*next_pos):
                reward = self.r3
                done = True
            elif self.isLooseKey(*next_pos):
                self.agent_hold = self.room[next_pos[0], next_pos[1]]
                # print('looskey found',self.agent_hold)
                reward = self.r2
                
                
            if self.agent_hold is  None:    
                self.room[0, 0] = id_bk
            else:
                self.room[0, 0] = self.agent_hold

            for i, (root, branch) in enumerate(self.branches):
                    if i>0:
                        if self.room[next_pos[0], next_pos[1]] in branch  or self.room[next_pos[0], next_pos[1]-1] in branch:
                        # print('negative****************')
                            reward = self.r4
                
            self.room[agent_pos[0], agent_pos[1]] = id_bk
            self.room[next_pos[0], next_pos[1]] = id_agent
            self.agent_pos = next_pos


        return self.get_obs(), reward, done or self.episode>=self.max_episode, {}
        return np.copy(self.room), reward, done, None

    def reset(self):
        self.episode=0
        room_size = 14
        channel = 3

        # initialize room
        room = np.ones((room_size, room_size))*id_bk 
        room[0,:]=id_wall
        room[-1,:]=id_wall
        room[:, 0]=id_wall
        room[:, -1] = id_wall

        def gen_pos():
            while True:
                real_size = room_size - 2
                idx = np.random.randint(real_size ** 2)
                row, col = idx // real_size + 1, idx % real_size + 1

                if room[row, col] == id_bk:
                    return row, col

        # set agent position
        row, col = gen_pos()
        room[row, col] = id_agent
        self.agent_pos = [row, col]

        # generate graph
        branches, node_num = self.graph_gen()

        # generate colors
        
        # set gem color
        # colors[branches[0][1][-1]] = (_GEM_COLOR, _GEM_COLOR, _GEM_COLOR)

        # set loose key
        row, col = gen_pos()
        room[row, col] = branches[0][1][0]

        # generate pair
        for j, (root,branch) in enumerate(branches):
            for i, node in enumerate(branch):
                while True:
                    row, col = gen_pos()

                    # check left side is space
                    if room[row, col-1] != id_bk:
                        continue
                    # check left side of pair is space or wall
                    if room[row, col-2] != id_bk and room[row, col-2] != id_wall:
                        continue
                    # check right side of pair is space or wall
                    if room[row, col+1] != id_bk and room[row, col+1] != id_wall:
                        continue

                    break
                # set key in pair
                
                    
                
                
                

                if i == 0 and j>0:
                    # pass
                    room[row, col] = branches[0][1][root]
                    room[row, col-1] = node 

                if i == 0 and j==0:
                    pass
                    # room[row, col-1] = branches[0][1][root]
                    # room[row, col] = node 


                if i >0:
                
                    
                    room[row, col-1] = branch[i] 
                    room[row, col] = branch[i-1] 
                    if branch[i]==id_gem:
                        self.gem_location=(row,col)
                    # if j==0 and node==branch[-1]:
                        # self.gem_location=(row,col)
                        # room[row, col-1] = id_gem

                
                    
                        

                

        self.room = room 
        self.branches = branches

        

        
        return self.get_obs()
        
        return np.copy(self.room)

    def getr(self):
        def all_idx(idx, axis):
            grid = np.ogrid[tuple(map(slice, idx.shape))]
            grid.insert(axis, idx)
            return tuple(grid)
        def onehot_initialization(a):
            ncols = a.max()+1
            out = np.zeros(a.shape + (ncols,), dtype=int)
            out[all_idx(a, axis=2)] = 1
            return out

        room = np.zeros( (14,14) )
        for i in range(14):
            for j in range(14):
                if self.isSpace(i,j):
                    room[i,j]=0
                elif self.isWall(i,j):
                    room[i,j]=0
                elif self.isAgent(i,j):
                    room[i,j]=1
                elif self.isGem(i,j):
                    room[i,j]=2
                else:
                    # if self.room[i,j] in self.colormap:
                    ind = self.colormap[ tuple(self.room[i,j].tolist())] 
                    room[i,j]=3+ind
        self.relational = onehot_initialization( room.astype(int))
        
        self.relational = np.pad(self.relational ,pad_width=[ [0,0],[0,0],[0,8-self.relational.shape[-1]] ],mode='constant' )
        
        
        return np.copy(self.relational)

    def graph_gen(self):
        
        length =  self.max_length #np.random.randint(self.max_length) + 1
        # length =  self.max_length #np.random.randint(self.max_length) + 1
        branch_num =  self.max_branch_num  
        branch_pos = np.arange(0,branch_num)#np.random.choice(length-1, branch_num)
        branch_pos.sort()
        branches = []

        arr = np.random.permutation(MAX_COLOR)
        # arr = np.random.permutation(self.max_length+branch_num*self.branch_length-1)
        
        # add master branch
        branches.append( [0, np.arange(length-1) ] )

        cnt = length -1

        # add sub branches
        for pos in branch_pos:
            branches.append((pos, np.arange(self.branch_length) + cnt))
            cnt += self.branch_length

        # print(branches)
        for b in branches:
            for i,k in enumerate(b[1]):
                # print('env',i,k-1, b.shape, arr.shape)
                b[1][i]=arr[k-1]+1
        branches[0][1] = np.concatenate( (branches[0][1],[id_gem]) )
        return branches, cnt

    def isWall(self, r, c):
        return self.room[r, c] == id_wall

    def isSpace(self, r, c):
        return self.room[r, c] == id_bk

    def isAgent(self, r, c):
        return self.room[r, c] == id_agent

    def isGem(self, r, c):
        return self.room[r, c] == id_gem
        # return ( r== self.gen_location[0] and c== self.gen_location[1] )
        # return np.all(self.room[r, c] == _GEM_COLOR * 255)

    def isLockOrKey(self, r, c):
        if self.isWall(r, c):
            return False
        elif self.isSpace(r, c):
            return False
        elif self.isAgent(r, c):
            return False
        else:
            return True

    def isLock(self, r, c):
        return self.isLockOrKey(r, c) and self.isLockOrKey(r, c - 1)

    def isLockedKey(self, r, c):
        return self.isLockOrKey(r, c) and self.isLockOrKey(r, c + 1)

    def isLooseKey(self, r, c):
        return self.isLockOrKey(r, c) and not self.isLockOrKey(r, c - 1) and not self.isLockOrKey(r, c + 1)

    def toImage(self,scale=16):
        
        output_size = 14 * scale
        output_shape = (output_size, output_size)
        image = np.zeros((*output_shape, 3))

        for y, row in enumerate(self.room):
            for x, col in enumerate(row):
                for dy in range(scale):
                    for dx in range(scale):
                        image[y*scale+dy, x*scale+dx] = self.colormap[ self.room[y, x] ]

        return image.astype(int)
    def render(self, return_rgb_array=False):
        img = self.toImage()
        # if self.viewer is None:
            # from gym.envs.classic_control import rendering
            # self.viewer = rendering.Viewer(img.shape[0], img.shape[1])
            # im = rendering.SimpleImageViewer()
            # self.viewer = rendering.SimpleImageViewer()
        from matplotlib import pyplot as plt
        plt.imshow(img/255, interpolation='nearest')
        plt.show()
        # plt.pause()

        # self.viewer.imshow(img/255.0)
        # return img

    def get_obs(self):
        return self.toImage(1)
        
    
    def pp(self):
        print(self.branches)
        print(self.room)
    def st(self,x,y):
        return self.step( (x-1)*12+ y-1)
class Action(enum.IntEnum):
    U = 0
    L = 1
    D = 2
    R = 3
