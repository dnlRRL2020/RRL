import os, logging

os.environ["DNLILP_OR_TYPE"] = "expsumlog"
os.environ["DNLILP_AND_TYPE"] = "expsumlog"
 
import pickle 

import sys
sys.path.append('..')

from Lib.logicLayers  import *
from Lib.predCollection import PredCollection
from Lib.BackgroundType2 import  BackgroundType2
from Lib.logicOps import LOP
from Lib.utils import  DotDict
import numpy as np
from time import sleep
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from BoxWorldEnv import *
from collections import deque
 

params = DotDict({})
params.ILP_VALUE=False
params.HARD_CHOICE=False
params.DBL_SOFTMAX=False
params.REMOVE_REP=False
params.RESET_RANDOM=True

params.MAX_EPISODES=58000
params.MAX_STEPS=20
params.EPS_TH=0

params.MAX_MEM_SIZE=20
params.NUM_BOX=5
params.DIM_V=5
params.DIM_H=5

 

params.DISCOUNT_GAMMA=.3
params.REWARD=20
params.PENALTY=-.02
IMG_SIZE = 64

params.MAX_EPISODES=200000
params.LR_ACTOR=.002
params.NORMALIZE_Q = False
params.ST= 4
params.IC_Lambda = 1

optimizer = tf.keras.optimizers.Adam(params.LR_ACTOR )
# optimizer = tfa.optimizers.SWA(optimizer,average_period=4)   

env = BoxWorldEnvImage(params.NUM_BOX,max_episode = params.MAX_STEPS, goal_type='stack buttom blue', reward=params.REWARD,penalty=params.PENALTY,error_penalty=params.PENALTY)

 

class Memory(object):
    def __init__(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []

    def store_transition(self, obs0, act, rwd):
        self.ep_obs.append(obs0)
        self.ep_act.append(act)
        self.ep_rwd.append(float(rwd))

    def covert_to_array(self):
        array_obs = np.stack(self.ep_obs,0)
        array_act = np.array(self.ep_act)
        array_rwd = np.array(self.ep_rwd)
        return array_obs, array_act, array_rwd

    def reset(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []

   
            
    
 
memory = Memory()

 
class ImageToRL(tf.keras.layers.Layer):
    def __init__(self, NUM_BOX,DIM_H,DIM_V,ACT='tanh'): 
        super(ImageToRL, self).__init__()
        self.NUM_BOX = NUM_BOX
        self.DIM_V = DIM_V
        self.DIM_H = DIM_H


        self.ACT = ACT



    def build(self, input_shape):
        self.conv2_1  = tf.keras.layers.Conv2D( 32 ,(3,3), strides=(2,2), padding="valid",  activation = self.ACT)
        # self.bn_1 = tf.keras.layers.BatchNormalization(center=True, scale=True,momentum=0.9)
        self.mp1 = tf.keras.layers.MaxPooling2D(2,2)
        
        self.conv2_2  = tf.keras.layers.Conv2D( 64, (3,3), strides=(2,2), padding="valid",  activation = self.ACT)
        # self.bn_2 = tf.keras.layers.BatchNormalization(center=True, scale=True,momentum=0.9)
        self.mp2 = tf.keras.layers.MaxPooling2D(2,2)
         

        self.fc1 = tf.keras.layers.Dense( self.NUM_BOX , activation='relu')
        
        self.fch = tf.keras.layers.Dense(  self.DIM_H, activation= 'softmax')
        self.fcv = tf.keras.layers.Dense(  self.DIM_V, activation= 'softmax')
        self.fc11 = tf.keras.layers.Dense( 4)
        self.fc12 = tf.keras.layers.Dense( 4)
        
        self.fc21 = tf.keras.layers.Dense( 4)
        self.fc22 = tf.keras.layers.Dense( 4)
    def feat2scores(self, feat,fc):
        m = fc(feat)
        m = tf.nn.softmax(m,1)
        return tf.transpose( m, [0,2,1])

    def call1(self,x):

        x = self.conv2_1(x)
        # x=self.bn_1(x)
        # x = self.mp1(x)


        x = self.conv2_2(x)
        # x=self.bn_2(x)
        x = self.mp2(x)

        x = tf.reshape( x, [-1, x.shape[1]*x.shape[2],  x.shape[3]] )

        x = self.fc1( x ) 

        x = tf.transpose( x, [0,2,1]  )
        posH  = self.fch( x ) 
        posV  = self.fcv( x ) 
 
        
        return posH,posV

    def call(self,x):

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        
       

        sz=x.shape[1]
        x = tf.reshape(x, [-1,x.shape[1]*x.shape[2],x.shape[-1]])
        x = custom_grad(tf.cast( tf.greater_equal(x, .3),tf.float32), x)


        state_x = self.feat2scores( x, self.fc11 )
        state_x = self.fc12( state_x )
        state_x = tf.nn.softmax(state_x,-1) 

        state_y = self.feat2scores( x, self.fc21 )
        state_y = self.fc22( state_y )
        state_y = tf.nn.softmax(state_y,-1) 
        
        return tf.pad(state_x, [[0,0],[1,0],[1,0]]),tf.pad(state_y, [[0,0],[1,0],[1,0]])


def get_predcoll(num_box, dim_v,dim_h):
    
    #define predicates
   
    B = ['%d'%i for i in range(num_box)]
    V = ['%d'%i for i in range(dim_v)]
    H = ['%d'%i for i in range(dim_h)]
    
    
    Constants = dict( {'B':B,'V':V,'H':H})  
    predColl = PredCollection (Constants)
    
    # state predicates
    predColl.add_pred(name='posH',arguments=['B','H'])
    predColl.add_pred(name='posV',arguments=['B','V'])
    
    # extensional
    predColl.add_pred(name='V_0',arguments=['V'])
    predColl.add_pred(name='V_1',arguments=['V'])
    predColl.add_pred(name='V_lt',arguments=['V','V'])
    predColl.add_pred(name='V_inc',arguments=['V','V'])
    
    predColl.add_pred(name='B_same',arguments=['B','B'])
    predColl.add_pred(name='H_same',arguments=['H','H'])
    predColl.add_pred(name='V_same',arguments=['V','V'])
    
    
    predColl.add_pred(name='floor'  ,arguments=['B'])
    predColl.add_pred(name='blue'  ,arguments=['B'])
    

    predColl.add_pred(name='sameCol'  ,arguments=['B','B']).add_fixed_rule(dnf=['posH(A,C), posH(B,C)'],variables=['H']) 
    predColl.add_pred(name='sameRow'  ,arguments=['B','B']).add_fixed_rule(dnf=['posV(A,C), posV(B,C)'],variables=['V']) 
    
    predColl.add_pred(name='above'  ,arguments=['B','B']).add_fixed_rule(dnf=['sameCol(A,B), posV(A,C), posV(B,D), V_lt(D,C)'],variables=['V','V']) 
    predColl.add_pred(name='below'  ,arguments=['B','B']).add_fixed_rule(dnf=['sameCol(A,B), posV(A,C), posV(B,D), V_lt(C,D)'],variables=['V','V']) 
    
    predColl.add_pred(name='lower'  ,arguments=['B','B']).add_fixed_rule(dnf=['posV(A,C), posV(B,D), V_lt(C,D)'],variables=['V','V']) 

    

    predColl.add_pred(name='on'  ,arguments=['B','B']).add_fixed_rule(   dnf=['sameCol(A,B), posV(A,C), posV(B,D), V_inc(D,C)', 'posV(A,C), V_1(C), floor(B)'],variables=['V','V']) 
    
    predColl.add_pred(name='covered'  ,arguments=['B']).add_fixed_rule(dnf=['on(B,A), not floor(A)'],variables=['B']) 

    predColl.add_pred(name='movable'  ,arguments=['B','B']).add_fixed_rule(
         dnf=['not covered(A), not covered(B), not B_same(A,B), not floor(A), not floor(B), not on(A,B), not blue(A), not lower(B,A)'],variables=[]) 


    def Fn():
        return DNFLayer( [14,1],sig=2., and_init=[-2,.1],or_init=[-2,1.1],pta=['movable(A,B)'] )#,) 
    incp= [ p.name for p in predColl.preds]
        
    predColl.add_pred( name='move'  ,arguments=['B','B'],Frules=LOP.and_op()) \
        .add_rule(variables=[  ] , Fn=Fn, inc_preds= incp, exc_conds=[('*','rep1')],arg_funcs=[],Fvar=LOP.or_op(), use_neg=True ) 
        


    predColl.add_pred(name='IC_1_0'  ,arguments=[]).add_fixed_rule(dnf=['posV(A,B), floor(A), not V_0(B)'],variables=['B','V'],Fvar=LOP.or_op_max) 
    # predColl.add_pred(name='IC_2_1'  ,arguments=[]).add_fixed_rule(dnf=['posV(A,B), floor(A)'],variables=['B','V'],Fvar=LOP.or_op_max) 
    predColl.add_pred(name='IC_3_1'  ,arguments=[]).add_fixed_rule(dnf=['posV(A,B), blue(A)'],variables=['B','V'],Fvar=LOP.or_op_max) 
    predColl.add_pred(name='IC_13_1'  ,arguments=[]).add_fixed_rule(dnf=['posH(A,B), blue(A)'],variables=['B','H'],Fvar=LOP.or_op_max) 
    predColl.add_pred(name='IC_4_0'  ,arguments=[]).add_fixed_rule(dnf=['posH(A,B), posH(A,C), not H_same(B,C)'],variables=['B','H','H'],Fvar=LOP.or_op_max) 
    predColl.add_pred(name='IC_5_0'  ,arguments=[]).add_fixed_rule(dnf=['posV(A,B), posV(A,C), not V_same(B,C)'],variables=['B','V','V'],Fvar=LOP.or_op_max) 


         
         


    predColl.initialize_predicates() 


    bg = BackgroundType2( predColl,Constants )
    bg.zero_all()

    bg.add_backgroud('V_0' ,('0',) ) 
    bg.add_backgroud('V_1' ,('1',) ) 
    
    
    bg.add_backgroud('floor' ,('0',) ) 
    bg.add_backgroud('blue' ,('1',) ) 
    

    for i in range(num_box):
        bg.add_backgroud('B_same' ,('%d'%i,'%d'%i,) )     

    for i in range(dim_h):
        bg.add_backgroud('H_same' ,('%d'%i,'%d'%i,) )    
    

    for i in range(dim_v):
        bg.add_backgroud('V_same' ,('%d'%i,'%d'%i,) )    
        for j in range(dim_v):
            if(i<j):
                bg.add_backgroud('V_lt' ,('%d'%i,'%d'%j,) )    
            if(i+1==j):
                bg.add_backgroud('V_inc' ,('%d'%i,'%d'%j,) )    
            
          
            
    bg.compile_bg()
    return predColl, bg 
 

 
 
# env = gym.make("AsterixDeterministic-v4")
# obs0 = env.reset( ) 
# img_bk=obs0.copy()


params.NEG_EXAMPLES_EXPERIENCE=0
memory = Memory()
predColl,bg = get_predcoll(5,5,5)

FC = ForwardChain(predColl)
img2RL = ImageToRL( 4 , 5,5)

CNT=0 

flx= np.zeros( (1,1,5), dtype=np.float32)
flx[0,0,:] = 1
fly=np.zeros( (1,1,5), dtype=np.float32)
fly[0,0,0] = 1

fly=tf.convert_to_tensor(fly, tf.float32)
flx=tf.convert_to_tensor(flx, tf.float32)

@tf.function
def get_logits( image, ST ):
    
    posH,posV = img2RL(image )
    
    
    # posH = tf.pad(posH, [[0,0],[1,0],[1,0]])+ flx[np.newaxis,:,:]
    # posV = tf.pad(posV, [[0,0],[1,0],[1,0]]) + fly[np.newaxis,:,:]
    
    BS=tf.shape(posH )[0]
    X,Inds = bg.make_batch( bs = BS )
    
    fly1 = tf.tile(fly,[BS,1,1] ) 
    flx1 = tf.tile(flx, [BS,1,1] ) 
    posH = tf.concat( (flx1, posH),1)
    posV = tf.concat( (fly1, posV),1)

    X['posH'] = tf.keras.layers.Flatten()(posH)
    X['posV'] = tf.keras.layers.Flatten()(posV)
    
    Xo = FC( X,Inds )
    
    moves = Xo['move']
    
    constraints=[]
    for p in predColl.preds:
        if( p.name.startswith('IC_')):
            if(p.name.endswith('_1')):
                constraints.append( 1.-Xo[p.name])
            else:
                constraints.append( Xo[p.name])
    constraints = tf.stack(constraints,-1)
    return moves*ST, tf.nn.softmax(moves*ST), constraints


@tf.function  (
    autograph=False,
    input_signature=[
        tf.TensorSpec(shape=[None,64,64,3], dtype=tf.float32),
        tf.TensorSpec(shape=[None,1], dtype=tf.float32),
		tf.TensorSpec(shape=[None ], dtype=tf.int32),
    ],
)
def train_op(OBS,Q,ACT):
    
    with tf.GradientTape() as tape:

        # _, calib_state = img2RL(calib_obs/255.) 
        logits,_,constraints = get_logits(OBS, tf.convert_to_tensor(params.ST,tf.float32) )


        # calib_loss = tf.nn.softmax_cross_entropy_with_logits ( labels=calib_state_label, logits=calib_state) 
        # calib_loss = tf.reduce_mean( calib_loss )

     
        # constraint_losses=[]
        # constraint_losses.append( tf.reduce_mean( neg_ent_loss( zeros+1, constraints[0]) ) )
        # constraint_losses.append( tf.reduce_mean( neg_ent_loss( zeros, constraints[1]) ) )
        # constraint_losses.append( tf.reduce_mean( neg_ent_loss( zeros, constraints[2]) ) )
        # constraint_losses.append( 5*tf.reduce_mean( neg_ent_loss( 1+zeros, constraints[3]) ) )
        constraint_losses =  tf.reduce_mean( neg_ent_loss(  tf.zeros_like(constraints),  constraints) )
        
        actor_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ACT) 
        actor_loss = tf.reduce_mean(actor_cross_entropy  * Q[:,0] ) 



        total_loss =  actor_loss +  params.IC_Lambda*constraint_losses #+  tf.add_n(constraint_losses) + 10.1*  calib_loss
        if  FC.losses:
            total_loss+= tf.add_n(FC.losses)*.01


    gradients = tape.gradient(total_loss,FC.variables+img2RL.variables )
    
    gs=[]
    for g  in  gradients:
        if g is not None:
            gs.append(tf.reduce_max(tf.abs(g)))
    optimizer.apply_gradients( zip(gradients,FC.variables+img2RL.variables))
    return total_loss ,gs
 
def compute_q_value( last_value, done, rwd):
        q_value = np.zeros_like(rwd)
        value = 0 if done else last_value
        for t in reversed(range(0, len(rwd))):
            value = value * params.DISCOUNT_GAMMA + rwd[t]
            q_value[t] = value

        # if params.NORMALIZE_Q: 
        #     q_value -= np.mean(q_value)
        #     q_value /= (1.e-5+np.std(q_value))
        return q_value[:, np.newaxis]

def learn():
    obs, act, rwd = memory.covert_to_array()
    q_value = compute_q_value(0, False, rwd)
    q_value[q_value>0]=1.0
    if len(act)>1: 
        total,  grs = train_op(  tf.convert_to_tensor(obs,tf.float32) , tf.convert_to_tensor(q_value,tf.float32) , tf.convert_to_tensor( act.astype(np.int32),tf.int32) )
        print( 'loss  ',np.round(total.numpy(),2),'grs:',[g.numpy() for g in grs] )

    memory.reset()  

def agent_step(obs,ST=None):
    if ST==None:
        ST = params.ST
    _,act,_  = get_logits( tf.convert_to_tensor(obs[np.newaxis],tf.float32), tf.convert_to_tensor(ST,tf.float32) ) 
    s = act.numpy().ravel()
    action = np.random.choice(range(act.shape[1]), p=s)
    return action
 
def env_step(action):
    obs,rwd, done, info = env.step(action)
    # obs =  obs0[24:152,8:152,:] - img_bk[24:152,8:152,:]
    # obs=get_img_all(img)
    return obs,rwd,done,info
 
def testrun(ST):
    env.reset( ) 
    img,rwd, done, _  = env_step(0)
     
    ep=0
    cnt=0
    while(True):
         
        act = agent_step(img[np.newaxis]  ,ST)    
         
        env.render()
        img,rwd, done, info  = env_step(act)
        
        cnt+=1
        ep+= rwd
        if done:
            print( 'test run finished, count ', cnt, '  reward ', ep)
            break
        
class MovingFn:
    def __init__(self,fn,window):
        self.fn = fn
        self.window=fn
        self.data = deque(maxlen=window)
    def add(self,val):
        # if len(self.data)>=self.window:
        #     self.data.popleft()
        self.data.append(val)
        return self.fn(self.data)
    def get(self):
        return self.fn(self.data)


avgScore = MovingFn( np.mean,100 )
avgSuccess = MovingFn( np.mean,100 )
max_cnt = 0
max_reward = 0   


 
for i_episode in range(params.MAX_EPISODES):
    
    obs0 = env.reset( ) 
    ep_rwd = 0

    cnt=0
    while True:
        act  = agent_step(obs0,params.ST)
        obs1, rwd, done, info = env.step(act)
        
        memory.store_transition(obs0, act, rwd)
        ep_rwd += rwd

        obs0 = obs1
        
        if done or cnt>params.MAX_STEPS:
           
            
            if done:
                
                learn()
            
            else:
                learn()
            

            break
            
        cnt+=1
        
    print('Ep: %i' % i_episode, "|Episode reward : %.2f, |average N: %.2f, average success %.2f " %  (ep_rwd,avgScore.add(cnt),avgSuccess.add(done and ep_rwd>=1 )) )

#%%
