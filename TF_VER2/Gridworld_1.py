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
import matplotlib.pyplot as plt
from GridWorldEnv import GridWorld
from collections import deque
def showimg(img):

    plt.imshow(img)
    plt.show()
    
COLOR_COUNT=10
IMG_SIZE = 14
IMG_DEPTH=3


params = DotDict({})
params.ILP_VALUE=False
params.HARD_CHOICE=False
params.DBL_SOFTMAX=False
params.REMOVE_REP=False
params.RESET_RANDOM=True

params.MAX_EPISODES=58000
params.MAX_STEPS=20
params.EPS_TH=0

params.MAX_STEPS=50
params.EPS_TH=0
 
params.MAX_MEM_SIZE=50 
params.NUM_BOX=4
params.BRANCH_COUNT = 1

  

params.MAX_EPISODES=200000
params.LR_ACTOR=.002
params.NORMALIZE_Q = False
params.ST= 10
params.IC_Lambda = 2.5

optimizer = tf.keras.optimizers.Adam(params.LR_ACTOR )
# optimizer = tfa.optimizers.SWA(optimizer,average_period=4)    

env = GridWorld(max_episode = params.MAX_STEPS, max_branch_num=params.BRANCH_COUNT)

env.set_rewards(0,1.,10.,-.1)
env.max_length = 3

params.LR_ACTOR= .001
 
params.DISCOUNT_GAMMA=.2


 

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
    def __init__(self  ,ACT='tanh'): 
        super(ImageToRL, self).__init__()
         

        



    def build(self, input_shape):
         
        self.fc1 = tf.keras.layers.Dense( 300 , activation='relu')
        self.fc2 = tf.keras.layers.Dense( 10 , activation='softmax')

       

    def call(self,x):

        y =  self.fc2( self.fc1(x/255.))  

        has_key = y[:,0,0,:]
        c = tf.reshape( y[:,1:-1,1:-1,:], [-1,12*12,10])
        c = tf.reshape( c, [-1,12*12*10])
        return c,has_key

     


def get_predcoll(num_box, dim_v,dim_h):
    
    #define predicates
    nCOLOR = 10
    Colors=[ str(i) for i in range(nCOLOR)]
    Pos = [str(i) for i in range(12)]
    Constants = dict( {'C':Colors,'P':Pos,'N':Pos}) 
        

    predColl = PredCollection (Constants)
    
    # state predicates
    predColl.add_pred(name='color',arguments=['P','N','C'])
    for i in range(nCOLOR):
            predColl.add_pred(name='is%d'%i  ,arguments=['C'])

    predColl.add_pred(name='has_key'  ,arguments=['C'] )
    predColl.add_pred(name='sameC'  ,arguments=['C','C'])
    predColl.add_pred(name='sameV'  ,arguments=['P','P'])
    predColl.add_pred(name='sameH'  ,arguments=['N','N'])
    # predColl.add_pred(name='incq'  ,arguments=['N','N'])


    predColl.add_pred(name='isBK'  ,arguments=['P','N']).add_fixed_rule(dnf=['color(A,B,C), is0(C)'],variables=['C']) 
    predColl.add_pred(name='isAgent'  ,arguments=['P','N']).add_fixed_rule(dnf=['color(A,B,C), is1(C)'],variables=['C']) 
    predColl.add_pred(name='isGem'  ,arguments=['P','N']).add_fixed_rule(dnf=['color(A,B,C), is2(C)'],variables=['C']) 

    predColl.add_pred(name='isAgentV'  ,arguments=['P']).add_fixed_rule(dnf=['isAgent(A,B)'],variables=['N']) 
    predColl.add_pred(name='isAgentH'  ,arguments=['N']).add_fixed_rule(dnf=['isAgent(B,A)'],variables=['P']) 

    predColl.add_pred(name='isGemV'  ,arguments=['P']).add_fixed_rule(dnf=['isGem(A,B)'],variables=['N']) 
    predColl.add_pred(name='isGemH'  ,arguments=['N']).add_fixed_rule(dnf=['isGem(B,A)'],variables=['P']) 

    predColl.add_pred(name='isItem'  ,arguments=['P','N']).add_fixed_rule(dnf=['not isBK(A,B), not isAgent(A,B)'],variables=[ ]) 

    # predColl.add_pred(name='locked'  ,arguments=['P','N']).add_fixed_rule(dnf=['isItem(A,B), isItem(A,C), incq(B,C)'],variables=[ 'N']) 
    # predColl.add_pred(name='isLock'  ,arguments=['P','N']).add_fixed_rule(dnf=['isItem(A,B), isItem(A,C), incq(C,B)'],variables=[ 'N']) 
    
    predColl.add_pred(name='itemcolor'  ,arguments=['C']).add_fixed_rule(dnf=['not is0(A), not is1(A)'],variables=[ ]) 
    
    predColl.add_pred(name='adjcolor'  ,arguments=['C','C']).add_fixed_rule(dnf=['color(C,D,A), color(C,P_D,B), itemcolor(A), itemcolor(B)'],variables=[ 'P','N'], arg_funcs=['P']) 
    
    predColl.add_pred(name='inGoal'  ,arguments=['C' ]).add_fixed_rule(dnf=['is2(A)', 'inGoal(B), adjcolor(B,A)'],variables=[ 'C']) 

    pt=[ 'isItem(A,B)' ]

     

    def Fn():
        return DNFLayer( [6,1],sig=2., and_init=[-1,.1],or_init=[-1, 1],pta=pt )#,) 
    incp= [ p.name for p in predColl.preds]
        
    predColl.add_pred( name='move'  ,arguments=['P','N'],Frules=LOP.and_op(), Fam = tf.maximum) \
        .add_rule(variables=[ 'C'] , Fn=Fn, inc_preds= incp, exc_conds=[('*','rep1')],Fvar=LOP.or_op_max, use_neg=True ,arg_funcs=['M','P' ]) 
        

    def genn(i):
        n=0
        while True:
            n+=1
            yield 'IC_%d_%d'%(n,i)

    Neg=genn(0)
    Pos=genn(1)


    predColl.add_pred(name=next(Pos)  ,arguments=[],wic=5.).add_fixed_rule(dnf=['isBK(A,B)'],variables=['P','N'],Fvar=LOP.or_op_max ) 
    predColl.add_pred(name=next(Pos)  ,arguments=[],wic=5.).add_fixed_rule(dnf=['isGem(A,B)'],variables=['P','N'],Fvar=LOP.or_op_max ) 
    predColl.add_pred(name=next(Pos)  ,arguments=[],wic=5.).add_fixed_rule(dnf=['isAgent(A,B)'],variables=['P','N'],Fvar=LOP.or_op_max ) 
    
    for _ in range(3,COLOR_COUNT):
        predColl.add_pred(name=next(Pos) ,arguments=[],wic=.2).add_fixed_rule(dnf=['color(A,B,C), is%d(C)'%i],variables=['P','N','C'],Fvar=LOP.or_op_max ) 
    
    # predColl.add_pred(name=next(Pos)  ,arguments=[]).add_fixed_rule(dnf=['isBK(A,B)'],variables=['P','N'],Fvar=LOP.or_op_max) 

    # predColl.add_pred(name='IC_1_0'  ,arguments=[]).add_fixed_rule(dnf=['isGem(A,B), isAgent(A,B)'],variables=['P','N'],Fvar=LOP.or_op_max) 
    # predColl.add_pred(name='IC_2_0'  ,arguments=[]).add_fixed_rule(dnf=['isGem(A,B), isBK(A,B)'],variables=['P','N'],Fvar=LOP.or_op_max) 
    # predColl.add_pred(name='IC_3_0'  ,arguments=[]).add_fixed_rule(dnf=['isBK(A,B), isAgent(A,B)'],variables=['P','N'],Fvar=LOP.or_op_max) 
    
    predColl.add_pred(name=next(Neg)  ,arguments=[],wic=5.).add_fixed_rule(dnf=['isGemH(A), isGemH(B), not sameH(A,B)'],variables=['N','N'],Fvar=LOP.or_op_max  ) 
    predColl.add_pred(name=next(Neg) ,arguments=[],wic=5.).add_fixed_rule(dnf= ['isGemV(A), isGemV(B), not sameV(A,B)'],variables=['P','P'],Fvar=LOP.or_op_max  ) 
    predColl.add_pred(name=next(Neg)  ,arguments=[],wic=5.).add_fixed_rule(dnf=['isAgentH(A), isAgentH(B), not sameH(A,B)'],variables=['N','N'],Fvar=LOP.or_op_max  ) 
    predColl.add_pred(name=next(Neg)  ,arguments=[],wic=5.).add_fixed_rule(dnf=['isAgentV(A), isAgentV(B), not sameV(A,B)'],variables=['P','P'],Fvar=LOP.or_op_max  ) 
    # for _ in range(3):
    #     predColl.add_pred(name=next(Neg)  ,arguments=[]).add_fixed_rule(dnf=['isItem(A,B), isItem(A,M_B)'],arg_funcs=['M'],variables=['P','N'],Fvar=LOP.or_op_max) 
    
    # # predColl.add_pred(name='IC_2_1'  ,arguments=[]).add_fixed_rule(dnf=['posV(A,B), floor(A)'],variables=['B','V'],Fvar=LOP.or_op_max) 
    # predColl.add_pred(name='IC_3_1'  ,arguments=[]).add_fixed_rule(dnf=['posV(A,B), blue(A)'],variables=['B','V'],Fvar=LOP.or_op_max) 
    # predColl.add_pred(name='IC_13_1'  ,arguments=[]).add_fixed_rule(dnf=['posH(A,B), blue(A)'],variables=['B','H'],Fvar=LOP.or_op_max) 
    # predColl.add_pred(name='IC_4_0'  ,arguments=[]).add_fixed_rule(dnf=['posH(A,B), posH(A,C), not H_same(B,C)'],variables=['B','H','H'],Fvar=LOP.or_op_max) 
    # predColl.add_pred(name='IC_5_0'  ,arguments=[]).add_fixed_rule(dnf=['posV(A,B), posV(A,C), not V_same(B,C)'],variables=['B','V','V'],Fvar=LOP.or_op_max) 


         
         


    predColl.initialize_predicates() 


    bg = BackgroundType2( predColl,Constants )
    bg.zero_all()

    # for i in range(nCOLOR):
    #     predColl.add_pred(name='is%d'%i  ,arguments=['C'])

    # bg.add_backgroud('Pos0',(0,) )
    # bg.add_backgroud('Pos11',(11,) )
    for i in range(nCOLOR):
        bg.add_backgroud('is%d'%i  , (str(i),))
        bg.add_backgroud('sameC',(str(i),str(i)) )
             
                

    for i in range(12):
        bg.add_backgroud('sameH',(str(i),str(i)) ) 
        bg.add_backgroud('sameV',(str(i),str(i)) ) 

    # for i in range(12):
    #     # bg.add_backgroud('eq',(i,i) )
    #     for j in range(12):
            
    #         # if i<j:
    #         #     bg.add_backgroud('lt',(i,j) )
    #         if j==i+1:
    #             bg.add_backgroud('incq',(str(i),str(j)) )
        


 
          
            
    bg.compile_bg()
    return predColl, bg 
 

 
 
# env = gym.make("AsterixDeterministic-v4")
# obs0 = env.reset( ) 
# img_bk=obs0.copy()


params.NEG_EXAMPLES_EXPERIENCE=0
memory = Memory()
predColl,bg = get_predcoll(5,5,5)

FC = ForwardChain(predColl)
img2RL = ImageToRL(  )

CNT=0 

  

def get_xo(img):
    c,has_key = img2RL(img )
    BS=tf.shape(c )[0]
    X,Inds = bg.make_batch( bs = BS )
    X['color'] = tf.keras.layers.Flatten()(c)
    X['has_key'] = tf.keras.layers.Flatten()(has_key)
    
    Xo = FC( X,Inds,T=5 )
    return Xo

np.set_printoptions( linewidth=200)
def show_obs():
    showimg( env.toImage(10)) 
    xo = get_xo( env.toImage(1)[np.newaxis].astype(np.float32)) 
    print(xo['has_key'])
    print( np.argmax( xo['color'].numpy().reshape(12,12,10),-1)) 
    return xo



@tf.function
def get_logits( image, ST ):
    
 
    
    
    Xo = get_xo(image)
    moves = Xo['move']
    
    constraints=[]
    cw=[]
    for p in predColl.preds:
        if( p.name.startswith('IC_')):
            if(p.name.endswith('_1')):
                constraints.append( 1.-Xo[p.name])
            else:
                constraints.append( Xo[p.name])
            cw.append( tf.zeros_like(constraints[-1]) + p.wic )
    if constraints:
        constraints = tf.stack(constraints,-1)
        cw = tf.stack(cw,-1)
    else:
        constraints=None
        cw=None
    return moves*ST, tf.nn.softmax(moves*ST), constraints, cw


@tf.function  (
    autograph=False,
    input_signature=[
        tf.TensorSpec(shape=[None,IMG_SIZE,IMG_SIZE,3], dtype=tf.float32),
        tf.TensorSpec(shape=[None,1], dtype=tf.float32),
		tf.TensorSpec(shape=[None ], dtype=tf.int32),
    ],
)
def train_op(OBS,Q,ACT):
    
    with tf.GradientTape() as tape:

        # _, calib_state = img2RL(calib_obs/255.) 
        logits,_,constraints,cw= get_logits(OBS, tf.convert_to_tensor(params.ST,tf.float32) )


        # calib_loss = tf.nn.softmax_cross_entropy_with_logits ( labels=calib_state_label, logits=calib_state) 
        # calib_loss = tf.reduce_mean( calib_loss )

     
        # constraint_losses=[]
        # constraint_losses.append( tf.reduce_mean( neg_ent_loss( zeros+1, constraints[0]) ) )
        # constraint_losses.append( tf.reduce_mean( neg_ent_loss( zeros, constraints[1]) ) )
        # constraint_losses.append( tf.reduce_mean( neg_ent_loss( zeros, constraints[2]) ) )
        # constraint_losses.append( 5*tf.reduce_mean( neg_ent_loss( 1+zeros, constraints[3]) ) )
        print('********************************************')
        print('********************************************')                
        print(constraints.shape)
        print(cw.shape)
        print('********************************************')
        print('********************************************')
        
        if constraints is not None:
            constraint_losses =  tf.reduce_mean( neg_ent_loss(  tf.zeros_like(constraints) ,   constraints)* cw,0 )
        else:
            constraint_losses=0
        
        actor_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ACT) 
        actor_loss = tf.reduce_mean(actor_cross_entropy  * Q[:,0] ) 

        

        total_loss =  actor_loss + params.IC_Lambda* tf.reduce_mean(constraint_losses) #+  tf.add_n(constraint_losses) + 10.1*  calib_loss
        if  FC.losses:
            total_loss+= tf.add_n(FC.losses)*.01


    gradients = tape.gradient(total_loss,FC.variables+img2RL.variables )
    
    gs=[]
    for g  in  gradients:
        if g is not None:
            gs.append(tf.reduce_max(tf.abs(g)))
    optimizer.apply_gradients( zip(gradients,FC.variables+img2RL.variables))
    return total_loss ,gs, constraint_losses
 
def compute_q_value( last_value, done, rwd):
        q_value = np.zeros_like(rwd)
        value = 0 if done else last_value
        for t in reversed(range(0, len(rwd))):
            value = value * params.DISCOUNT_GAMMA + rwd[t]
            q_value[t] = value

        if params.NORMALIZE_Q: 
            q_value -= np.mean(q_value)
            q_value /= (1.e-5+np.std(q_value))
        return q_value[:, np.newaxis]*50

def learn():
    obs, act, rwd = memory.covert_to_array()
    q_value = compute_q_value(0, False, rwd)
    
    if len(act)>1: 
        total,  grs , ct = train_op(  tf.convert_to_tensor(obs,tf.float32) , tf.convert_to_tensor(q_value,tf.float32) , tf.convert_to_tensor( act.astype(np.int32),tf.int32) )
        print( 'loss  ',np.round(total.numpy(),2),'grs:',[g.numpy() for g in grs] )
        print( 'constraint loss',np.round(ct.numpy(),3) )

    memory.reset()  

def agent_step(obs,ST=None):
    if ST==None:
        ST = params.ST
    _,act,_,_  = get_logits( tf.convert_to_tensor(obs[np.newaxis],tf.float32), tf.convert_to_tensor(ST,tf.float32) ) 
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
    obs0=env.toImage(1)

    if(i_episode%20==0):
        xo=show_obs()

    ep_rwd = 0

    cnt=0
    while True:
        act  = agent_step(obs0,params.ST)
        obs1, rwd, done, info = env.step(act)
        obs1=env.toImage(1)
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
        
    print('Ep: %i' % i_episode, "|Episode reward : %.2f, |average N: %.2f, average success %.2f " %  (ep_rwd,avgScore.add(cnt),avgSuccess.add(done and ep_rwd>=5)) )

#%%
