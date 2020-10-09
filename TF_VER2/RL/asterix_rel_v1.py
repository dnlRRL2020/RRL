import os, logging

os.environ["DNLILP_OR_TYPE"] = "expsumlog"
os.environ["DNLILP_AND_TYPE"] = "expsumlog"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)



from Lib.PredicateLib import PredCollection
from Lib.logicLayers  import *
from Lib.BackgroundType2 import  BackgroundType2
from Lib.logicOps import LOP
from Lib.utils import  DotDict
import numpy as np
from time import sleep
import gym
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from asterix_utils import *

ACTIONS = ['noop','up','right','left', 'down']
DIM1=8
DIM2=12
IMG_X=128
IMG_Y=144
F_COUNT = 4

def get_predcoll():
    
    #define predicates
   
    X = ['%d'%i for i in range(DIM1)]
    Y = ['%d'%i for i in range(DIM2)]
    Constants = dict( {'N':X,'Y':Y})
        
        
    predColl = PredCollection (Constants)
    # predColl.add_pred(name='false'  ,arguments=[])
    predColl.add_pred(name='X_U',arguments=['N'])
    predColl.add_pred(name='X_D',arguments=['N'])
    predColl.add_pred(name='Y_L',arguments=['Y'])
    predColl.add_pred(name='Y_R',arguments=['Y'])
    predColl.add_pred(name='ltY',arguments=['Y','Y'])
    predColl.add_pred(name='close',arguments=['Y','Y'])
    for i in range(F_COUNT):
        predColl.add_pred(name='f%i'%i,arguments=['N','Y'])
    predColl.add_pred(name='agentX'  ,arguments=['N']).add_fixed_rule(dnf=['f0(A,B)'],variables=['Y']) 
    predColl.add_pred(name='agentY'  ,arguments=['Y']).add_fixed_rule(dnf=['f0(B,A)'],variables=['N']) 
    predColl.add_pred(name='pred'  ,arguments=['N','Y']).add_fixed_rule(dnf=['f1(A,B)','f2(A,B)'],variables=[ ]) 
    def Fn():
        return DNFLayer( [8,1],sig=2., and_init=[-2,.1],or_init=[-2,1.1],pta=['f0(A,B)'])#,pta=['f0(A,B)']
    def Fnn():
        return DNFLayer( [8,1],sig=2., and_init=[-2,.1],or_init=[-2,1.1],negate=True,pta=['f0(A,B)'])#,pta=['f0(A,B)']
        # return DNFLayer( [4,1],sig=2., and_init=[-2,.1],or_init=[-1,.1],first_only=[6,20])
    incp= [ p.name for p in predColl.preds]
        
    predColl.add_pred( name='up'  ,arguments=[],Frules=LOP.and_op()) \
        .add_rule(variables=[ 'N','Y',  'Y' ] , Fn=Fn, inc_preds= incp, exc_conds=[('*','rep1')],arg_funcs=['M'],Fvar=LOP.or_op_max)  \
        .add_rule(variables=[ 'N','Y',  'Y' ] , Fn=Fnn, inc_preds= incp, exc_conds=[('*','rep1')],arg_funcs=['M'],Fvar=LOP.or_op_max)  
        
    predColl.add_pred( name='down'  ,arguments=[],Frules=LOP.and_op()) \
        .add_rule(variables=[ 'N','Y',  'Y' ] , Fn=Fn, inc_preds= incp, exc_conds=[('*','rep1')],arg_funcs=['P'],Fvar=LOP.or_op_max) \
        .add_rule(variables=[ 'N','Y',  'Y' ] , Fn=Fnn, inc_preds= incp, exc_conds=[('*','rep1')],arg_funcs=['P'],Fvar=LOP.or_op_max)  
        
    # .add_fixed_rule(dnf=['pred(P_A,C), close(B,C)'], arg_funcs=['P'],variables=[ 'N','Y',  'Y' ]) 
    # .add_rule(
    #     
    predColl.add_pred( name='left'  ,arguments=[],Frules=LOP.and_op()) \
        .add_rule(variables=[ 'N','Y',  'Y' ] , Fn=Fn, inc_preds= incp, exc_conds=[('*','rep1')],arg_funcs=[ ],Fvar=LOP.or_op_max) \
        .add_rule(variables=[ 'N','Y',  'Y' ] , Fn=Fnn, inc_preds= incp, exc_conds=[('*','rep1')],arg_funcs=[ ],Fvar=LOP.or_op_max)  

    predColl.add_pred( name='right'  ,arguments=[],Frules=LOP.and_op()) \
        .add_rule(variables=[ 'N','Y',  'Y' ] , Fn=Fn, inc_preds= incp, exc_conds=[('*','rep1')],arg_funcs=[ ],Fvar=LOP.or_op_max) \
        .add_rule(variables=[ 'N','Y',  'Y' ] , Fn=Fnn, inc_preds= incp, exc_conds=[('*','rep1')],arg_funcs=[ ],Fvar=LOP.or_op_max)  

    predColl.add_pred( name='noop'  ,arguments=[],Frules=LOP.and_op()) \
        .add_rule(variables=[ 'N','Y',  'Y' ] , Fn=Fn, inc_preds= incp, exc_conds=[('*','rep1')],arg_funcs=[ ],Fvar=LOP.or_op_max) \
        .add_rule(variables=[ 'N','Y',  'Y' ] , Fn=Fnn, inc_preds= incp, exc_conds=[('*','rep1')],arg_funcs=[ ],Fvar=LOP.or_op_max)  
            
            # predColl.add_pred(name='noop').add_fixed_rule(dnf=['right()', 'left()'],Fvar=LOP.or_op_max) 
    predColl.initialize_predicates() 


    bg = BackgroundType2( predColl,Constants )
    bg.zero_all()

    bg.add_backgroud('X_U' ,('%d'%0,) ) 
    bg.add_backgroud('X_D' ,('%d'%(DIM1-1),) ) 
    bg.add_backgroud('Y_L' ,('%d'%0,) ) 
    # bg.add_backgroud('Y_L' ,('%d'%1,) ) 
    bg.add_backgroud('Y_R' ,('%d'%(DIM2-1),) ) 
    # bg.add_backgroud('Y_R' ,('%d'%(DIM2-2),) ) 
    for i in range(DIM2):
        for j in range(DIM2):
            if i<j:
                bg.add_backgroud('ltY' ,('%d'%i,'%d'%j,) )     
            if abs(i-j)<3:
                bg.add_backgroud('close' ,('%d'%i,'%d'%j,) )     
            
    bg.compile_bg()
    return predColl, bg 
 


params = DotDict({})
ACTIONS = ['noop','up','right','left', 'down']
params.LIFE_PENALTY= -100 
params.IDLE_REWARD = 0
params.EAT_REWARD = 10
params.OBJ_TH = .05
params.MAX_EPISODES=200000
# params.MAX_LEN = 250
params.BATCH_SIZE = 40
params.F_COUNT=4

params.LR_ACTOR=.01
params.DISCOUNT_GAMMA=-3
params.NORMALIZE_Q = False
params.LOSS_METHOD=2
params.ST= 3
 
env = gym.make("AsterixDeterministic-v4")
obs0 = env.reset( ) 
img_bk=obs0.copy()


params.NEG_EXAMPLES_EXPERIENCE=0
memory = Memory(params.DISCOUNT_GAMMA)
predColl,bg = get_predcoll()

optimizer = tf.keras.optimizers.Adam(params.LR_ACTOR )
# optimizer = tfa.train.
 
# optimizer = tfa.optimizers.Yogi(.01)
optimizer = tfa.optimizers.SWA(optimizer)   

CNT=0 
 
FC = ForwardChain(predColl)

@tf.function
def get_logits( state, ST ):
    
    # state=tf.convert_to_tensor(state.astype(np.float32),tf.float32)
    bs = tf.shape(state)[0]
    X,Inds = bg.make_batch(bs)
    for i in range(F_COUNT):
        X['f%d'%i] = state[:,:,i+1]
    
    Xo = FC( X,Inds )
    
    moves = [ 1.0-Xo[i] for i in ACTIONS]
    moves =  tf.concat(moves,-1) 
    return moves*ST, tf.nn.softmax(moves*ST)


lossfn = tfa.losses.SparsemaxLoss(reduction=tf.keras.losses.Reduction.NONE)
@tf.function
def train_op(OBS,Q,ACT):
    
    with tf.GradientTape() as tape:
        
        logits,_ = get_logits(OBS, tf.convert_to_tensor(params.ST,tf.float32) )

        # act = tf.one_hot(ACT,depth=5,axis=-1)
        # actor_cross_entropy = lossfn(y_true = act,y_pred=logits)
        
        actor_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ACT) 

        actor_loss = tf.reduce_mean(actor_cross_entropy  * Q[:,0]  * 1.) 
        
        total_loss =  actor_loss 
        if  FC.losses:
            total_loss+= tf.add_n(FC.losses)*.01


    gradients = tape.gradient(total_loss,FC.variables )
    
    gs=[]
    for g  in  gradients:
        if g is not None:
            gs.append(tf.reduce_max(tf.abs(g)))
    optimizer.apply_gradients( zip(gradients,FC.variables))
    return total_loss ,gs
 
def learn():
    obs, act, rwd, q_value = memory.get_buffer(normalizeQ = params.NORMALIZE_Q,L=params.BATCH_SIZE)
        
    obs1, act1, q_value1 = memory.get_neg(20) 
    if obs1 is not None:
        obs=np.concatenate((obs,obs1),0)
        act=np.concatenate((act,act1),0) 
        q_value=np.concatenate((q_value,q_value1),0)
    
    if len(act)>1: 
        total,  grs = train_op(  tf.convert_to_tensor(obs,tf.float32) , tf.convert_to_tensor(q_value,tf.float32) , tf.convert_to_tensor( act.astype(np.int32),tf.int32) )
        print( 'loss  ',np.round(total.numpy(),2),'grs:',[g.numpy() for g in grs] )
       
def agent_step(obs,ST=params.ST):
    _,act  = get_logits( tf.convert_to_tensor(obs,tf.float32), tf.convert_to_tensor(ST,tf.float32) ) 
    s = act.numpy().ravel()
    action = np.random.choice(range(act.shape[1]), p=s)
    return action
 
def env_step(action):
    obs0,rwd, done, info = env.step(action)
    img =  obs0[24:152,8:152,:] - img_bk[24:152,8:152,:]
    obs=get_img_all(img)
    return obs,rwd,done,info
 
def testrun(ST):
    obs0 = env.reset( ) 
    img,rwd, done, info  = env_step(0)
     
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
        
        

 

max_cnt = 0
max_reward = 0   
 

    
    
          
for i_episode in range(params.MAX_EPISODES):
    
    
    
    if i_episode%10==0 and i_episode>0:
        print('***********************************************************')
        print('\n\n running test mode: ST=10')
        testrun(10.)
        
            
        
    obs0 = env.reset( ) 
    img,rwd, done, info  = env_step(0)
    
    
    ep_rwd = 0
    cnt_total=0
    lives=3
    
    oldact=0
    while True:
    
        
        
        if cnt_total%10==0:
            env.render()
        act  = agent_step(img[np.newaxis] ,i_episode)
        
         
        CNT+=1
        
        img1,R, done, info  = env_step(act)
        
        cnt_total+=1
        ep_rwd+=R
        if R==0:
            R = params.IDLE_REWARD
        elif R>0:
            R = params.EAT_REWARD
         
        lost_life=False
        if info['ale.lives']<lives:
            lost_life=True
            lives=info['ale.lives']
            R=params.LIFE_PENALTY
            # print ('lives:',lives, 'score:', ep_rwd)
             
            
        
        memory.store_transition( img, act, R,lost_life)
        img=img1.copy()
        
        
        if info['ale.lives']==0:
            while memory.get_len()> params.BATCH_SIZE:
                learn()
            memory.reset()
            break
        
        
        if memory.get_len()> params.BATCH_SIZE*5:
            learn()
            
        
        
    max_cnt = max( max_cnt, cnt_total)
    max_reward = max( max_reward, ep_rwd)
        
    
    print('Ep: %i' % i_episode,   '       cnt     ',  CNT,  "         |Ep_r:             %.2f            ,             cnt =                %d               , , |max : %.2f,%d" %  (ep_rwd,cnt_total, max_reward,max_cnt) )
    # ilp_mdl.mdl.print_preds(agent.sess,.1,preds=['bad_up1','bad_down1','bad_left1'])
     
    cnt_total=0