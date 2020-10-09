import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2
DIM1=8
DIM2=12

with open('patterns_new2.pkl', 'rb') as handle:
    filts = pickle.load(handle)


def get_img(im,filt,th=.7):
    O = np.zeros( (DIM1,DIM2) , np.float32)
    result = cv2.matchTemplate(im,filt,cv2.TM_CCOEFF_NORMED)
    locs = np.argwhere(result>th) 
    cx,cy= result.shape
    Xs = locs[:,0] * DIM1 / cx 
    Ys = locs[:,1] * DIM2 / cy 
    for x,y in zip(Xs,Ys):
        O[ int(x),int(y)]=1
    return O

def get_img_pos(im,filt,th=.7):
    O = np.zeros( (DIM1,DIM2) , np.float32)
    result = cv2.matchTemplate(im,filt,cv2.TM_CCOEFF_NORMED)
    locs = np.argwhere(result>th) 
    cx,cy= result.shape
    Xs = locs[:,0] * DIM1 / cx 
    Ys = locs[:,1] * DIM2 / cy 
    return zip(Xs,Ys)
            
def showimg(img):

    plt.imshow(img)
    plt.show()
    

def get_img_all( img):
    O1 = get_img(img, filts['ag'],th=.5)
    O2 = get_img(img, filts['ag2'],th=.5)
    
    O3 = get_img(img, filts['ag3'],th=.5)
    O4 = get_img(img, filts['ag4'],th=.5)
    
    O3 = np.logical_or( O4 ,  O3) 
    O2 = np.logical_or( O1 ,  O2) 
    O2 = np.logical_or( O3 ,  O2) 

    O3 = get_img(img, filts['e1'])*1
    O4 = get_img(img, filts['e2'])*1
    O51 = get_img(img, filts['f'])*1
    O52 = get_img(img, filts['f2'])*1
    O5 = np.logical_or( O51 ,  O52) 
    
    O1 = 1 - O2-O3-O4-O5
    O1[O1<0]=0

    O = np.stack( [O1,O2,O3,O4,O5] ,-1)
    return np.reshape( O, [DIM1*DIM2,5])

def get_img_all32( img ):
    O1 = get_img(img, filts['ag'],th=.5)
    O2 = get_img(img, filts['ag2'],th=.5)
    O3 = get_img(img, filts['ag3'],th=.5)
    O4 = get_img(img, filts['ag4'],th=.5)
    O3 = np.logical_or( O4 ,  O3) 
    O2 = np.logical_or( O1 ,  O2) 
    O2 = np.logical_or( O3 ,  O2) 

    O3 = get_img(img, filts['e1'])*1
    O4 = get_img(img, filts['e2'])*1
    # O3 = np.logical_or( O4 ,  O3) 
    
    # O4 = get_img(img, filts['f'])*1
    O51 = get_img(img, filts['f'])*1
    O52 = get_img(img, filts['f2'])*1
    O4 = np.logical_or( O51 ,  O52) 
    
    
    O1 = 1 - O2-O3-O4
    O1[O1<0]=0

    O = np.stack( [O1,O2,O3,O4] ,-1)
    return np.reshape( O, [DIM1*DIM2,4])


class Memory(object):
    def __init__(self, gamma):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []
        self.gamma=gamma
        
        self.negexp_obs=[]
        self.negexp_act=[]
        self.negexp_q=[]
     
        
    def get_neg(self,L):
        if len(self.negexp_obs)==0:
            return None,None,None
        else:
            L = min(L,len(self.negexp_obs))
            inds=np.random.permutation(len(self.negexp_obs))
            array_obs = np.stack(self.negexp_obs,0)
            array_act = np.array(self.negexp_act)
            array_q = np.array(self.negexp_q)
            return array_obs[inds[:L]],array_act[inds[:L]],array_q[inds[:L],np.newaxis]
            return array_obs[-L:],array_act[-L:],array_q[-L:,np.newaxis]
       
    def store_transition(self, obs, act, rwd,lost_life  ):
        if lost_life:
            del self.ep_obs[-31:]
            del self.ep_act[-31:]
            del self.ep_rwd[-31:]
            self.ep_rwd[-1]=rwd
            
            try:
                for i in range(1):
                    self.negexp_obs.append( self.ep_obs[-1])
                    self.negexp_act.append( self.ep_act[-1])
                    self.negexp_q.append( rwd)
                    
                    self.negexp_obs.append( self.ep_obs[-2])
                    self.negexp_act.append( self.ep_act[-2])
                    self.negexp_q.append( rwd)
                    
                    # self.negexp_obs.append( self.ep_obs[-3])
                    # self.negexp_act.append( self.ep_act[-3])
                    # self.negexp_q.append( rwd*.1)
                    
                    
                    
                    while len(self.negexp_obs)> 30:
                        del self.negexp_obs[0]
                        del self.negexp_act[0]
                        del self.negexp_q[0]
                    
            except:
                pass

        else:
            self.ep_obs.append(obs)
            self.ep_act.append(act)
            self.ep_rwd.append( float(rwd) )
            
       
        
                
    
    def get_len(self):
        return len(self.ep_obs)
     
            
            
    def get_buffer(self, normalizeQ=False,L=None):
        
        
        q_value = np.zeros_like(self.ep_rwd, dtype=np.float32)
        
        
        if self.gamma>0:
            value = 0
            for t in reversed(range(0, len(self.ep_rwd))):
                value = value * self.gamma + self.ep_rwd[t]
                q_value[t] = value
        else:
            for t in reversed(range(0, len(self.ep_rwd))):
                maxv = min( t+3, len(self.ep_rwd))
                q_value[t] = sum( self.ep_rwd[t:maxv])
        
        
        
        if normalizeQ:
            q_value -= np.mean(q_value)
            q_value /= (1.e-5+np.std(q_value))
            
    
        
        array_obs = np.stack(self.ep_obs,0)
        array_act = np.array(self.ep_act)
        array_rwd = np.array(self.ep_rwd)
        
        if L is None:
            self.reset()
            return array_obs,array_act,array_rwd,q_value[:, np.newaxis]
        else:
            del self.ep_obs[:L]
            del self.ep_act[:L]
            del self.ep_rwd[:L]

            
            return array_obs[:L],array_act[:L],array_rwd[:L],q_value[:L, np.newaxis]
            
            
            
            

    def reset(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []

 