import numpy as np
from .logicOps import  LOP
from .utils import *
from tensorflow.keras.layers import Input, Dense
 
variable_list=['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
class CNTLayer(tf.keras.layers.Layer):
    def __init__(self, cnt_type, neg=True,N=0,**args):
        super(CNTLayer, self).__init__()
        self.cnt_type=cnt_type
        self.neg=neg
        self.N=N
        self.args=args
        
    def build(self, input_shape):
        

           
        if self.cnt_type=='v2':
            self.Wci =  tf.Variable( tf.keras.initializers.TruncatedNormal()( (1,input_shape[-1],self.N) , tf.float32) ,name='Wci')
            self.Wai =  tf.Variable( tf.keras.initializers.TruncatedNormal()( (1,input_shape[-1],self.N) , tf.float32) )
            self.Wb  = tf.Variable( tf.keras.initializers.Zeros()( (1,input_shape[-1]) , tf.float32) )
            self.Wbeta  = tf.Variable( tf.keras.initializers.Constant(.6)( (1,input_shape[-1],1) , tf.float32) )
    
    
        if self.cnt_type=='v3':
          
            N=self.N
            th = np.zeros((1,input_shape[-1],N))

            qs=np.linspace(0,1,N+2)[1:-1]
            for i in range(N):
                th[0,:,i]=qs[i] 
            th=inv_sig(th)
            
            self.Wci =  tf.Variable( tf.keras.initializers.TruncatedNormal()( (1,input_shape[-1],self.N) , tf.float32) ,name='Wci')
            self.Wai =  tf.Variable( tf.keras.initializers.TruncatedNormal()( (1,input_shape[-1],self.N) , tf.float32) )
            self.Wb  = tf.Variable( tf.keras.initializers.Zeros()( (1,input_shape[-1]) , tf.float32) )
            self.WCT = tf.Variable( tf.keras.initializers.Constant(.4)( (1,input_shape[-1],1) , tf.float32) )
            
 
        if self.cnt_type=='n2':
            N=2
            th = np.zeros((1,input_shape[-1],N))
            qs=np.linspace(0,1,N+2)[1:-1]
            for i in range(N):
                th[0,:,i]=qs[i] 
            th=inv_sig(th)
            
            th1  = np.ones_like(th[:,:,0])*4.0
            self.WTH = tf.Variable( th.astype(np.float32), tf.float32)
            self.WCT = tf.Variable( th1.astype(np.float32), tf.float32)
        
 
        if self.cnt_type=='n2x':
            N=2
            th = np.zeros((1,input_shape[-1],N))

            qs=np.linspace(0,1,N+2)[1:-1]
            for i in range(N):
                th[0,:,i]=qs[i] 
            
            th1  = np.ones_like(th[:,:,0])*6.0
            self.WTH = tf.Variable( th.astype(np.float32), tf.float32)
            self.WCT = tf.Variable( th1.astype(np.float32), tf.float32)
            
            
            
        if self.cnt_type=='n':
            N=self.N
            th = np.zeros((1,input_shape[-1],N))

            qs=np.linspace(0,1,N+2)[1:-1]
            for i in range(N):
                th[0,:,i]=qs[i] 
            th=inv_sig(th)
            
            th1  = np.ones_like(th[:,:,0])*4.0
            self.WTH = tf.Variable( th.astype(np.float32), tf.float32)
            self.WCT = tf.Variable( th1.astype(np.float32), tf.float32)
            
        if self.cnt_type=='nN':
            N=self.N
            
            if 'th' in self.args:
                th= self.args['th']
            else:
                th = np.zeros((1,input_shape[-1],N))
                qs=np.linspace(0,1,N+2)[1:-1]
                for i in range(N):
                    th[0,:,i]=qs[i] 
            
            self.WTH = tf.Variable( th.astype(np.float32), tf.float32)
            self.WCT = tf.Variable( tf.keras.initializers.Constant(4.)( (1,input_shape[-1],N), tf.float32) )
        
        if self.cnt_type=='nF':
            N=self.N
            
            if 'th' in self.args:
                th= self.args['th']
            else:
                th = np.zeros((1,input_shape[-1],N))
    
                qs=np.linspace(0,1,N+2)[1:-1]
                for i in range(N):
                    th[0,:,i]=qs[i] 
                    
            
            self.WTH = tf.convert_to_tensor(th.astype(np.float32))
             
    
    def call(self, inputs):
        
        
        if self.cnt_type=='n2':
            d1 = sharp_sigmoid_c  ( inputs - tf.sigmoid(self.WTH[:,:,0]) ,100 , tf.exp(self.WCT) )
            d2 = sharp_sigmoid_c  (  tf.sigmoid(self.WTH[:,:,1]) -inputs ,100,  tf.exp(self.WCT) )
            res = d1*d2
         
        if self.cnt_type=='n2x':
            d1 = sharp_sigmoid_c  ( inputs -  self.WTH[:,:,0]  ,100, tf.exp(self.WCT) )
            d2 = sharp_sigmoid_c  (   self.WTH[:,:,1]  -inputs ,100, tf.exp(self.WCT) )
            res = d1*d2
        
        
        if self.cnt_type=='n':
            res=[]
            for i in range (self.N//2):
                WTH = 1.4*(tf.sigmoid(self.WTH)-.1)
                WCT = tf.math.exp(self.WCT)
                
                
                dg1 = sharp_sigmoid  ( inputs - WTH[:,:,i*2]   ,   WCT  )
                dg2 = sharp_sigmoid  (  WTH[:,:,i*2+1] -inputs  ,  WCT  )
                res.append(dg1*dg2) 
                 

            res=tf.concat( res,-1)
            
            
        if self.cnt_type=='nN': 
            
            res2 = tf.sigmoid  (  ( tf.expand_dims(inputs,-1) - self.WTH )* tf.math.exp(self.WCT)   )
            res=tf.keras.layers.Flatten()(res2)
        
        if self.cnt_type=='nF': 
            
            res = tf.cast( tf.greater( tf.expand_dims(inputs,-1), self.WTH),tf.float32)
            res=tf.keras.layers.Flatten()(res )
            
        if self.cnt_type=='bin':
            res=inputs


         
            
            
        if self.cnt_type=='v2':
            Wci = tf.nn.softmax(self.Wci,-1)
            Wci = tf.math.cumsum(Wci,-1) 

            d = tf.expand_dims(inputs,-1)- Wci
            a = -tf.square(self.Wbeta*d)
            d1 = tf.nn.softmax(a,-1)*self.Wai
            res = tf.sigmoid( tf.math.reduce_sum(d1,-1)+self.Wb   )

       
        
        if self.cnt_type=='v3':
            # Wci = tf.sigmoid(self.Wci)
            Wci = tf.nn.softmax(self.Wci,-1)
            Wci = tf.math.cumsum(Wci,-1)
            d = tf.expand_dims(inputs,-1)- Wci
            d1  = sharp_sigmoid(d, tf.exp(self.WCT)) *self.Wai
            res = tf.sigmoid( tf.math.reduce_sum(d1 ,-1)+self.Wb   )
            
            
        if self.neg:
            res= tf.concat( (res,1.0-res),-1)
        return  res 