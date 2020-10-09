 
import numpy as np
from tqdm import tqdm,trange
import random
import os
import sys

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

from .logicOps import  LOP
from .utils import *
from .logicLayers import ForwardChain 
 
class ILPEngine(object):

    def __init__(self,params,predColl):
        print( 'Tensorflow Version : ', tf.__version__)
        
         
        self.params=params
        self.predColl = predColl 
 
        self.opts={}
        self.grad_vars={}
        
        print(params)
        print(self.params)
        
        tf.random.set_seed(self.params.SEED) 
         
        self.model_classes=dict()
         
        self.class_pred_names = dict()
        self.all_vars=None
        self.ForwardChain = ForwardChain(predColl)
        
        print('variables:\n', '\n'.join( [ str((x.name,x.shape)) for x in self.ForwardChain.variables]) )
        
       
    
    def define_model(self,loss_preds, target_preds , opts,loss_fn=None):
        
        if loss_fn is None:
            loss_fn = lambda x,y: tf.math.reduce_mean ( neg_ent_loss_p(x,y,.5) )

        
        @tf.function
        def train_step(bgs):
            
            
            with tf.GradientTape() as tape:
                
                
                loss=0.0
                
                for bg in bgs:
                    
                    XO = self.ForwardChain(bg.X,bg.X_inds,bg.X_cnts,self.params.T) 
                    labels = bg.LBL
                    
                    for p in loss_preds:
                        loss +=  loss_fn (  labels[p], XO[p]   ) 
                total_loss = loss + sum(self.ForwardChain.losses )
                
            gradients = tape.gradient(total_loss,self.ForwardChain.variables )
            for opt in opts:
                grs=[]
                vs=[]
                for i,v in enumerate(self.ForwardChain.variables):
                    if opt in v.name:
                        grs.append(gradients[i])
                        vs.append(v)
                 
                opts[opt].apply_gradients(zip(grs,vs))
             
            return total_loss
        
        @tf.function
        def test_step( bg):
            
            
            loss=0.0
                
            XO = self.ForwardChain (bg.X,bg.X_inds,bg.X_cnts,self.params.T+1) 
            labels = bg.LBL
            
            for p in loss_preds:
                loss +=   loss_fn (  labels[p], XO[p]   )  
             
            
            total_loss = loss + sum(self.ForwardChain.losses )
            
            if target_preds is None:
                return XO,total_loss
            return { p:XO[p] for p in target_preds},total_loss
        
        return train_step,test_step
     
        
    
    
    def get_slice(self, X, labels, inds):
        X_o=dict()
        
        
        for item in X:
            X_o[item] = X[item][inds].astype(np.float32)
        
        if type(labels) in (dict,):
            labels_o=dict()
            for item in labels:
                labels_o[item] = labels[item][inds].astype(np.float32)
        else:
            labels_o = labels[inds].astype(np.float32)

        return X_o,labels_o

    def process_cl_dataset( self, is_train, X,labels, target_name, batch_size,plogent, randomize = False   ):

        maxN = labels.shape[0]
        nte = int( np.round( 1.0 * maxN / batch_size  ) )
    
        
        aoutp=[]
        aloss_gr=[]
        
        if randomize:
            R=np.random.permutation(maxN)
        else:
            R=np.arange(maxN)


        
    
        trrr=  trange(nte, desc='loss', leave=True)
        for i in trrr:
            
            ub=min(maxN,(i+1)*batch_size)
            lb = i*batch_size
            inds = np.s_[R[lb:ub]]
            
            
            X_i,labels_i = self.get_slice(X,labels,inds)
            if is_train:
                outp,loss_gr  = self.train_step(  X_i, labels_i, tf.constant(plogent))    
                
            else:
                outp,loss_gr = self.run_step(X_i,labels_i, tf.constant(plogent))    


            aoutp.append(outp)
            
            aloss_gr.append( loss_gr)

            trrr.set_description("loss %.4f"%(np.mean(aloss_gr)))
            
            trrr.refresh()

        outp = np.concatenate( aoutp , 0)  
        aoutp = np.zeros_like(outp)
        aoutp[R] =outp[:]
        aloss_gr=np.mean(aloss_gr)
        return  aoutp,   aloss_gr 
        

    def process_cl_dataset_gen( self, is_train,fn, X,labels, target_names, batch_size,plogent, randomize = False   ):

        maxN = labels[target_names[0]].shape[0]

        all_inds=[]   
        nte = int( np.round( 1.0 * maxN / batch_size  ) )
        
        if is_train:
            print('start training on %d rows'%maxN,flush=True)
        else:
            print('start evaluating %d rows'%maxN,flush=True)
            
        
        
        aoutp={}
        for p in target_names:
            aoutp[p]=[]
        
        aloss_gr=[]
        
        if randomize:
            R=np.random.permutation(maxN)
        else:
            R=np.arange(maxN)


        
    
        trrr=  trange(nte, desc='loss', leave=True)
        for i in trrr:
            
            ub=min(maxN,(i+1)*batch_size)
            lb = i*batch_size
            inds = np.s_[R[lb:ub]]
            
            
            X_i,labels_i = self.get_slice(X,labels,inds)
            
            if type(fn) not in ( tuple,list):
                fn=[fn]
            
            for f in fn:
                if is_train:
                    loss_gr  = f(  X_i, labels_i, plogent)    
                    outp={}
                else:
                    outp,loss_gr  = f(  X_i, labels_i, plogent)    
                
            
            for p in target_names:
                if p in outp:
                    aoutp[p].append(outp[p])
            
            aloss_gr.append( loss_gr)

            if is_train:
                trrr.set_description("train loss %.4f"%(np.mean(aloss_gr)))
            else:
                trrr.set_description("test loss %.4f"%(np.mean(aloss_gr)))
                
            
            # trrr.refresh()
            
        trrr.close()
        print('',flush=True)
        for p in target_names:
            if p in aoutp:
                outp = np.concatenate( aoutp[p] , 0)  
                aoutp[p] = np.zeros_like(outp)
                aoutp[p][R] =outp[:]
            
        aloss_gr=np.mean(aloss_gr)
        return  aoutp,   aloss_gr 
    
    def process_cl_dataset_train( self, fn, X,labels, target_names, batch_size,plogent, randomize = False   ):
        
        
        maxN = labels[target_names[0]].shape[0]

        all_inds=[]   
        nte = int( np.round( 1.0 * maxN / batch_size  ) )
    
        print('start training on %d rows'%maxN,flush=True)
         
        aloss_gr=[]
        
        if randomize:
            R=np.random.permutation(maxN)
        else:
            R=np.arange(maxN)


        
    
        trrr=  trange(nte, desc='loss', leave=True)
        for i in trrr:
            
            ub=min(maxN,(i+1)*batch_size)
            lb = i*batch_size
            inds = np.s_[R[lb:ub]]
            
            
            X_i,labels_i = self.get_slice(X,labels,inds)
            
            if type(fn) not in ( tuple,list):
                fn=[fn]
            
            for f in fn:
                loss_gr  = f(  X_i, labels_i, plogent)    
            
            
            aloss_gr.append( loss_gr.numpy())

            trrr.set_description("train loss %.4f"%(np.mean(aloss_gr)))
           
        trrr.close()
        print('',flush=True)
        return np.mean(aloss_gr) 