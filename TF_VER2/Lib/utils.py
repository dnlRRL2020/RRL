import numpy as np
from sklearn import metrics
import tensorflow as tf


def get_auc(y_true,y_pred):
     
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc_score = metrics.auc(fpr, tpr)
    
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred) 
    aupr_score = metrics.auc(recall, precision)
    return auc_score,aupr_score

def custom_grad(fx,gx):
    t = gx
    return t + tf.stop_gradient(fx - t)




def sharp_sigmoid(x,c=5):
    return tf.sigmoid(c*x)
        
def sharp_sigmoid_c(x,th,c=5.):
    return tf.sigmoid(c*x)
    th=float(th)
    return tf.sigmoid(tf.grad_pass_through(tf.clip_by_value(c*x,-th,th)))
 


def neg_ent_loss(label,prob,eps=1.0e-4):
    return - ( label*tf.math.log(eps+prob) + (1.0-label)*tf.math.log(eps+1.0-prob))



def neg_ent_loss_p(label,prob,p=.5,eps=1.0e-4):
    return - ( p*label*tf.math.log(eps+prob) + (1.0-p)*(1.0-label)*tf.math.log(eps+1.0-prob))



def count_number_trainable_params():
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    for trainable_variable in tf.compat.v1.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def inv_sig(x):
    return 1/(1+np.exp(-x))

def relu1(x):
    return tf.nn.relu( 1.0-tf.nn.relu(1.0-x))
    return tf.math.maximum( 0., tf.minimum(x,1.0))

import tensorflow_addons as tfa
def relu2(x):
    return tfa.activations.hardshrink(x,lower=1.0,upper=0.0)
class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val
            
            
            