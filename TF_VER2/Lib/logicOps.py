import tensorflow as tf
from os import environ
from .utils import custom_grad,relu1


class LOP:
    
    
    eps=1e-6
    def __init__(self):
        pass
           

    @staticmethod
    def or_op_sum(x,axis=-1,keepdims=False):
        return relu1( tf.math.reduce_sum( x  , axis=axis,keepdims=keepdims) )
    
    
    @staticmethod
    def or_op_prod(x,axis=-1,keepdims=False):
        return 1.0-tf.math.reduce_prod( 1.0-x  , axis=axis,keepdims=keepdims) 
    @staticmethod
    def and_op_prod(x,axis=-1,keepdims=False):
        return tf.math.reduce_prod( x  , axis=axis,keepdims=keepdims) 
    

    @staticmethod
    def or_op_max(x,axis=-1,keepdims=False):
        return  tf.math.reduce_max( x  , axis=axis,keepdims=keepdims) 
    @staticmethod
    def and_op_max(x,axis=-1,keepdims=False):
        return tf.math.reduce_min( x  , axis=axis,keepdims=keepdims) 
    


    @staticmethod
    def or_op_expsumlog(x,axis=-1,keepdims=False):
        return 1.0-tf.math.exp ( tf.math.reduce_sum( tf.math.log( (1.0-x+LOP.eps)/(1+LOP.eps))  , axis=axis, keepdims=keepdims))  
    @staticmethod
    def and_op_expsumlog(x,axis=-1,keepdims=False):
        return tf.math.exp ( tf.math.reduce_sum(  tf.math.log( (x+LOP.eps)/(1+LOP.eps)  )  , axis=axis,name='my_prod',keepdims=keepdims) )

    @staticmethod
    def or_op_grad(x,axis=-1,keepdims=False):
        return custom_grad( LOP.or_op_max(x,axis,keepdims), LOP.or_op_expsumlog(x,axis,keepdims) )
    
    @staticmethod
    def and_op_grad(x,axis=-1,keepdims=False):
        return custom_grad( LOP.and_op_max(x,axis,keepdims), LOP.and_op_expsumlog(x,axis,keepdims) )


    @staticmethod     
    def or2_prod(x,y):
       return 1.0- (1.0-x)*(1.0-y)
   
    @staticmethod     
    def ond2_prod(x,y):
       return x*y
     
    @staticmethod
    def or_op():
        if environ["DNLILP_OR_TYPE"] == "expsumlog":
            return LOP.or_op_expsumlog
        if environ["DNLILP_OR_TYPE"] == "prod":
            return LOP.or_op_prod

        if environ["DNLILP_OR_TYPE"] == "max":
            return LOP.or_op_max

        if environ["DNLILP_OR_TYPE"] == "grad":
            return LOP.or_op_grad
        
        if environ["DNLILP_OR_TYPE"] == "sum":
            return LOP.or_op_sum
        return None
    
    
    @staticmethod
    def and_op():
        if environ["DNLILP_AND_TYPE"] == "expsumlog":
            return LOP.and_op_expsumlog
        if environ["DNLILP_AND_TYPE"] == "prod":
            return LOP.and_op_prod
        
        if environ["DNLILP_AND_TYPE"] == "max":
            return LOP.and_op_max
    
        if environ["DNLILP_AND_TYPE"] == "grad":
            return LOP.and_op_grad


        return None
    
    @staticmethod     
    def or2_op():
       
        if environ["DNLILP_OR_TYPE"] == "expsumlog":
            return LOP.or2_prod
        if environ["DNLILP_OR_TYPE"] == "prod":
            return LOP.or2_prod
        
        if environ["DNLILP_OR_TYPE"] == "max":
            return tf.math.maximum
        
        return None


        
    @staticmethod     
    def and2_op():
        
        if environ["DNLILP_OR_TYPE"] == "expsumlog":
             return LOP.and2_prod
        if environ["DNLILP_OR_TYPE"] == "prod":
             return LOP.and2_prod
         
        if environ["DNLILP_OR_TYPE"] == "max":
             return tf.math.minimum
         
         
    @staticmethod     
    def eq2_op():
        return lambda x,y: y
        