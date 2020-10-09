import numpy as np
from .logicOps import  LOP
from .utils import *
from tensorflow.keras.layers import Input, Dense
 
variable_list=['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
 
   
class LogicLayer(tf.keras.layers.Layer):
    def __init__(self, 
    logic_type,                         # or, and, and*or
    num_outputs,                        
    sig,
    mean,   
    std,
    L1=0,
    L2=0,
    max_term=1000,
    randmaskpercent=0.,
    first_only=None):

        super(LogicLayer, self).__init__()
        self.num_outputs = num_outputs
        self.sig=sig
        self.std=std
        self.mean=mean
        self.logic_type=logic_type
        self.L1=L1
        self.L2=L2
        self.max_term=max_term
        self.randmaskpercent=randmaskpercent
        self.first_only = first_only
        
        
        
    def build(self, input_shape):
    
        if self.first_only is None: 
            initializer = tf.keras.initializers.TruncatedNormal(mean=self.mean,stddev=self.std)
            self.W = tf.Variable(initializer(shape=[self.num_outputs, input_shape[-1]], dtype=tf.float32),name='W_'+self.logic_type)
        else:
            
            w = np.zeros( [self.num_outputs, input_shape[-1]], np.float32 )-self.first_only[0]
            w[:,0]=self.first_only[1]
            self.W = tf.Variable(w, dtype=tf.float32,name='W_'+self.logic_type)
            
        if self.randmaskpercent>0.:
            self.random_mask = ( np.random.uniform(0,1., size=[self.num_outputs, input_shape[-1]] )> self.randmaskpercent ).astype(np.float32)
            self.random_mask = tf.convert_to_tensor(self.random_mask )
    
    
    def getW(self):
        if self.sig>0:
            W = sharp_sigmoid(self.W,self.sig)
        else:
            W = relu2(self.W)
        return W
  
    def call(self, inputs):
        W = self.getW() 
        if self.randmaskpercent>0.:
            W = W * self.random_mask
        
        size = len(inputs.get_shape().as_list() )
        for _ in range(size-1):
            W = tf.expand_dims(W,axis=0) 

        
        if self.logic_type=='and':
            Z = 1.0- W * (1.0-tf.expand_dims(inputs,axis=-2) )
            S=LOP.and_op()( Z  )

        if self.logic_type=='or':
            Z = W* tf.expand_dims(inputs,axis=-2)
            S = LOP.or_op()( Z )
        
        if self.logic_type=='andor':
            inp = tf.expand_dims(inputs,axis=-2)
            
            S1=LOP.and_op()( 1.0- W * (1.0-inp ) )
            S2 = LOP.or_op()( W*inp )
            S=S1*tf.stop_gradient(S2)

        if self.L1>0:
            s = tf.math.reduce_sum( W,-1)
            L1 = tf.math.reduce_mean(  tf.nn.relu( s-self.max_term)  )
            self.add_loss(L1*self.L1)
        if self.L2>0:
            L2 = tf.math.reduce_mean( W*(1.0-W))
            self.add_loss(L2*self.L2)
        return S

#####################################################

class DNFLayer(tf.keras.layers.Layer):
    def __init__(self, 
                dims,
                sig, 
                and_init=[], 
                or_init=[],
                L1=0,
                L2=0,
                max_term=1000, 
                or_count=0,
                negate=False,
                cnf=False,
                randmaskpercent=0.,
                first_only=None,
                andor=False, 
                pta=None):

                
        super(DNFLayer, self).__init__()
        self.dims=dims
        self.sig=sig
        self.and_init=and_init
        self.or_init=or_init
        self.or_count=or_count
        self.cnf=cnf
        self.negate = negate
        self.rule=None
        self.andor=andor
        self.pta=pta

        if pta is not None:
            self.PTA = FixedDNFLayer(dnf=pta)
        if self.cnf:
            self.and_net = LogicLayer('or', dims[0], sig,and_init[0],and_init[1],L1=L1,L2=L2,max_term=max_term,randmaskpercent=randmaskpercent,first_only=first_only)
            self.or_net = LogicLayer('and', dims[1], sig, or_init[0],or_init[1])
            
        else:
            if self.andor:
                self.and_net = LogicLayer('andor', dims[0], sig,and_init[0],and_init[1],L1=L1,L2=L2,max_term=max_term,randmaskpercent=randmaskpercent,first_only=first_only)
            else:
                self.and_net = LogicLayer('and', dims[0], sig,and_init[0],and_init[1],L1=L1,L2=L2,max_term=max_term,randmaskpercent=randmaskpercent,first_only=first_only)

            self.or_net = LogicLayer('or', dims[1], sig, or_init[0],or_init[1])
            
            
        if self.or_count>0:
            self.or0_net = LogicLayer('or', self.or_count , sig, and_init[0],and_init[1] )
        
  
     
    def get_and(self, decimals=None):
        if decimals is None:
            return self.and_net.getW()
        else:
            return  np.round(self.and_net.getW(),  decimals=decimals)
    def get_or(self, decimals=None): 
        if decimals is None:
            return self.or_net.getW()
        else:
            return  np.round(self.or_net.getW(),  decimals=decimals)
        
    def get_or0(self, decimals=None): 
        if decimals is None:
            return self.or0_net.getW()
        else:
            return  np.round(self.or0_net.getW(),  decimals=decimals)
   
    def print(self,precision=1):
        
        ors=self.get_or(precision)[0,:]
        ws=self.get_and(precision)
        L = len(self.rule.I)
        disjs=[]
        st = ','.join(variable_list[:len(self.rule.predicate.arguments)])
        conc = '%s(%s) :- '%(self.rule.predicate.name,st)                    
        
        for i in range(ors.size):
            if ors[i]>.5:
                disj=[]
                for j in range(ws[i].size):
                    if ws[i,j]>.5:
                        if j>=L:
                            s= 'not_' + self.rulerule.I[j-L]
                        else:
                            s = self.rule.I[j]
                        disj.append(s)
                disjs.append(conc+ ', '.join(disj) )
        
        print( '\n'.join(disjs))
        
        
    def call(self, inputs ):
        if self.or_count>0:
            inputs = tf.concat( (inputs,self.or0_net(inputs)),-1)
        res = self.or_net( self.and_net (inputs))

        if self.pta is not None:
            self.PTA.rule=self.rule
            res = self.PTA(inputs)*res


        if self.negate:
            return 1.0-res
        return res

#####################################################

class FixedDNFLayer(tf.keras.layers.Layer):
    def __init__(self, dnf=None,rule=None, op = LOP.or_op()):
        super(FixedDNFLayer, self).__init__()
        self.dnf=dnf
        self.rule=rule
        self.op = op
  
    def call(self, inputs):
         
        names=self.rule.I
        res = []
        for i,a in enumerate(self.dnf):
            
            resi=[]
            for item in a.split(', '):
                
                resi.append( inputs[:,:,:, names.index(item)] )
                
            resi=tf.stack(resi,-1)
            
            
            
            resi = LOP.and_op()(resi)
            
            res.append(resi)
        if len(res)==1:
            return tf.expand_dims(res[0],-1)
        res = tf.stack(res,-1)
        res=self.op(res,keepdims=True) 
        return res

#####################################################

class ForwardChain  (tf.keras.layers.Layer):
    def __init__(self, predColl=None):
        
        super(ForwardChain, self).__init__()
        self.predColl=predColl

        self.fn_dict={}
        self.generate_funcs()

    def generate_funcs(self):
        for p in self.predColl.preds:
            for r in p.rules:
                if r.Fn is not None:
                    self.fn_dict[str(r)] = r.Fn()
                    self.fn_dict[str(r)].rule = r

    
    
    def call(self, inputs,input_inds,input_cnts=None,T=1):
       
        def pred_func(xs,rule,inp_inds):
            xi=tf.gather( tf.pad( xs, [[0,0],[1,0]], mode='CONSTANT', constant_values=0.0 )    , inp_inds  ,axis=1  )

            # if rule.predicate.name=='up':
            #     print(rule)
            if rule.use_neg: 
                xi = tf.concat( (xi,1.0-xi),-1)

            
            xo = self.fn_dict[str(rule)](xi )[:,:,:,0]
            return xo
                
                
        XO={}
        for pred in self.predColl.preds:
            XO[pred.name]=inputs[pred.name]
             
        for t in   range(T):   
            
            for pred in self.predColl.preds:
                if not pred.rules:
                    continue    
                
                rule_outputs=[]
                for rule in  pred.rules:
                    
                    xs = tf.concat( [ XO[p.name]  for p in  self.predColl.preds if rule.Lx_details_dic[p.name]>0] , -1)
                    xo = pred_func( xs,rule,input_inds[str(rule)])
                    
                    #apply arbitrary function to the output 
                    if rule.Fout is not None:
                        xo = rule.Fout(xo)

                    #aggregate over variable permutations
                    xo =  rule.Fvar ( xo)
                    
                    rule_outputs.append(xo)
                
                if len(rule_outputs)==1:
                    rule_output = rule_outputs[0]
                else:
                    rule_output = pred.Frules(tf.stack(rule_outputs,-1))
                    
                XO[pred.name] = pred.Fam(XO[pred.name],rule_output)
            
             
        return XO