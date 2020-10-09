from .BackgroundBase import BackgroundBase
import numpy as np
from itertools import product
from .utils import DotDict
import tensorflow as tf
variable_list=['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
variable_list_dict = { i:variable_list.index(i) for i in variable_list}

class BackgroundType2(BackgroundBase): 

    def __init__(self, predColl, constants ):
        
        self.predColl = predColl
        self.constants=constants
        
      
                
        self.groundings = {}
        self.groundings_index = {}
        self.X = {}
         
        
        
            
        for p in predColl.preds:
            
            
            self.groundings[p.name]=[]
            self.groundings_index[p.name]={}
            self.X[ p.name] =[]
           
    
 
                
        
    def add_backgroud(self,pred_name , pair ,value=1.0 ):
        pair=tuple(pair)
         
        try:
            i =  self.groundings_index[pred_name][pair]
            self.X[ pred_name][i]=value
        except:
            
            self.groundings_index[pred_name][pair] = len( self.groundings[ pred_name] )
            self.groundings[pred_name].append(pair)
            self.X[ pred_name].append(value)
             
    
    def add_zero_backgroud(self,pred_name):
        
        lists = [self.constants[x] for x in self.predColl[pred_name].arguments ]
        for pair in product( *lists):
            if pair not in self.groundings_index[pred_name]:
                self.groundings_index[pred_name][pair] = len( self.groundings[ pred_name] )
                self.groundings[pred_name].append(pair)
                self.X[ pred_name].append(0)
             
    
    def get_X(self,pred_name):
        X = np.array( self.X[pred_name],np.float32 )
        return X   
     
    def apply_func_args(self,Cs):
        
        if 'C' in Cs:
            for i,v in enumerate(Cs['C']):
                if v.startswith('t_'):
                    if v=='t_':
                        Cs['C'][i] = ''
                    else:
                        Cs['C'][i]=v[-1]
        if 'L' in Cs:
            for i,v in enumerate(Cs['L']):
                if v.startswith('H_'):
                    if v=='H_':
                        Cs['L'][i] = ''
                    else:
                        Cs['L'][i]= v[2:-1]
        if 'C' in Cs:
            for i,v in enumerate(Cs['C']):
                if v.startswith('h_'):
                    if v=='h_':
                        Cs['C'][i] = ''
                    else:
                        Cs['C'][i]=v[2]
        if 'L' in Cs:
            for i,v in enumerate(Cs['L']):
                if v.startswith('T_'):
                    if v=='T_':
                        Cs['L'][i] = ''
                    else:
                        Cs['L'][i]= v[3:]
        if 'N' in Cs:
            for i,v in enumerate(Cs['N']):
                if type(v) in (str,) and v.startswith('P_'):
                    Cs['N'][i] = '%d'%( int(v[2:])+1)
                if type(v) in (str,) and v.startswith('M_'):
                    val= max(0,int(v[2:])-1)
                    Cs['N'][i] = '%d'%(val)
                    
        return Cs
    
    def create_inds(self,rule):
        
        pred=rule.predicate
        pairs_var = list ( product ( *[self.constants[i] for i in rule.variables] ) )
        
        len_pairs= len(self.groundings[pred.name])
        len_pairs_var = len(pairs_var)

        InputIndices  = np.zeros( [len_pairs  , len_pairs_var , rule.Lx], np.int64)
        print('******************************************************************')
        print('predicate rule [%s:%d] parameters :'%(pred.name,rule.ruleid), '  Lx', rule.Lx, '  shape', InputIndices.shape )
        print('Lx Details', [ '%s[%d]'%(name,rule.Lx_details_dic[name]) for  name in rule.Lx_details_dic if  rule.Lx_details_dic[name]>0])
                    
        
        args=pred.arguments+rule.variables
        for i in range( len_pairs ):
            for j in range( len_pairs_var ):
                L = 0
                ii=0
                
                if rule.arg_funcs is not None:
                    Cs = self.predColl.get_constant_list(pred,rule,self.groundings[pred.name][i] + pairs_var[j])
                    Cs = self.apply_func_args(Cs)
                else:
                    vl = self.groundings[pred.name][i] + pairs_var[j]
                    # Cs = self.predColl.get_constant_list_fast(pred,rule,vl )
                    
                    Cs=dict( { k:[] for k in self.constants})
                    for k,cl in enumerate( args ):
                          Cs[cl].append( vl[k])
            
                
                for p in self.predColl.preds:
                    # print(pred.name,rule.ruleid,p.name)   
                    if rule.Lx_details_dic[p.name]==0:
                        continue
                    if rule.inc_preds is not None and p.name not in rule.inc_preds:
                        continue
                    if rule.exc_preds is not None and p.name in rule.exc_preds:
                        continue
                    if len(p.arguments)>0:
                        name_set = product( *[Cs[i] for i in p.arguments] )
                        
                    else:
                        name_set=[()]
                    
                    
                    
                    for k,n in  enumerate(name_set):
                        if k in rule.exc_term_inds[p.name]:
                            continue
                        
                        try:
                            # ind = p.pairs.index(n)
                            ind = self.groundings_index[p.name][n]
                            InputIndices[i,j,ii]= ind+L+1
                        except:
                            InputIndices[i,j,ii]=0
                        
                        ii+=1
                    
                    L +=  len(self.groundings[p.name])
                    
        return InputIndices      
              

    def zero_all(self):
        for p in self.predColl.preds:
            self.add_zero_backgroud(p.name)            
    def compile_bg(self ):
        
        self.XC={}
        self.INDC={}

        for p in self.predColl.preds:
            
            self.XC[p.name] =  self.get_X(p.name)[np.newaxis]  
            for r in p.rules:
                self.INDC[ str(r)] = self.create_inds( r )

           
        
        
    def make_batch(self,bs):
        
        O = {}
        for p in self.XC:
            O[p]  =  tf.tile( self.XC[p]  , [bs,1]  )  
        return O,self.INDC        