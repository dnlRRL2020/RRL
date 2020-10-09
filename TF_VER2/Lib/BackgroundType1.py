from .BackgroundBase import BackgroundBase
import numpy as np
from itertools import product
from .utils import DotDict

variable_list=['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
variable_list_dict = { i:variable_list.index(i) for i in variable_list}

class BackgroundType1(BackgroundBase): 

    def __init__(self, predColl ):
        
        self.predColl = predColl
        self.constants={}
        
        for p in predColl.preds:
            for c in p.arguments:
                self.constants[c]=[]
           
                
        self.groundings = {}
        self.groundings_index = {}
        self.X = {}
        self.Y = {}
        
        
            
        for p in predColl.preds:
            
            
            self.groundings[p.name]=[]
            self.groundings_index[p.name]={}
            
            self.X[ p.name] =[]
            if p.rules:
                self.Y[p.name] = []
                
    
    def update_constants(self,pred_name,pair):
        pred = self.predColl[pred_name]
        
        for i,p in enumerate(pair):
            if p not in self.constants[pred.arguments[i]]:
                self.constants[pred.arguments[i]].append(p)
                
                
        
    def add_backgroud(self,pred_name , pair ,value=1.0 ):
        pair=tuple(pair)
        self.update_constants(pred_name,pair)
        try:
            i =  self.groundings_index[pred_name][pair]
            self.X[ pred_name][i]=value
        except:
            
            self.groundings_index[pred_name][pair] = len( self.groundings[ pred_name] )
            self.groundings[pred_name].append(pair)
            self.X[ pred_name].append(value)
            if pred_name in self.Y:
                self.Y[ pred_name].append(0)
                
            
        
         
    def add_example(self,pred_name , pair ,value=1.0 ):
        pair=tuple(pair)
        self.update_constants(pred_name,pair)
        try:
            i =  self.groundings_index[pred_name][pair]
            self.Y[ pred_name][i]=value
        except:
            
            self.groundings_index[pred_name][pair] = len( self.groundings[ pred_name] )
            self.groundings[pred_name].append(pair)
            self.Y[ pred_name].append(value)
            self.X[ pred_name].append(0)
    
    def add_zero_backgroud(self,pred_name):
        
        lists = [self.constants[x] for x in self.predColl[pred_name].arguments ]
        for pair in product( *lists):
            if pair not in self.groundings_index[pred_name]:
                self.groundings_index[pred_name][pair] = len( self.groundings[ pred_name] )
                self.groundings[pred_name].append(pair)
                self.X[ pred_name].append(0)
                if pred_name in self.Y:
                    self.Y[ pred_name].append(0)
                
    
    def get_X(self,pred_name):
        X = np.array( self.X[pred_name],np.float32 )
        return X   
    def get_Y(self,pred_name):
        Y = np.array( self.Y[pred_name] ,np.float32 )
        return Y
    
    
    
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
              
        
    def add_backgroud_dnf(self,pred_name, formula,variables):
        # formula= [ ['ta(A,B,C)'] ] 

        disjs=[]
        for row in formula:
            conjs=[]
            for col in row:
                term = DotDict() 
                if col.startswith('not '):
                    term.neg=True
                    col=col[4:]
                else:
                    term.neg=False
                c = re.split( ',|\(|\)',col)
                term.pred=c[0]
                term.args=c[1:-1]
                conjs.append(term )
                
            disjs.append(conjs)    
        
        
        
        pred = self.predColl[pred_name]
        
        pairs_arg = list ( product ( *[self.constants[i] for i in pred.arguments] ) )   
        pairs_var = list ( product ( *[self.constants[i] for i in variables] ) )   
        
        
        
        def check_sat(args):
            for conjs in disjs:
                conj_res=True
                for conj in conjs:
                    pair=tuple( args[variable_list_dict[i]] for i in conj.args)
                    try:
                        val = self.X[conj.pred][ self.groundings_index[conj.pred][pair] ]
                    except:
                        val= 0
                    
                    if (not conj.neg and val==0) or (conj.neg and val==1):
                        conj_res=False
                        break
                if conj_res:
                    return True
            return False    
        
        # v = check_sat(('Course49', 'Person361', 'Winter_0304'))
        for p1 in pairs_arg:
            sat=False
            
            for p2 in pairs_var:
                if check_sat(p1+p2):
                    sat=True
                    break
            if sat:
                self.add_backgroud(pred_name,p1)
            
                
    ##################################################################################################
    
    def get_all_fact(self,v,t):
        res=[]
        for p in self.groundings:
            if t in self.predColl[p].arguments:
                for i,pair in enumerate(self.groundings[p]):
                    if v in pair  and (self.X[p][i]==1 or (p in self.Y and self.Y[p][i]==1) )  :
                        xx = ','.join(pair)
                        res.append('%s(%s)'%(p,xx))
                
        return res
        
                
    def compile_bg(self):
         bg=DotDict()
         bg.X={}
         bg.X_inds={}
         bg.X_cnts={}
         bg.LBL={}
         
         for p in self.predColl.preds:
             bg.X[p.name] = self.get_X(p.name)[np.newaxis,:]
             if p.name in self.Y:
                 bg.LBL[p.name] = self.get_Y(p.name)[np.newaxis,:]
             for r in p.rules:
                 bg.X_inds[ str(r)] = self.create_inds( r )
                 
         return bg
            
                 