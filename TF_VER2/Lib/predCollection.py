import  numpy as np
from itertools import product,permutations
from collections import OrderedDict 
from datetime import datetime
from collections import Counter
from .logicOps import  LOP
from .logicLayers import  FixedDNFLayer
from .predicate import  Predicate
from .utils import DotDict
import re

variable_list=['A','B','C','D','E','F','G','H','I','J','K','L','M','N']

#####################################################

def _gen_all_orders( v, r,var=False):
    if not var:
        inp = [v[i] for i in r]
    else:
        inp = [v[i[0]] for i in r]
        
    p = product( *inp)
    return [kk for kk in p]

#####################################################

class PredCollection:
    
    def __init__(self, constants  ):
        
        self.constants = constants
        self.preds = []
        self.preds_by_name=dict({})
        self.continuous_vars={}
        
        
    def add_continous_vars( self, name, size):
        self.continuous_vars[name]=size
     
    def get_constant_list_fast( self,pred , rule, vl ):
        Cs=dict( { k:[] for k in set(pred.arguments+rule.variables)})
        for i,cl in enumerate( pred.arguments+rule.variables ):
            Cs[cl].append( vl[i])
             
        return Cs
    
    def get_constant_list( self,pred , rule, vl ):
        Cs=dict( { k:[] for k in self.constants.keys()})
        for i,cl in enumerate( pred.arguments+rule.variables ):
            Cs[cl[0]].append( vl[i])
            if cl[0]=='N' and  rule.arg_funcs:
                if 'M' in rule.arg_funcs: 
                    Cs['N'].append('M_' + vl[i] )
                if 'P' in rule.arg_funcs: 
                    Cs['N'].append('P_' + vl[i] )
                    
            if cl[0]=='L' and  rule.arg_funcs:
                
                if 'tH' in rule.arg_funcs:
                    Cs['L'].append('H_'+vl[i])
                    Cs['C'].append('t_'+vl[i])
                
                if 'Th' in rule.arg_funcs:
                    Cs['C'].append('h_'+vl[i])
                    Cs['L'].append('T_'+vl[i])
        
        return Cs
        
    def add_pred( self,**args):
        p = Predicate( **args)
        
        self.preds.append(p)
        self.preds_by_name[p.name] = p
        return p


    def __len__(self):
        return len(self.preds)
    def __getitem__(self, key):
        return self.preds_by_name[key]
        
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
    
     

    def initialize_predicates(self):
        
        
        for pred in self.preds:
            
            for rule in pred.rules:

                rule.Lx_details =[]
                Cs = self.get_constant_list(pred,rule,variable_list)
                
                for p in self.preds:
                    
                    rule.exc_term_inds[p.name]=[]    
                    if rule.inc_preds is not None and p.name not in rule.inc_preds:
                        rule.Lx_details.append(0)
                        rule.Lx_details_dic[p.name]=0
                        continue
                    if rule.exc_preds is not None and p.name in rule.exc_preds:
                        rule.Lx_details.append(0)
                        rule.Lx_details_dic[p.name]=0
                        continue

                    if len(p.arguments)>0:
                        name_set = list ( product ( *[Cs[i] for i in p.arguments] ) )
                    else:
                        name_set=[()]

                    Li=0
                    
                    for i,n in enumerate(name_set):
                        term = p.name + '(' + ','.join(n)+')'
                        pcond = False
                        for c in rule.exc_conds:
                            if p.name == c[0] or c[0]=='*':
                                cl=Counter(n)
                                l = list(cl.values())
                                if len(l)>0:
                                    if c[1]=='rep1':
                                        if max(l)>1:
                                            pcond=True
                                            break
                                    if c[1]=='rep2':
                                        if max(l)>2:
                                            pcond=True
                                            break
                        if term not in rule.exc_terms and not pcond:
                            Li+=1
                            rule.I.append( term)
                        else:
                            rule.exc_term_inds[p.name].append(i)
                        
                    rule.Lx_details.append(Li)
                    rule.Lx_details_dic[p.name]=Li
                
                if rule.use_neg:
                    negs=[]
                    for k in rule.I:
                        negs.append( 'not '+k)  
                    rule.I.extend(negs)
                rule.Lx = sum(rule.Lx_details)
    