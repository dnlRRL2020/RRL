from .logicOps import  LOP
from .logicLayers import  FixedDNFLayer
from .rule import Rule

class Predicate:
    
    def __init__(self,
                name=None, 
                arguments=[], 
                Fam= LOP.eq2_op(), 
                Frules=LOP.or_op(),
                wic=1.0
                ):
        
        self.wic=wic
        self.name=name 
        self.Fam=Fam
        self.Frules=Frules
        self.rules=[]
        self.arity=len(arguments)
        self.arguments = arguments 
        
    def __repr__(self):
        return "predicate %s"%(self.name)  


    def add_fixed_rule( self,dnf,**kwargs):
        
        Fn = lambda: FixedDNFLayer(dnf=dnf,rule=None)
        r = Rule(  Fn=Fn,**kwargs)
        
        r.use_neg=True
        r.predicate = self
        r.ruleid = len(self.rules)
        
        self.rules.append(r)
        return self
    
    def add_rule( self,**args):
        r = Rule( **args)
        r.predicate = self
        r.ruleid = len(self.rules)
        self.rules.append(r)
        return self