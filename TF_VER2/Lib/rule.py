from .logicOps import  LOP


class Rule:
    def __init__(self, 
                variables=[], 
                Fn=None,  
                convertors=None,  
                use_neg=False,
                arg_funcs=None, 
                inc_preds=None,
                exc_preds=None, 
                exc_terms=[],
                exc_conds=[], 
                Fout=None,
                Fvar=LOP.or_op(),
                max_rows=None):


        self.variables=variables 
        self.Fn = Fn 
        self.predicate=None
        self.convertors = convertors
        self.Fout=Fout    
        self.Fvar=Fvar    
        

        self.inc_preds=inc_preds
        self.exc_preds=exc_preds

        self.exc_terms=exc_terms
        self.exc_conds=exc_conds

        

        self.use_neg = use_neg
        self.arg_funcs = arg_funcs 
        
        self.max_rows = max_rows
        
        self.I = []
        self.Lx=0
        self.Lx_details=[]
        self.Lx_details_dic={}
        
        
        self.ruleid = None

        
        self.exc_term_inds={}
        self.index_ins = None
        
        
                
        
    def get_term_index(self,term ):

        if not 'not ' in term:
            ind = self.I.index(term)
        else:
            ind = self.I.index(term[4:]) + self.Lx

        return ind  
        
          
    def __repr__(self):
        return "predicate %s,rule:%d"%(self.predicate.name, self.ruleid)  
