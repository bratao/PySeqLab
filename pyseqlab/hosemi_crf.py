'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''

import numpy
from .hosemi_crf_ad import HOSemiCRFADModelRepresentation, HOSemiCRFAD
from .utilities import  vectorized_logsumexp

class HOSemiCRFModelRepresentation(HOSemiCRFADModelRepresentation):
    """Model representation that will hold data structures to be used in :class:`HOSemiCRF` class
    """ 
    def __init__(self):
        super().__init__()
        self.z_pky_map = None
        self.z_pi_piy_map = None
        self.S_codebook = None
        self.S_len = None
        self.S_nunmchar = None
        self.b_transition = None
        self.siy_codebook = None  
        self.siy_numchar = None
        self.siy_components = None
        self.siy_z = None
        self.si_siy_codebook = None
        
    def setup_model(self, modelfeatures, states, L):
        super().setup_model(modelfeatures, states, L)
        self.generate_instance_properties()
        
    def generate_instance_properties(self):
        super().generate_instance_properties()
        self.z_pky_map, self.z_pi_piy_map = self.map_pky_z()
        self.S_codebook = self.pky_codebook
        self.S_len, self.S_numchar = self.get_S_info()
        self.siy_codebook, self.siy_numchar, self.siy_components = self.get_siy_info()
        self.b_transition = self.get_backward_transitions()
        self.siy_z = self.map_siy_z()
        self.si_siy_codebook = self.get_si_siy_codebook()        

    def map_pky_z(self):
        """generate a map between elements of the Z set and PY set"""
        
        f_transition = self.f_transition
        Z_codebook = self.Z_codebook
        # given that we demand to have a unigram label features then Z set will always contain Y elems
        Z_numchar = self.Z_numchar
        P_numchar = self.P_numchar
        pky_codebook = self.pky_codebook
        P_codebook = self.P_codebook
        
        z_pi_piy = {}
        z_pky = {}
        for pi in f_transition:
            for pky, pk_y_tup in f_transition[pi].items():
                pk, y = pk_y_tup
                # get number of characters in the pky 
                if(pk == ""):
                    len_pky =  Z_numchar[y]
                else:
                    # +1 is for the separator '|'
                    len_pky = P_numchar[pk] + Z_numchar[y] + 1
                
                for z in Z_codebook:
                    len_z = Z_numchar[z]
                    # check suffix relation
                    start_pos = len_pky - len_z
                    if(start_pos >= 0):
                        check = pky[start_pos:] == z
                        if(check):
                            pky_c = pky_codebook[pky]
                            pk_c = P_codebook[pk]
                            if(z in z_pky):
                                z_pky[z].append(pky_c)
                                z_pi_piy[z][0].append(pk_c)
                                z_pi_piy[z][1].append(pky_c)
                            else:
                                z_pky[z] = [pky_c]
                                z_pi_piy[z] = ([pk_c], [pky_c])
        return(z_pky, z_pi_piy) 
    
    def get_S_info(self): 
        S_codebook = self.S_codebook
        S_len = {}
        S_numchar = {}
        for si in S_codebook:
            if(si == ""):
                S_len[si] = 0
                S_numchar[si] = 0
            else:
                S_len[si] = len(si.split("|"))
                S_numchar[si] = len(si)
        return(S_len, S_numchar)    
    
    def get_siy_info(self):
        S_codebook = self.S_codebook
        Y_codebook = self.Y_codebook
        Z_numchar = self.Z_numchar
        S_numchar = self.S_numchar
        
        siy_components = {}
        siy_codebook = {}
        siy_numchar = {}
        counter = 0
        for si in S_codebook:
            for y in Y_codebook:
                siy = si + "|" + y
                siy_codebook[siy] = counter
                siy_numchar[siy] = S_numchar[si] + Z_numchar[y] + 1
                siy_components[siy] = (si, y)
                counter += 1        
        return(siy_codebook, siy_numchar, siy_components)
    
    def get_backward_transitions(self):
        S_codebook = self.S_codebook
        S_numchar = self.S_numchar
        si_y_suffix = {}
        siy_components = self.siy_components
        siy_numchar = self.siy_numchar
        
        for sk in S_codebook:
            len_sk = S_numchar[sk] 
            for siy in siy_components:
                len_ref = siy_numchar[siy]
                start_pos = len_ref - len_sk
                if(start_pos >= 0): 
                    # check suffix relation
                    check = siy[start_pos:] == sk
                    #check = self.check_suffix(sk, si + "|" + y)
                    if(check):
                        si_y_tup = siy_components[siy]
                        if(si_y_tup in si_y_suffix):
                            prev_sk = si_y_suffix[si_y_tup]
                            len_prev_sk = S_numchar[prev_sk]
                            if(len_sk > len_prev_sk):
                                si_y_suffix[si_y_tup] = sk
                        else:
                            si_y_suffix[si_y_tup] = sk
        
        #print("si_y_suffix {}".format(si_y_suffix))
#         si_y_suffix = self.keep_largest_suffix(si_y_suffix)
        #print("si_y_suffix {}".format(si_y_suffix))
        b_transition = {}
        for (si,y), sk in si_y_suffix.items():
            elmkey = si + "|" + y
            if(si in b_transition):
                b_transition[si][elmkey] = sk
            else:
                b_transition[si] = {elmkey:sk}
        return(b_transition)    

                
    def map_siy_z(self):
        b_transition = self.b_transition
        Z_codebook = self.Z_codebook
        # given that we demand to have a unigram label features then Z set will always contain Y elems
        Z_numchar = self.Z_numchar
        siy_codebook = self.siy_codebook
        siy_numchar = self.siy_numchar
        
        z_siy = {}
        for si in b_transition:
            for siy in b_transition[si]:
                # get number of characters in the siy 
                # +1 is for the separator '|'
                len_siy = siy_numchar[siy] 
                for z in Z_codebook:
                    len_z = Z_numchar[z]
                    # check suffix relation
                    start_pos = len_siy - len_z
                    if(start_pos >= 0):
                        check = siy[start_pos:] == z
                        if(check):
                            siy_c = siy_codebook[siy]
                            if(z in z_siy):
                                z_siy[z].append(siy_c)
                            else:
                                z_siy[z] = [siy_c]
        return(z_siy)    


    def get_si_siy_codebook(self):
        b_transition = self.b_transition
        siy_codebook = self.siy_codebook
        S_codebook = self.S_codebook
        
        si_siy_codebook = {}
        for si in b_transition:
            si_siy_codebook[si] = ([],[])
            for siy, sk in b_transition[si].items():
                si_siy_codebook[si][0].append(siy_codebook[siy])
                si_siy_codebook[si][1].append(S_codebook[sk])

        return(si_siy_codebook)
    

class HOSemiCRF(HOSemiCRFAD):
    """higher-order semi-CRF model 
    
       it implements the model discussed in:
       http://www.jmlr.org/papers/volume15/cuong14a/cuong14a.pdf
    """
    def __init__(self, model, seqs_representer, seqs_info, load_info_fromdisk = 5):
        super().__init__(model, seqs_representer, seqs_info, load_info_fromdisk)
    
    def compute_bpotential(self, w, active_features):
        model = self.model
        siy_codebook = model.siy_codebook
        z_siy = model.siy_z
        b_potential = numpy.zeros(len(siy_codebook))
        # to consider caching the w_indx and fval as in cached_pf
        for z in active_features:
            w_indx, f_val = active_features[z]
            potential = numpy.dot(w[w_indx], f_val)
            # get all ysk's in coded format where z maintains a prefix relation with them
            siy_c_list = z_siy[z]
            b_potential[siy_c_list] += potential

        return(b_potential)
    
    def compute_backward_vec(self, w, seq_id):
        model = self.model
        si_siy_codebook = model.si_siy_codebook
        S_codebook = model.S_codebook
        L = model.L
        T = self.seqs_info[seq_id]["T"]
        activefeatures = self.seqs_info[seq_id]['activefeatures']
        beta = numpy.ones((T+2,len(S_codebook)), dtype='longdouble') * (-numpy.inf)
        beta[T+1,] = 0
        
        for j in reversed(range(1, T+1)):
            accumulator = numpy.ones((len(S_codebook), L), dtype='longdouble') * -numpy.inf
            for d in range(L):
                u = j 
                v = j + d
                if(v > T):
                    break
                b_potential = self.compute_bpotential(w, activefeatures[u,v])
                for si in si_siy_codebook:
                    si_c = S_codebook[si]
                    vec = b_potential[si_siy_codebook[si][0]] + beta[v+1, si_siy_codebook[si][1]]
                    accumulator[si_c, d] = vectorized_logsumexp(vec)   
            for si in si_siy_codebook:
                si_c = S_codebook[si]
                if(L>1):
                    beta[j, si_c] = vectorized_logsumexp(accumulator[si_c, :])
                else:
                    beta[j, si_c] = accumulator[si_c, :]
                    
        return(beta)
    
    def compute_marginals(self, seq_id):
        model = self.model
        Z_codebook = model.Z_codebook
        z_pi_piy = model.z_pi_piy_map
        T = self.seqs_info[seq_id]["T"]
        L = self.model.L

        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"] 
        Z = self.seqs_info[seq_id]["Z"]   
        fpotential_perboundary = self.seqs_info[seq_id]['fpotential']

        P_marginals = numpy.zeros((L, T+1, len(self.model.Z_codebook)), dtype='longdouble')
         
        for j in range(1, T+1):
            for d in range(L):
                u = j
                v = j + d
                if(v > T):
                    break
                boundary = (u, v)
                if(boundary in fpotential_perboundary):
                    f_potential = fpotential_perboundary[boundary]
                    for z in Z_codebook:
                        pi_c, piy_c = z_pi_piy[z]
                        numerator = alpha[u-1, pi_c] + f_potential[piy_c] + beta[v+1, piy_c]
                        P_marginals[d, j, Z_codebook[z]] = numpy.exp(vectorized_logsumexp(numerator) - Z)
        return(P_marginals)

if __name__ == "__main__":
    pass