'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''


from collections import OrderedDict
import numpy
from .ho_crf_ad import  HOCRFADModelRepresentation, HOCRFAD
from .utilities import vectorized_logsumexp

class HOCRFModelRepresentation(HOCRFADModelRepresentation):
    """Model representation that will hold data structures to be used in :class:`HOCRF` class
    """ 
    def __init__(self):
        super().__init__()
        self.S_codebook = None
        self.S_len = None
        self.S_numchar = None
        self.b_transition = None
        self.ysk_codebook = None
        self.si_ysk_map = None
        self.z_ysk_map = None

        
    def setup_model(self, modelfeatures, states, L):
        """setup and create the model representation
        
           Creates all maps and codebooks needed by the :class:`HOSemiCRFAD` class
           
           Args:
               modelfeatures: set of features defining the model
               states: set of states (i.e. tags)
               L: length of longest segment
        """
        super().setup_model(modelfeatures, states, L)
    
    def generate_instance_properties(self):
        """generate instance properties that will be later used by :class:`HOSemiCRFAD` class
        """
        super().generate_instance_properties()
        self.S_codebook = self.get_backward_states()
        self.S_len, self.S_numchar = self.get_S_info()
        self.b_transition = self.get_backward_transitions()
        self.ysk_codebook = self.get_ysk_codebook()
        self.si_ysk_map= self.get_si_ysk_map()
        self.z_ysk_map = self.map_z_ysk()


    def get_backward_states(self):
        Y_codebook = self.Y_codebook
        Z_elems = self.Z_elems
        Z_len = self.Z_len
        S = {}
        for z_patt in Z_elems:
            elems = Z_elems[z_patt]
            z_len = Z_len[z_patt]
            #print("z_patt")
            for i in range(1, z_len):
                S["|".join(elems[i:])] = 1
                #print("i = {}".format(i))
                #print("suffix {}".format("|".join(elems[i:])))
        for y in Y_codebook:
            S[y] = 1
        # empty element         
        S[""] = 1
        S_codebook = {s:i for (i, s) in enumerate(S)}
        #print("S_codebook ", S_codebook)
        return(S_codebook)
    
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
                
    def get_backward_transitions(self):
        Y_codebook = self.Y_codebook
        S_codebook = self.S_codebook
        S_numchar = self.S_numchar
        sk_y = {}
        
        for s in S_codebook:
            for y in Y_codebook:
                sk_y[(s, y)] = 1
        
        sk_y_prefix = {}
        for s in S_codebook:
#             if(s != ""):
            len_s = S_numchar[s]
            for (sk, y) in sk_y:
                ref_str = y + "|" + sk
                #check prefix relation
                check = ref_str[:len_s] == s
                #check = self.check_prefix(s, ref_str)
                if(check):
                    if((sk, y) in sk_y_prefix):
                        sk_y_prefix[(sk, y)].append(s)
                    else:
                        sk_y_prefix[(sk, y)] = [s]
                            
        sk_y_prefix = self.keep_longest_elems(sk_y_prefix)
        b_transition = {}
        for (sk, y), si in sk_y_prefix.items():
            if(sk == ""):
                elmkey = y
            else:
                elmkey = y + "|" + sk
            if(si in b_transition):
                b_transition[si][elmkey] = sk
            else:
                b_transition[si] = {elmkey:sk}
        return(b_transition)    
        
    def get_ysk_codebook(self):
        b_transition = self.b_transition
        ysk_codebook = {}
        counter = 0
        for si in b_transition:
            for ysk in b_transition[si]:
                ysk_codebook[ysk] = counter
                counter += 1
        return(ysk_codebook)
    
    def map_z_ysk(self):
        Z_codebook = self.Z_codebook
        Z_numchar = self.Z_numchar
        ysk_codebook = self.ysk_codebook
        z_ysk = {}
        
        for ysk in ysk_codebook:
            for z in Z_codebook:
                len_z = Z_numchar[z]
                #check prefix relation
                check = ysk[:len_z] == z
                if(check):
                    ysk_c = ysk_codebook[ysk]
                    if(z in z_ysk):
                        z_ysk[z].append(ysk_c)
                    else:
                        z_ysk[z] = [ysk_c]
        return(z_ysk)  
    
    def get_si_ysk_map(self):
        b_transition = self.b_transition
        ysk_codebook = self.ysk_codebook
        S_codebook = self.S_codebook
        si_ysk_map = {}
        for si in b_transition:
            si_ysk_map[si] = ([],[])
            for ysk, sk in b_transition[si].items():
                si_ysk_map[si][0].append(ysk_codebook[ysk])
                si_ysk_map[si][1].append(S_codebook[sk])

        return(si_ysk_map)
    
        
class HOCRF(HOCRFAD):
    """higher-order CRF model 
    
        - currently it supports *only* search-based training methods such as `COLLINS-PERCEPTRON` or `SAPO`
        - it implements the model discussed in:
          https://papers.nips.cc/paper/3815-conditional-random-fields-with-high-order-features-for-sequence-labeling.pdf
    
    """
    def __init__(self, model, seqs_representer, seqs_info, load_info_fromdisk = 5):
        super().__init__(model, seqs_representer, seqs_info, load_info_fromdisk) 
    
    def compute_bpotential(self, w, active_features):
        model = self.model
        ysk_codebook = model.ysk_codebook
        z_ysk = model.z_ysk_map
        b_potential = numpy.zeros(len(ysk_codebook))
        # to consider caching the w_indx and fval as in cached_pf
        for z in active_features:
            w_indx, f_val = active_features[z]
            potential = numpy.dot(w[w_indx], f_val)
            # get all ysk's in coded format where z maintains a prefix relation with them
            ysk_c_list = z_ysk[z]
            b_potential[ysk_c_list] += potential

        return(b_potential)
    
    def compute_backward_vec(self, w, seq_id):
        model = self.model
        si_ysk_map = model.si_ysk_map
        S_codebook = model.S_codebook
        ysk_codebook = model.ysk_codebook
        patts_len = model.patts_len
        Z_len = model.Z_len
        T = self.seqs_info[seq_id]["T"]
        activefeatures_perboundary = self.seqs_info[seq_id]['activefeatures']

        beta = numpy.ones((T+2, len(S_codebook)), dtype='longdouble') * (-numpy.inf)
        beta[T+1, S_codebook[""]] = 0
        
        for j in reversed(range(1, T+1)):
            for si in si_ysk_map:
                b_potential = numpy.zeros(len(ysk_codebook))
                si_c = S_codebook[si]
                for z_len in patts_len:
                    b = j + z_len - 1
                    if(b <= T):
                        boundary = (b, b)
                        active_features = activefeatures_perboundary[boundary]
                        features = {z:active_features[z] for z in active_features if Z_len[z] == z_len}
                        # compute b_potential vector
                        b_potential += self.compute_bpotential(w, features)
                ysk_list_c, sk_list_c = si_ysk_map[si]
                vec = b_potential[ysk_list_c] + beta[j+1, sk_list_c]
                beta[j, si_c] = vectorized_logsumexp(vec)  
                    
        return(beta)
    
    def compute_seq_gradient(self, w, seq_id, grad):
        """sequence gradient computation
        
           .. warning::
        
              the :class:`HOCRF` currently **does not support** gradient based training.
              Use search based training methods such as `COLLINS-PERCEPTRON` or `SAPO`
              
              this class is used for demonstration of the computation of the backward matrix
              using suffix relation as outlined in:
              https://papers.nips.cc/paper/3815-conditional-random-fields-with-high-order-features-for-sequence-labeling.pdf
        """
        try:
            raise("this model can only be trained using search-based methods. Use HOCRFAD if you want gradient based training")
        except Exception:
            print("this model can only be trained using search-based methods. Use HOCRFAD if you want gradient based training")
     
    def validate_forward_backward_pass(self, w, seq_id):
        self.clear_cached_info([seq_id])
        # this will compute alpha and beta matrices and save them in seqs_info dict
        l = OrderedDict()
        l['activefeatures'] = (seq_id, )
        l['alpha'] = (w, seq_id)
        l['beta'] = (w, seq_id)
        self.check_cached_info(seq_id, l)
        
        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"]
        
        Z_alpha = vectorized_logsumexp(alpha[-1,:])
        Z_beta = vectorized_logsumexp(beta[1, :])
        raw_diff = numpy.abs(Z_alpha - Z_beta)
        print("alpha[-1,:] = {}".format(alpha[-1,:]))
        print("beta[1,:] = {}".format(beta[1,:]))
        print("Z_alpha : {}".format(Z_alpha))
        print("Z_beta : {}".format(Z_beta))
        print("Z_aplha - Z_beta {}".format(raw_diff))
 
        rel_diff = raw_diff/(Z_alpha + Z_beta)
        print("rel_diff : {}".format(rel_diff))
        self.clear_cached_info([seq_id])
        #print("seqs_info {}".format(self.seqs_info))
        return((raw_diff, rel_diff))   
    
if __name__ == "__main__":
    pass