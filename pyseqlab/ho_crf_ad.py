'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''


import numpy
from .hosemi_crf_ad import HOSemiCRFADModelRepresentation, HOSemiCRFAD
from .utilities import HO_AStarSearcher, vectorized_logsumexp

class HOCRFADModelRepresentation(HOSemiCRFADModelRepresentation):
    """Model representation that will hold data structures to be used in :class:`HOCRFAD` class
    
       it includes all attributes in the  :class:`HOSemiCRFADModelRepresentation` parent class
    """ 
    def __init__(self):
        # call super class
        super().__init__()
        
    def filter_activated_states(self, activated_states, accum_active_states, boundary):
        """filter/prune states and y features 
        
           Args:
               activaed_states: dictionary containing possible active states/y features
                                it has the form {patt_len:{patt_1, patt_2, ...}}
               accum_active_states: dictionary of only possible active states by position
                                    it has the form {pos_1:{state_1, state_2, ...}}
               boundary: tuple (u,v) representing the current boundary in the sequence
        """

        Z_elems = self.Z_elems
        filtered_activestates = {}
        __, pos = boundary

        for z_len in activated_states:
            if(z_len == 1):
                continue
            start_pos = pos - z_len + 1
            if((start_pos, start_pos) in accum_active_states):
                filtered_activestates[z_len] = set()
                for z_patt in activated_states[z_len]:
                    check = True
                    zelems = Z_elems[z_patt]
                    for i in range(z_len):
                        pos_bound = (start_pos+i, start_pos+i)
                        if(pos_bound not in accum_active_states):
                            check = False
                            break
                        if(zelems[i] not in accum_active_states[pos_bound]):
                            check = False
                            break
                    if(check):                        
                        filtered_activestates[z_len].add(z_patt)
        return(filtered_activestates)  
                        
class HOCRFAD(HOSemiCRFAD):
    """higher-order CRF model that uses algorithmic differentiation in gradient computation
    
      Args:
          model: an instance of :class:`HOCRFADModelRepresentation` class
          seqs_representer: an instance of :class:`SeqsRepresenter` class
          seqs_info: dictionary holding sequences info
       
      Keyword Arguments:
          load_info_fromdisk: integer from 0 to 5 specifying number of cached data 
                              to be kept in memory. 0 means keep everything while
                              5 means load everything from disk
                               
      Attributes:
          model: an instance of :class:`HOCRFADModelRepresentation` class
          weights: a numpy vector representing feature weights
          seqs_representer: an instance of :class:`pyseqlab.feature_extraction.SeqsRepresenter` class
          seqs_info: dictionary holding sequences info
          beam_size: determines the size of the beam for state pruning
          fun_dict: a function map
          def_cached_entities: a list of the names of cached entities sorted (descending)
                               based on estimated space required in memory 
                               
    """
    def __init__(self, model, seqs_representer, seqs_info, load_info_fromdisk = 5):

        super().__init__(model, seqs_representer, seqs_info, load_info_fromdisk)
    
    def compute_fpotential(self, w, active_features):
        """compute the potential of active features in a specified boundary 
        
           Args:
               w: weight vector (numpy vector)
               active_features: dictionary of activated features in a specified boundary
        """
        model = self.model
        pky_codebook = model.pky_codebook
        z_pky_map = model.z_pky_map
        f_potential = numpy.zeros(len(pky_codebook))

        # to consider caching the w_indx and fval as in cached_pf
        for z in active_features:
            w_indx, f_val = active_features[z]
            potential = numpy.dot(w[w_indx], f_val)
            # get all pky's in coded format where z maintains a suffix relation with them
            pky_c_list = z_pky_map[z]
            f_potential[pky_c_list] += potential

        return(f_potential)
               
    def compute_forward_vec(self, w, seq_id):
        """compute the forward matrix (alpha matrix)
           
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
            
           .. note:: 
            
              activefeatures need to be loaded first in :attr:`seqs.info`
        """
        model = self.model
        pi_pky_map = model.pi_pky_map
        P_codebook = model.P_codebook
        P_len = model.P_len
        T = self.seqs_info[seq_id]["T"]
        active_features = self.seqs_info[seq_id]['activefeatures']

        alpha = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        alpha[0,P_codebook[""]] = 0
        
        fpotential_perboundary = {}
        for j in range(1, T+1):
            boundary = (j, j)
            # compute f_potential
            f_potential = self.compute_fpotential(w, active_features[boundary])
            fpotential_perboundary[boundary] = f_potential
            for pi in pi_pky_map:
                pi_c = P_codebook[pi]
                if(j >= P_len[pi]):
                    pky_c_list, pk_c_list = pi_pky_map[pi]
                    vec = f_potential[pky_c_list] + alpha[j-1, pk_c_list]
                    alpha[j, pi_c] = vectorized_logsumexp(vec)
                    
        self.seqs_info[seq_id]['fpotential'] = fpotential_perboundary
        
        return(alpha)

    
    def compute_backward_vec(self, w, seq_id):
        """compute the backward matrix (beta matrix)
        
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
            
           .. note:: 
           
              fpotential per boundary dictionary should be available in :attr:`seqs.info`
        """
        model = self.model
        pi_pky_map = model.pi_pky_map
        P_codebook = model.P_codebook
        len_P = len(P_codebook)
        T = self.seqs_info[seq_id]["T"]
        fpotential_perboundary = self.seqs_info[seq_id]['fpotential']
        
        beta = numpy.ones((T+2, len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        beta[T+1, :] = 0

        for j in reversed(range(1, T+1)):
            track_comp = numpy.ones((len_P, len_P), dtype='longdouble') * (-numpy.inf)
            f_potential = fpotential_perboundary[j, j]
            for pi in pi_pky_map:
                pi_c = P_codebook[pi]
                pky_c_list, pk_c_list = pi_pky_map[pi]
                vec = f_potential[pky_c_list] + beta[j+1, pi_c]
                track_comp[pk_c_list, pi_c] = vec
            for p_c in P_codebook.values():
                beta[j, p_c] = vectorized_logsumexp(track_comp[p_c, :])  
        return(beta)
    
    def compute_marginals(self, seq_id):
        """ compute the marginal (i.e. probability of each y pattern at each position)
        
            Args:
                seq_id: integer representing unique id assigned to the sequence
            
            .. note::
            
               - fpotential per boundary dictionary should be available in :attr:`seqs.info`
               - alpha matrix should be available in :attr:`seqs.info`
               - beta matrix should be available in :attr:`seqs.info`
               - Z (i.e. P(x)) should be available in :attr:`seqs.info`
        """
        model = self.model
        Z_codebook = model.Z_codebook
        z_pi_piy = model.z_pi_piy_map
        T = self.seqs_info[seq_id]["T"]
        L = self.model.L
        
        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"] 
        Z = self.seqs_info[seq_id]["Z"]   
        fpotential_perboundary = self.seqs_info[seq_id]['fpotential']
 
        P_marginals = numpy.zeros((T+1, len(model.Z_codebook)), dtype='longdouble')
         
        for j in range(1, T+1):
            for d in range(L):
                u = j
                v = j + d
                if(v > T):
                    break
                boundary = (u, v)
                f_potential = fpotential_perboundary[boundary]
                for z in Z_codebook:
                    pi_c, piy_c, pk_c = z_pi_piy[z]
                    numerator = alpha[u-1, pi_c] + f_potential[piy_c] + beta[v+1, pk_c]
                    P_marginals[j, Z_codebook[z]] = numpy.exp(vectorized_logsumexp(numerator) - Z)        
        
        return(P_marginals)
    
#     def compute_marginals(self, w, seq_id):
#         model = self.model
#         P_codebook = model.P_codebook
#         len_P = len(P_codebook)
#         pi_z_pk = model.pi_z_pk
#         Z_codebook = model.Z_codebook
#         T = self.seqs_info[seq_id]["T"]
#         alpha = self.seqs_info[seq_id]['alpha']
#         beta = self.seqs_info[seq_id]['beta']
#         P_marginals = numpy.zeros((T+1, len(model.Z_codebook)), dtype='longdouble')
#         print("alpha ", alpha)
#         Z = self.seqs_info[seq_id]['Z']
#         print("Z ", Z)
#         fpotential_perboundary = self.seqs_info[seq_id]['fpotential']
#         print(pi_z_pk)
#         print(P_codebook)
#         f_transition = model.f_transition
#         pky_z = model.z_pky
#         pky_codebook = model.pky_codebook
#         print("pky_z ", pky_z)
#         for j in reversed(range(1, T+1)):
#             marginal_dict = {}
#             f_potential = fpotential_perboundary[j, j]
#             for pi in f_transition:
#                 beta_pi = beta[j+1, P_codebook[pi]]
#                 for pky in f_transition[pi]:
#                     pk, y = f_transition[pi][pky]
#                     accum = alpha[j-1, P_codebook[pk]] + f_potential[pky_codebook[pky]] + beta_pi
#                     for z_patt in pky_z[pky]:
#                         if(z_patt in marginal_dict):
#                             marginal_dict[z_patt] = numpy.logaddexp(marginal_dict[z_patt], accum)
#                         else:
#                             marginal_dict[z_patt] = accum
#             print("j ", j)
#             print("marginal ", marginal_dict)
#             for z_patt in marginal_dict:
#                 P_marginals[j, Z_codebook[z_patt]] = numpy.exp(marginal_dict[z_patt]-Z)
#         self.seqs_info[seq_id]['P_marginal'] = P_marginals
#         print(P_marginals)
#         return(P_marginals)
    
    def compute_feature_expectation(self, seq_id, P_marginals, grad):
        """compute the features expectations (i.e. expected count of the feature based on learned model)
        
           Args:
               seq_id: integer representing unique id assigned to the sequence
               P_marginals: probability matrix for y patterns at each position in time
               grad: numpy vector with dimension equal to the weight vector. It represents the gradient
                     that will be computed using the feature expectation and the global features of the sequence
            
           .. note::
            
             - activefeatures (per boundary) dictionary should be available in :attr:`seqs.info`
             - P_marginal (marginal probability matrix) should be available in :attr:`seqs.info`
        """        
        activefeatures = self.seqs_info[seq_id]["activefeatures"]
        Z_codebook = self.model.Z_codebook
        for boundary, features_dict in activefeatures.items():
            u, __ = boundary
            for z_patt in features_dict:
                w_indx, f_val = features_dict[z_patt]
                grad[w_indx] += f_val * P_marginals[u, Z_codebook[z_patt]]
    
    def prune_states(self, j, delta, beam_size):
        """prune states that fall off the specified beam size
        
           Args:
               j: current position (integer) in the sequence
               delta: score matrix 
               beam_size: specified size of the beam (integer)
        """
        P_codebook_rev = self.model.P_codebook_rev
        P_elems = self.model.P_elems
#         pi_lendict = self.model.pi_lendict

#         # sort the pi in descending order of their score
#         indx_sorted_pi = numpy.argsort(delta[j,:])[::-1]
#         # identify states falling out of the beam
#         indx_falling_pi = indx_sorted_pi[beam_size:]
#         # identify top-k states/pi
#         indx_topk_pi = indx_sorted_pi[:beam_size]
#         # remove the effect of states/pi falling out of the beam
#         delta[j, indx_falling_pi] = -numpy.inf

        # using argpartition as better alternative to argsort
        indx_partitioned_pi = numpy.argpartition(-delta[j, :], beam_size)
        # identify top-k states/pi
        indx_topk_pi = indx_partitioned_pi[:beam_size]
#         # identify states falling out of the beam
#         indx_falling_pi = indx_partitioned_pi[beam_size:]
#         # remove the effect of states/pi falling out of the beam
#         delta[j, indx_falling_pi] = -numpy.inf

        # get topk states
        topk_pi = {P_codebook_rev[indx] for indx in indx_topk_pi}
        topk_states = set()
        for pi in topk_pi:
            topk_states.add(P_elems[pi][-1])
        return(topk_states)
    

    def viterbi(self, w, seq_id, beam_size, stop_off_beam = False, y_ref=[], K=1):
        """decode sequences using viterbi decoder 
                
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
               beam_size: integer representing the size of the beam
                
           Keyword Arguments:
               stop_off_beam: boolean indicating if to stop when the reference state \
                              falls off the beam (used in perceptron/search based learning)
               y_ref: reference sequence list of labels (used while learning)
               K: integer indicating number of decoded sequences required (i.e. top-k list)
                          
        """
        model = self.model
        P_elems = model.P_elems
        pi_pky_map = model.pi_pky_map
        P_codebook = model.P_codebook
        P_codebook_rev = model.P_codebook_rev
        len_P = len(P_codebook)
        P_len = model.P_len
        num_states = model.num_states
        T = self.seqs_info[seq_id]["T"]
        # records max score at every time step
        delta = numpy.ones((T+1,len_P), dtype='longdouble') * (-numpy.inf)
        # the score for the empty sequence at time 0 is 1
        delta[0, P_codebook[""]] = 0
        back_track = {}
        # records where violation occurs -- it is 1-based indexing 
        viol_index = []

        if(beam_size == num_states):
            # case of exact search and decoding
            l = {}
            l['activefeatures'] = (seq_id, )
            self.check_cached_info(seq_id, l)
            active_features = self.seqs_info[seq_id]['activefeatures']
            for j in range(1, T+1):
                boundary = (j, j)
                # vector of size len(pky)
                f_potential = self.compute_fpotential(w, active_features[boundary])
                #^print("f_potential ", f_potential)
                for pi in pi_pky_map:
                    pi_c = P_codebook[pi]
                    pky_c_list, pk_c_list = pi_pky_map[pi]
                    vec = f_potential[pky_c_list] + delta[j-1, pk_c_list]
                    delta[j, pi_c] = numpy.max(vec)
                    #print("max chosen ", delta[j, P_codebook[pi]])
                    argmax_ind = numpy.argmax(vec)
                    #print("argmax chosen ", argmax_ind)
                    pk_c_max = pk_c_list[argmax_ind]
                    pk = P_codebook_rev[pk_c_max]
                    y = P_elems[pk][-1]
                    back_track[j, pi_c] = (pk_c_max, y)
            
        else:
            # case of inexact search and decoding
            l = {}
            l['seg_features'] = (seq_id, )
            self.check_cached_info(seq_id, l)
            # tracks active states by boundary
            accum_activestates = {}
            for j in range(1, T+1):
                boundary = (j, j)
                active_features = self.identify_activefeatures(seq_id, boundary, accum_activestates)
                # vector of size len(pky)
                f_potential = self.compute_fpotential(w, active_features)
                #^print("f_potential ", f_potential)
                for pi in pi_pky_map:
                    pi_c = P_codebook[pi]
                    pky_c_list, pk_c_list = pi_pky_map[pi]
                    vec = f_potential[pky_c_list] + delta[j-1, pk_c_list]
                    delta[j, pi_c] = numpy.max(vec)
                    #print("max chosen ", delta[j, P_codebook[pi]])
                    argmax_ind = numpy.argmax(vec)
                    #print("argmax chosen ", argmax_ind)
                    pk_c_max = pk_c_list[argmax_ind]
                    pk = P_codebook_rev[pk_c_max]
                    y = P_elems[pk][-1]
                    back_track[j, pi_c] = (pk_c_max, y)
                        
                topk_states = self.prune_states(j, delta, beam_size)
                # update tracked active states -- to consider renaming it          
                accum_activestates[boundary] = accum_activestates[boundary].intersection(topk_states)
                #^print('delta[{},:] = {} '.format(j, delta[j,:]))
                #^print("topk_states ", topk_states)
                if(y_ref):
                    if(y_ref[j-1] not in topk_states):
                        viol_index.append(j)
                        if(stop_off_beam):
                            T = j
                            break
        if(K == 1):
            # decoding the sequence
            Y_decoded = []
            p_T_c = numpy.argmax(delta[T,:])
            p_T = P_codebook_rev[p_T_c]
            y_T = P_elems[p_T][-1]
            Y_decoded.append((p_T_c,y_T))
            t = T - 1
            while t>0:
                p_tplus1_c = Y_decoded[-1][0]
                p_t_c, y_t = back_track[(t+1, p_tplus1_c)]
                Y_decoded.append((p_t_c, y_t))
                t -= 1
            Y_decoded.reverse()
     
            Y_decoded = [yt for __,yt in Y_decoded]
#             print("Y_decoded {}".format(Y_decoded))
#             print('delta ', delta)
#             print('backtrack ', back_track)
#             print("P_codebook ", P_codebook)
            return(Y_decoded, viol_index)
        else:
            asearcher = HO_AStarSearcher(P_codebook_rev, P_elems)
            topK = asearcher.search(delta, back_track, T, K)
#             print('topk ', topK)
            return(topK, viol_index)
    
if __name__ == "__main__":
    pass