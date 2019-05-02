'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''

from collections import OrderedDict
import numpy
from .linear_chain_crf import LCRFModelRepresentation, LCRF
from .utilities import FO_AStarSearcher, vectorized_logsumexp

class FirstOrderCRFModelRepresentation(LCRFModelRepresentation):
    """Model representation that will hold data structures to be used in :class:`FirstOrderCRF` class
        
       it includes all attributes in the :class:`LCRFModelRepresentation` parent class
       
       Attributes:
           Y_codebook_rev: reversed codebook (dictionary) of :attr:`Y_codebook`
           startstate_flag: boolean indicating if to use an edge/boundary state (i.e. __START__ state) 

    """ 
    def __init__(self):

        super().__init__()
        self.Y_codebook_rev = None
        self.startstate_flag = None

        
    def setup_model(self, modelfeatures, states, L):
        """setup and create the model representation
        
           Creates all maps and codebooks needed by the :class:`FirstOrderCRF` class
           
           Args:
               modelfeatures: set of features defining the model
               states: set of states (i.e. tags)
               L: length of longest segment
        """
        super().setup_model(modelfeatures, states, L)

    
    def generate_instance_properties(self):
        """generate instance properties that will be later used by :class:`FirstOrderCRF` class
        """
        super().generate_instance_properties()
        self.Y_codebook_rev = self.get_Y_codebook_reversed()
    
    def get_modelstates_codebook(self, states):
        """create states codebook by mapping each state to a unique code/number
        
           Args:
               states: set of tags identified in training sequences
           
           Example::
           
               states = {'B-PP', 'B-NP', ...}
        """
        start_state = '__START__'
        if(start_state in states):
            del states[start_state]
            Y_codebook = {s:i+1 for (i, s) in enumerate(states)}
            Y_codebook[start_state] = 0
            states[start_state] = 1
            self.startstate_flag = True
        else:
            Y_codebook = {s:i for (i, s) in enumerate(states)}
            self.startstate_flag = False
        return(Y_codebook)  
        
    def get_Y_codebook_reversed(self):
        """generate reversed codebook of :attr:`Y_codebook`
        """
        Y_codebook = self.Y_codebook
        return({code:state for state, code in Y_codebook.items()})
                    
class FirstOrderCRF(LCRF):
    """first-order CRF model 
    
      Args:
          model: an instance of :class:`FirstOrderCRFModelRepresentation` class
          seqs_representer: an instance of :class:`SeqsRepresenter` class
          seqs_info: dictionary holding sequences info
       
      Keyword Arguments:
          load_info_fromdisk: integer from 0 to 5 specifying number of cached data 
                              to be kept in memory. 0 means keep everything while
                              5 means load everything from disk
                               
      Attributes:
          model: an instance of :class:`FirstOrderCRFModelRepresentation` class
          weights: a numpy vector representing feature weights
          seqs_representer: an instance of :class:`pyseqlab.feature_extraction.SeqsRepresenter` class
          seqs_info: dictionary holding sequences info
          beam_size: determines the size of the beam for state pruning
          fun_dict: a function map
          def_cached_entities: a list of the names of cached entities sorted (descending)
                               based on estimated space required in memory 
                               
    """
    def __init__(self, model, seqs_representer, seqs_info, load_info_fromdisk=5):
        super().__init__(model, seqs_representer, seqs_info, load_info_fromdisk)
        
    def cached_entitites(self, load_info_fromdisk):
        """construct list of names of cached entities in memory
        """
        def_cached_entities = super().cached_entitites(load_info_fromdisk)
        inmemory_info = ["alpha", "Z", "beta", "potentialmat_perboundary"]
        def_cached_entities += inmemory_info
        return(def_cached_entities)

#     def compute_psi_potential(self, w, seq_id):
#         """ assumes that activefeatures_matrix has been already generated and saved in self.seqs_info dictionary """
#         Y_codebook = self.model.Y_codebook
#         Z_lendict = self.model.Z_lendict
#         Z_elems = self.model.Z_elems
#         # T is the length of the sequence 
#         T = self.seqs_info[seq_id]["T"]
#         # number of possible states including the __START__ and __STOP__ states
#         M = self.model.num_states
#         # get activefeatures_matrix
#         activefeatures = self.seqs_info[seq_id]["activefeatures"]
#         potential_matrix = numpy.zeros((T+1,M,M), dtype='longdouble')
#         for boundary in activefeatures:
#             t = boundary[0]
#             for y_patt in activefeatures[boundary]:
#                 f_val = list(activefeatures[boundary][y_patt].values())
#                 w_indx = list(activefeatures[boundary][y_patt].keys())
#                 
#                 potential = numpy.dot(w[w_indx], f_val)
#                 if(Z_lendict[y_patt] == 1):
#                     y_c = Z_elems[y_patt][0]
#                     potential_matrix[t, :, Y_codebook[y_c]] += potential
#                 else:
#                     # case of len(parts) = 2
#                     y_p = Z_elems[y_patt][0]
#                     y_c = Z_elems[y_patt][1]
#                     potential_matrix[t, Y_codebook[y_p], Y_codebook[y_c]] += potential
# #         print("potential_matrix {}".format(potential_matrix))
#         return(potential_matrix)
    
    def compute_potential(self, w, active_features):
        """compute the potential matrix of active features in a specified boundary 
        
           Args:
               w: weight vector (numpy vector)
               active_features: dictionary of activated features in a specified boundary
        """
        model = self.model
        Y_codebook = model.Y_codebook
        Z_len = model.Z_len
        Z_elems = model.Z_elems
        # number of possible states including the __START__ and __STOP__ states
        M = model.num_states
        # get activefeatures_matrix
        potential_matrix = numpy.zeros((M,M), dtype='longdouble')

        for y_patt in active_features:
            w_indx, f_val = active_features[y_patt]
            potential = numpy.dot(w[w_indx], f_val)
            if(Z_len[y_patt] == 1):
                y_c = Z_elems[y_patt][0]
                potential_matrix[:, Y_codebook[y_c]] += potential
            else:
                # case of len(parts) = 2
                y_p = Z_elems[y_patt][0]
                y_c = Z_elems[y_patt][1]
                potential_matrix[Y_codebook[y_p], Y_codebook[y_c]] += potential
#         print("potential_matrix {}".format(potential_matrix))
        return(potential_matrix)

    def compute_forward_vec(self, w, seq_id):
        """compute the forward matrix (alpha matrix)
            
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
             
           .. note:: 
             
              activefeatures need to be loaded first in :attr:`seqs.info`
        """
        model = self.model
        # T is the length of the sequence 
        T = self.seqs_info[seq_id]["T"]
        # number of possible states including the __START__ and __STOP__ states
        M = model.num_states
        startstate_flag = model.startstate_flag
        active_features = self.seqs_info[seq_id]['activefeatures']
        potentialmat_perboundary = {}
        alpha = numpy.ones((T+1, M), dtype='longdouble') * (-numpy.inf)
        
        if(startstate_flag):
            alpha[0,0] = 0
        # corner case at t = 1
        t = 1; i = 0
        boundary = (t,t)
        potential_matrix = self.compute_potential(w, active_features[boundary])
        potentialmat_perboundary[boundary] = potential_matrix
        alpha[t, :] = potential_matrix[i, :]
        
        for t in range(1, T):
            boundary = (t+1, t+1)
            potential_matrix = self.compute_potential(w, active_features[boundary])
            potentialmat_perboundary[boundary] = potential_matrix
            for j in range(M):
                alpha[t+1, j] = vectorized_logsumexp(alpha[t, :] + potential_matrix[:, j])
                
        self.seqs_info[seq_id]['potentialmat_perboundary'] = potentialmat_perboundary
        return(alpha)
  
    def compute_backward_vec(self, w, seq_id):
        """compute the backward matrix (beta matrix)
        
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
            
           .. note:: 
           
              potential matrix per boundary dictionary should be available in :attr:`seqs.info`
        """
        # length of the sequence without the appended states __START__ and __STOP__
        T = self.seqs_info[seq_id]["T"]
        # number of possible states including the __START__ and __STOP__ states
        M = self.model.num_states
        beta = numpy.ones((T+1, M), dtype = 'longdouble') * (-numpy.inf)
        beta[T, :] = 0
        # get the potential matrix 
        potentialmat_perboundary = self.seqs_info[seq_id]["potentialmat_perboundary"]
        for t in reversed(range(1, T+1)):
            potential_matrix = potentialmat_perboundary[t,t]
            for i in range(M):
                beta[t-1, i] = vectorized_logsumexp(potential_matrix[i, :] + beta[t, :])

        return(beta) 

    def compute_marginals(self, seq_id):
        """ compute the marginal (i.e. probability of each y pattern at each position)
        
            Args:
                seq_id: integer representing unique id assigned to the sequence
            
            .. note::
            
               - potential matrix per boundary dictionary should be available in :attr:`seqs.info`
               - alpha matrix should be available in :attr:`seqs.info`
               - beta matrix should be available in :attr:`seqs.info`
               - Z (i.e. P(x)) should be available in :attr:`seqs.info`
        """
        model = self.model
        Y_codebook = model.Y_codebook
        Z_codebook = model.Z_codebook
        Z_len = model.Z_len
        Z_elems = model.Z_elems
        
        T = self.seqs_info[seq_id]["T"]
        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"] 
        Z = self.seqs_info[seq_id]["Z"]   
        potentialmat_perboundary = self.seqs_info[seq_id]["potentialmat_perboundary"]
        
        P_marginals = numpy.zeros((T+1, len(Z_codebook)), dtype='longdouble') 
         
        for j in range(1, T+1):
            potential_matrix = potentialmat_perboundary[j, j]
            for y_patt in Z_codebook:
#                 print("y_patt {}".format(y_patt))
                if(Z_len[y_patt] == 1):
                    y_c = Y_codebook[Z_elems[y_patt][0]]
                    accumulator = alpha[j, y_c] + beta[j, y_c] - Z
                else:
                    # case of len(parts) = 2
                    y_b = Y_codebook[Z_elems[y_patt][0]]
                    y_c = Y_codebook[Z_elems[y_patt][1]]
                    accumulator = alpha[j-1, y_b] + potential_matrix[y_b, y_c] + beta[j, y_c] - Z
                P_marginals[j, Z_codebook[y_patt]] = numpy.exp(accumulator)
        return(P_marginals)
    
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
            t = boundary[0]
            for z_patt in features_dict:
                w_indx, f_val = features_dict[z_patt]
                grad[w_indx] += f_val * P_marginals[t, Z_codebook[z_patt]]
            
    def prune_states(self, j, score_mat, beam_size):
        """prune states that fall off the specified beam size
        
           Args:
               j: current position (integer) in the sequence
               score_mat: score matrix 
               beam_size: specified size of the beam (integer)
        """
        Y_codebook_rev = self.model.Y_codebook_rev
        # using argpartition as better alternative to argsort
        indx_partitioned_y = numpy.argpartition(-score_mat[j, :], beam_size)
        # identify top-k states/pi
        indx_topk_y = indx_partitioned_y[:beam_size]
#         # identify states falling out of the beam
        indx_falling_y = indx_partitioned_y[beam_size:]
        # remove the effect of states/pi falling out of the beam
        score_mat[j, indx_falling_y] = -numpy.inf
        
        # get topk states
        topk_y = {Y_codebook_rev[indx] for indx in indx_topk_y}
        
        return(topk_y)
    
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
        # number of possible states
        M = model.num_states
        T = self.seqs_info[seq_id]['T']
        Y_codebook_rev = model.Y_codebook_rev
        Y_codebook = model.Y_codebook
        score_mat = numpy.ones((T+1, M), dtype='longdouble') * -numpy.inf
        score_mat[0,0] = 0
        # back pointer to hold the index of the state that achieved highest score while decoding
        backpointer = numpy.ones((T+1, M), dtype='int') * (-1)
        viol_index = []
        
        if(beam_size == M):
            # case of exact search and decoding
            l = {}
            l['activefeatures'] = (seq_id, )
            self.check_cached_info(seq_id, l)
            active_features = self.seqs_info[seq_id]['activefeatures']

            # corner case at t = 1
            t = 1; i = 0
            potential_matrix = self.compute_potential(w, active_features[t,t])
            score_mat[t, :] = potential_matrix[i, :]
            backpointer[t, :] = 0
            
            for t in range(2, T+1):
                potential_matrix = self.compute_potential(w, active_features[t,t])
                for j in range(M):
                    vec = score_mat[t-1, :] + potential_matrix[:, j]
                    score_mat[t, j] = numpy.max(vec)
                    backpointer[t, j] = numpy.argmax(vec)
        else:
            # case of inexact search and decoding
            l = {}
            l['seg_features'] = (seq_id, )
            self.check_cached_info(seq_id, l)
            
            accum_activestates = {}

            for t in range(1, T+1):
                boundary = (t, t)
                active_features = self.identify_activefeatures(seq_id, boundary, accum_activestates)
                potential_matrix = self.compute_potential(w, active_features)
                for j in range(M):
                    vec = score_mat[t-1, :] + potential_matrix[:, j]
                    score_mat[t, j] = numpy.max(vec)
                    backpointer[t, j] = numpy.argmax(vec)
                    
                topk_states = self.prune_states(t, score_mat, beam_size)
                # update tracked active states -- to consider renaming it          
                accum_activestates[t,t] = accum_activestates[t,t].intersection(topk_states)
                #^print('score_mat[{},:] = {} '.format(j, score_mat[j,:]))
                #^print("topk_states ", topk_states)
                if(y_ref):
                    if(y_ref[t-1] not in topk_states):
                        viol_index.append(t)
                        if(stop_off_beam):
                            T = t
                            break
        if(K == 1):
            # decoding the sequence
            y_c_T = numpy.argmax(score_mat[T:])
            Y_decoded = [y_c_T]
            counter = 0
            for t in reversed(range(2, T+1)):
                Y_decoded.append(backpointer[t, Y_decoded[counter]])
                counter += 1
            Y_decoded.reverse()
           
            Y_decoded = [Y_codebook_rev[y_code] for y_code in Y_decoded]
            return(Y_decoded, viol_index)
        else:
            asearcher = FO_AStarSearcher(Y_codebook_rev)
            topK = asearcher.search(score_mat, backpointer, T, K)
            return(topK, viol_index)
    
    def perstate_posterior_decoding(self, w, seq_id):
        """decode sequences using posterior probability (per state) decoder 
                
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
                          
        """
        Y_codebook_rev = self.model.Y_codebook_rev
        # get alpha, beta and Z
        l = OrderedDict()
        l['activefeatures'] = (seq_id, )
        l['alpha'] = (w, seq_id)
        l['beta'] = (w, seq_id)
        self.check_cached_info(seq_id, l)
        
        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"]
        Z = self.seqs_info[seq_id]["Z"]
#         print("alpha \n {}".format(alpha))
#         print("beta \n {}".format(beta))
        score_mat = alpha + beta - Z
#         print("score mat is \n {}".format(score_mat))
        # remove the corner cases t=0  and t=T+1
        score_mat_ = score_mat[:,1:-1]
        max_indices = list(numpy.argmax(score_mat_, axis = 0))
#         print("max indices \n {}".format(max_indices))
        
        Y_decoded = max_indices
        Y_decoded = [Y_codebook_rev[y_code] for y_code in Y_decoded]
        return(Y_decoded)
    
    def validate_forward_backward_pass(self, w, seq_id):
        """check the validity of the forward backward pass 
                 
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
                                           
        """
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
        Z_beta = numpy.max(beta[0, :])
        raw_diff = numpy.abs(Z_alpha - Z_beta)

        print("alpha[-1,:] = {}".format(alpha[-1,:]))
        print("beta[0,:] = {}".format(beta[0,:]))
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
    