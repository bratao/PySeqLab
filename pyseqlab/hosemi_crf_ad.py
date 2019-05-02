'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''
import numpy
from .linear_chain_crf import LCRFModelRepresentation, LCRF
from .utilities import HOSemi_AStarSearcher, vectorized_logsumexp, generate_partitions, generate_partition_boundaries

class HOSemiCRFADModelRepresentation(LCRFModelRepresentation):
    r"""Model representation that will hold data structures to be used in :class:`HOSemiCRF` class
    
      Attributes:
          P_codebook: set of proper prefixes of the elements in the set of patterns :attr:`Z_codebook`
                      e.g. {'':0, 'P':1, 'L':2, 'O':3, 'L|O':4, ...}
          P_codebook_rev: reversed codebook of :attr:`P_codebook`
                          e.g. {0:'', 1:'P', 2:'L', 3:'O', 4:'L|O', ...}
          P_len: dictionary comprising the length (i.e. number of elements) of elements in :attr:`P_codebook`
                 e.g. {'':0, 'P':1, 'L':1, 'O':1, 'L|O':2, ...} 
          P_elems: dictionary comprising the composing elements of every prefix in :attr:`P_codebook`
                   e.g. {'':('',), 'P':('P',), 'L':('L',), 'O':('O',), 'L|O':('L','O'), ...}
          P_numchar: dictionary comprising the number of characters for every prefix in :attr:`P_codebook`
                     e.g. {'':0, 'P':1, 'L':1, 'O':1, 'L|O':3, ...} 
          f_transition: a dictionary representing forward transition data structure having the form:
                        {pi:{pky, (pk, y)}} where pi represents the longest prefix element in :attr:`P_codebook`
                        for pky (representing the concatenation of elements in :attr:`P_codebook` and :attr:`Y_codebook`)         
          pky_codebook: generate a codebook for the elements of the set PY (the product of set P and Y)
          pi_pky_map: a map between P elements and PY elements 
          z_pky_map: a map between elements of the Z set and PY set    
                     it has the form/template {ypattern:[pky_elements]}           
          z_pi_piy_map: a map between elements of the Z set and PY set    
                        it has the form/template {ypattern:(pk, pky, pi)}        

    """ 
    def __init__(self):
        # call super class
        super().__init__()
        self.P_codebook = None
        self.P_codebook_rev = None
        self.P_len = None
        self.P_elems = None
        self.P_numchar = None
        self.f_transition = None
        self.pky_codebook = None
        self.pi_pky_map = None
        self.z_pky_map = None
        self.z_pi_piy_map = None
        
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
        
        self.P_codebook = self.get_forward_states()
        self.P_codebook_rev = self.get_P_codebook_rev()
        self.P_len, self.P_elems, self.P_numchar = self.get_P_info()
        
        self.f_transition = self.get_forward_transition()
        
        self.pky_codebook = self.get_pky_codebook()
        self.pi_pky_map = self.get_pi_pky_map()
        
        self.z_pky_map, self.z_pi_piy_map = self.map_pky_z()

    def get_forward_states(self):
        """create set of forward states (referred to set P) and map each element to unique code
           
           P is set of proper prefixes of the elements in :attr:`Z_codebook` set
        """
        Y_codebook = self.Y_codebook
        Z_elems = self.Z_elems
        Z_len = self.Z_len
        P = {}
        for z_patt in Z_elems:
            elems = Z_elems[z_patt]
            z_len = Z_len[z_patt]
            for i in range(z_len-1):
                P["|".join(elems[:i+1])] = 1
        for y in Y_codebook:
            P[y] = 1
        # empty element         
        P[""] = 1
        P_codebook = {s:i for (i, s) in enumerate(P)}
        #print("P_codebook ", P_codebook)
        return(P_codebook) 
    
    def get_P_codebook_rev(self):
        """generate reversed codebook of :attr:`P_codebook`
        """
        P_codebook = self.P_codebook
        P_codebook_rev = {code:pi for pi, code in P_codebook.items()}
        return(P_codebook_rev)
    
    def get_P_info(self): 
        """get the properties of P set (proper prefixes)
        """
        P_codebook = self.P_codebook
        P_len = {}
        P_numchar = {}
        P_elems = {}
        
        for pi in P_codebook:
            elems = pi.split("|")
            P_elems[pi] = elems 
            if(pi == ""):
                P_len[pi] = 0
                P_numchar[pi] = 0
                
            else:
                P_len[pi] = len(elems)
                P_numchar[pi] = len(pi)
        return(P_len, P_elems, P_numchar)
    
                
    def get_forward_transition(self):
        """generate forward transition data structure
        
           Main tasks:
               - create a set PY from the product of P and Y sets
               - for each element in PY, determine the longest suffix existing in set P
               - include all this info in :attr:`f_transition` dictionary
           
        """
        Y_codebook = self.Y_codebook
        P_codebook = self.P_codebook
        P_numchar = self.P_numchar
        Z_numchar = self.Z_numchar
        
#         pk_y= {}
#         for p in P_codebook:
#             for y in Y_codebook:
#                 pk_y[(p, y)] = 1
        pk_y = {(p,y) for p in P_codebook for y in Y_codebook}

        pk_y_suffix = {}
        for p in P_codebook:
            if(p != ""):
                len_p = P_numchar[p]
                for (pk, y) in pk_y:
                    ref_str = pk + "|" + y
                    if(pk == ""):
                        len_ref = Z_numchar[y] + 1
                    else:
                        len_ref = P_numchar[pk] + Z_numchar[y] + 1

                    start_pos = len_ref - len_p

                    if(start_pos>=0):
                        # check suffix relation
                        check = ref_str[start_pos:] == p
                        #check = self.check_suffix(p, ref_str)
                        if(check):
                            if((pk, y) in pk_y_suffix):
                                pk_y_suffix[(pk, y)].append(p)
                            else:
                                pk_y_suffix[(pk, y)] = [p]
                            
        pk_y_suffix = self.keep_longest_elems(pk_y_suffix)
        f_transition = {}
        
        for (pk, y), pi in pk_y_suffix.items():
            if(pk == ""):
                elmkey = y
            else:
                elmkey = pk + "|" + y
            if(pi in f_transition):
                f_transition[pi][elmkey] = (pk, y)
            else:
                f_transition[pi] = {elmkey:(pk, y)}
        #print("f_transition ", f_transition)
        return(f_transition)
    
    def get_pky_codebook(self):
        """generate a codebook for the elements of the set PY (the product of set P and Y)
        """
        f_transition = self.f_transition
        pky_codebook = {}
        counter = 0
        for pi in f_transition:
            for pky in f_transition[pi]:
                pky_codebook[pky] = counter
                counter += 1
        return(pky_codebook)
    

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
                                z_pi_piy[z][2].append(P_codebook[pi])
                            else:
                                z_pky[z] = [pky_c]
                                z_pi_piy[z] = ([pk_c], [pky_c], [P_codebook[pi]])
        return(z_pky, z_pi_piy) 
    
    
    def get_pi_pky_map(self):
        """ generate map between P elements and PY elements
        
            Main tasks:
                - for every element in PY, determine the longest suffix in P
                - determine the two components in PY (i.e. p and y element)
                - represent this info in a dictionary that will be used for forward/alpha matrix
        """
        
        f_transition = self.f_transition
        pky_codebook = self.pky_codebook
        P_codebook = self.P_codebook
        pi_pky_map = {}
        for pi in f_transition:
            pi_pky_map[pi]=[[],[]]
            for pky, (pk, __) in f_transition[pi].items():
                pi_pky_map[pi][0].append(pky_codebook[pky])
                pi_pky_map[pi][1].append(P_codebook[pk])
            # convert list to numpy arrays
#             for i in range(2):
#                 pi_pky_map[pi][i] = numpy.array(pi_pky_map[pi][i])
#             pi_pky_map[pi] = tuple(pi_pky_map[pi])

        return(pi_pky_map)
    
    def filter_activated_states(self, activated_states, accum_active_states, curr_boundary):
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
        
        # generate partition boundaries
        depth_node_map = {}
        generate_partitions(curr_boundary, self.L, self.max_patt_len, {}, depth_node_map, None)
        partition_boundaries = generate_partition_boundaries(depth_node_map)

        for z_len in activated_states:
            if(z_len == 1):
                continue
            if(z_len in partition_boundaries):
                partitions = partition_boundaries[z_len]
                filtered_activestates[z_len] = set()
                for partition in partitions:
                    for z_patt in activated_states[z_len]:
                        check = True
                        zelems = Z_elems[z_patt]
                        for i in range(z_len):
                            bound = partition[i]
                            if(zelems[i] not in accum_active_states[bound]):
                                check = False
                                break
                        if(check):                        
                            filtered_activestates[z_len].add(z_patt)
        return(filtered_activestates)
                      
class HOSemiCRFAD(LCRF):
    """higher-order semi-CRF model that uses algorithmic differentiation in gradient computation
    
      Args:
          model: an instance of :class:`HOSemiCRFADModelRepresentation` class
          seqs_representer: an instance of :class:`SeqsRepresenter` class
          seqs_info: dictionary holding sequences info
       
      Keyword Arguments:
          load_info_fromdisk: integer from 0 to 5 specifying number of cached data 
                              to be kept in memory. 0 means keep everything while
                              5 means load everything from disk
                               
      Attributes:
          model: an instance of :class:`HOSemiCRFADModelRepresentation` class
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

    def cached_entitites(self, load_info_fromdisk):
        """construct list of names of cached entities in memory
        """
        def_cached_entities = super().cached_entitites(load_info_fromdisk)
        inmemory_info = ["alpha", "Z", "beta", "fpotential"]
        def_cached_entities += inmemory_info
        return(def_cached_entities)
    
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
        P_len = model.P_len
        P_codebook = model.P_codebook
        T = self.seqs_info[seq_id]["T"]
        L = self.model.L
        activefeatures = self.seqs_info[seq_id]['activefeatures']
        alpha = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        alpha[0,P_codebook[""]] = 0
        fpotential_perboundary = {}
                       
        for j in range(1, T+1): 
            accumulator = numpy.ones((len(P_codebook), L), dtype='longdouble') * -numpy.inf
            for d in range(L):
                u = j - d
                if(u <= 0):
                    break
                v = j
                f_potential = self.compute_fpotential(w, activefeatures[u,v])
                fpotential_perboundary[u,v] = f_potential
                for pi in pi_pky_map:
                    if(j>=P_len[pi]):
                        pi_c = P_codebook[pi]
                        pky_c_list, pk_c_list = pi_pky_map[pi]
                        vec = f_potential[pky_c_list] + alpha[u-1, pk_c_list]
                        accumulator[pi_c, d] = vectorized_logsumexp(vec)
            for pi in pi_pky_map:
                if(j>=P_len[pi]):
                    pi_c = P_codebook[pi]
                    if(L>1):
                        alpha[j, pi_c] = vectorized_logsumexp(accumulator[pi_c, :])
                    else:
                        alpha[j, pi_c] = accumulator[pi_c, 0]
         
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
        L = model.L
        fpotential_perboundary = self.seqs_info[seq_id]['fpotential']
        
        beta = numpy.ones((T+2, len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        beta[T+1, :] = 0
               
        for j in reversed(range(1, T+1)):
            accum_mat = numpy.ones((len_P, L), dtype='longdouble') * (-numpy.inf)
            for d in range(L):
                track_comp = numpy.ones((len_P, len_P), dtype='longdouble') * (-numpy.inf)
                u = j
                v = j + d
                if(v>T):
                    break
                f_potential = fpotential_perboundary[u, v]
                for pi in pi_pky_map:
                    pi_c = P_codebook[pi]
                    pky_c_list, pk_c_list = pi_pky_map[pi]
                    vec = f_potential[pky_c_list] + beta[v+1, pi_c]
                    track_comp[pk_c_list, pi_c] = vec
                for p_c in P_codebook.values():
                    accum_mat[p_c, d] = vectorized_logsumexp(track_comp[p_c, :]) 
            for p_c in P_codebook.values():
                beta[u, p_c] = vectorized_logsumexp(accum_mat[p_c, :])  
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
 
        P_marginals = numpy.zeros((L, T+1, len(self.model.Z_codebook)), dtype='longdouble')
         
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
                    P_marginals[d, j, Z_codebook[z]] = numpy.exp(vectorized_logsumexp(numerator) - Z)
        
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
            u, v = boundary
            d = v-u
            for z_patt in features_dict:
                w_indx, f_val = features_dict[z_patt]
                grad[w_indx] += f_val * P_marginals[d, u, Z_codebook[z_patt]]
                      
    def prune_states(self, score_vec, beam_size):
        """prune states that fall off the specified beam size
        
           Args:
               score_vec: score matrix 
               beam_size: specified size of the beam (integer)
        """
        P_codebook_rev = self.model.P_codebook_rev
        P_elems = self.model.P_elems

        # using argpartition as better alternative to argsort
        indx_partitioned_pi = numpy.argpartition(-score_vec, beam_size)
        # identify top-k states/pi
        indx_topk_pi = indx_partitioned_pi[:beam_size]
        # get topk states
        topk_pi = {P_codebook_rev[indx] for indx in indx_topk_pi}
        topk_states = {P_elems[pi][-1] for pi in topk_pi}
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
                  A* searcher with viterbi will be used to generate k-decoded list
                          
        """  
        model = self.model
        P_elems = model.P_elems
        pi_pky_map = model.pi_pky_map
        P_codebook = model.P_codebook
        P_codebook_rev = model.P_codebook_rev
        L = model.L
        len_P = len(P_codebook)
        num_states = model.num_states
        
        T = self.seqs_info[seq_id]["T"]
        # records max score at every time step
        delta = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        pi_mat = numpy.ones((len_P, L), dtype='longdouble')* (-numpy.inf)
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
                # reset pi_mat at every loop
                pi_mat.fill(-numpy.inf)
                backpointer = {}
                for d in range(L):
                    u = j-d
                    if(u <= 0):
                        break
                    v = j
                    boundary = (u, v)
                    # vector of size len(pky)
                    f_potential = self.compute_fpotential(w, active_features[boundary])
                    for pi in pi_pky_map:
                        pi_c = P_codebook[pi]
                        pky_c_list, pk_c_list = pi_pky_map[pi]
                        vec = f_potential[pky_c_list] + delta[u-1, pk_c_list]
#                         print("f_potential[pky_c_list] ", f_potential[pky_c_list])
#                         print("delta[u-1, pk_c_list] ", delta[u-1, pk_c_list])
#                         print("vec ", vec)
                        pi_mat[pi_c, d] = numpy.max(vec)
                        argmax_indx = numpy.argmax(vec)
                        #print("argmax chosen ", argmax_ind)
                        pk_c_max = pk_c_list[argmax_indx]
                        #print('pk_c ', pk_c)
                        pk = P_codebook_rev[pk_c_max]
                        y = P_elems[pk][-1]
                        backpointer[d, pi_c] =  (pk_c_max, y)
#                 print("backpointer  ")
#                 print(backpointer)
#                 print("pi_mat")
#                 print(pi_mat)
                # get the max for each pi across all segment lengths
                for pi in pi_pky_map:
                    pi_c = P_codebook[pi]
                    delta[j, pi_c] = numpy.max(pi_mat[pi_c, :])
                    argmax_indx = numpy.argmax(pi_mat[pi_c, :])     
                    pk_c, y = backpointer[argmax_indx, pi_c] 
                    back_track[j, pi_c] = (argmax_indx, pk_c, y)
#             print("delta ")
#             print(delta)
#             print("backtrack ")
#             print(back_track)
        else:
            # case of inexact search and decoding
            l = {}
            l['seg_features'] = (seq_id, )
            self.check_cached_info(seq_id, l)
            # tracks active states by boundary
            accum_activestates = {}
            for j in range(1, T+1):
                # reset pi_mat at every loop
                pi_mat.fill(-numpy.inf)
                backpointer = {}
                for d in range(L):
                    u = j-d
                    if(u <= 0):
                        break
                    v = j
                    boundary = (u, v)
                    active_features = self.identify_activefeatures(seq_id, boundary, accum_activestates)
                    # vector of size len(pky)
                    f_potential = self.compute_fpotential(w, active_features)
                    for pi in pi_pky_map:
                        pi_c = P_codebook[pi]
                        pky_c_list, pk_c_list = pi_pky_map[pi]
                        vec = f_potential[pky_c_list] + delta[u-1, pk_c_list]
                        pi_mat[pi_c, d] = numpy.max(vec)
                        argmax_indx = numpy.argmax(vec)
                        #print("argmax chosen ", argmax_ind)
                        pk_c_max = pk_c_list[argmax_indx]
                        #print('pk_c ', pk_c)
                        pk = P_codebook_rev[pk_c_max]
                        y = P_elems[pk][-1]
                        backpointer[d, pi_c] =  (pk_c_max, y)
                    
                    topk_states = self.prune_states(pi_mat[:,d], beam_size)
                    # update tracked active states -- to consider renaming it          
                    accum_activestates[boundary] = accum_activestates[boundary].intersection(topk_states)
                # get the max for each pi across all segment lengths
                for pi in pi_pky_map:
                    pi_c = P_codebook[pi]
                    delta[j, pi_c] = numpy.max(pi_mat[pi_c, :])
                    argmax_indx = numpy.argmax(pi_mat[pi_c, :])     
                    pk_c, y = backpointer[argmax_indx, pi_c] 
                    back_track[j, pi_c] = (argmax_indx, pk_c, y)
                    
                # in case we are using viterbi for learning    
                if(y_ref):    
                    topk_states = self.prune_states(delta[j, :], beam_size)           
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
            
            d, pt_c, yt = back_track[T, p_T_c]
            for _ in range(d+1):
                Y_decoded.append(y_T)
                
            t = T - d - 1
            while t>0:
                new_d, new_pt_c, new_yt = back_track[t, pt_c]
                for _ in range(new_d+1):
                    Y_decoded.append(yt)
                t = t - new_d -1
                pt_c = new_pt_c
                yt = new_yt
                
            Y_decoded.reverse()
            #print("y_decoded ", Y_decoded)
            return(Y_decoded, viol_index)
        else:
            asearcher = HOSemi_AStarSearcher(P_codebook_rev, P_elems)
            topK = asearcher.search(delta, back_track, T, K)
#             print('topk ', topK)
            return(topK, viol_index)
    
if __name__ == "__main__":
    pass
