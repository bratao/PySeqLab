'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''

import os
from copy import deepcopy
from collections import OrderedDict
import numpy
from .utilities import ReaderWriter, create_directory, vectorized_logsumexp

class LCRFModelRepresentation(object):
    """Model representation that will hold data structures to be used in :class:`LCRF` class
           
       Attributes:
           modelfeatures: set of features defining the model
           modelfeatures_codebook: dictionary mapping each features in :attr:`modelfeatures` to a unique code
           Y_codebook: dictionary mapping the set of states (i.e. tags) to a unique code each
           L: length of longest segment
           Z_codebook: dictionary for the set Z, mapping each element to unique number/code        
           Z_len: dictionary comprising the length of each element in :attr:`Z_codebook`
           Z_elems: dictionary comprising the composing elements of each member in the Z set (:attr:`Z_codebook`)
           Z_numchar: dictionary comprising the number of characters of each member in the Z set (:attr:`Z_codebook`)
           patts_len: set of lengths extracted from :attr:`Z_len` (i.e. set(Z_len.values()))
           max_patts_len: maximum pattern length used in the model
           modelfeatures_inverted: inverted model features (i.e inverting the :attr:`modelfeatures` dictionary)
           ypatt_features: state features (i.e. y pattern features) that depend only on the states
           ypatt_activestates: possible/potential activated y patterns/features using the observation features
           num_features: total number of features in the model
           num_states: total number of states in the model
    """ 
    def __init__(self):
        self.modelfeatures = None
        self.modelfeatures_codebook = None
        self.Y_codebook = None
        self.L = None
        self.Z_codebook = None
        self.Z_len = None
        self.Z_elems = None
        self.Z_numchar= None
        self.patts_len = None
        self.max_patt_len = None
        self.modelfeatures_inverted = None
        self.ypatt_features = None
        self.ypatt_activestates = None
        self.num_features = None
        self.num_states = None
        
    def setup_model(self, modelfeatures, states, L):
        """setup and create the model representation
        
           Creates all maps and codebooks needed by the :class:`LCRF` class
           
           Args:
               modelfeatures: set of features defining the model
               states: set of states (i.e. tags)
               L: length of longest segment
        """
        self.modelfeatures = modelfeatures
        self.modelfeatures_codebook = self.get_modelfeatures_codebook()
        self.Y_codebook = self.get_modelstates_codebook(states)
        self.L = L
        self.generate_instance_properties()
    
    def generate_instance_properties(self):
        """generate instance properties that will be later used by :class:`LCRF` class
        """
        self.Z_codebook = self.get_Z_pattern()
        self.Z_len, self.Z_elems, self.Z_numchar = self.get_Z_info()
        self.patts_len = set(self.Z_len.values())
        self.max_patt_len = max(self.patts_len)

        self.modelfeatures_inverted, self.ypatt_features = self.get_inverted_modelfeatures()
        self.ypatt_activestates = self.find_activated_states(self.ypatt_features, self.patts_len)
        
        self.num_features = self.get_num_features()
        self.num_states = self.get_num_states()

    def get_modelfeatures_codebook(self):
        r"""setup model features codebook
        
           it flatten :attr:`modelfeatures` and map each element to a unique code
           :attr:`modelfeatures` are represented in a dictionary with this form::
           
               {y_patt_1:{featureA:value, featureB:value, ...}
                y_patt_2:{featureA:value, featureC:value, ...}}
               
           Example::
           
               modelfeatures:
                    {'B-PP': Counter({'w[0]=at': 1,
                                      'w[0]=by': 1,
                                      'w[0]=for': 4,
                                      ...
                                     }),
                    'B-PP|B-NP': Counter({'w[0]=16': 1,
                                          'w[0]=July': 1,
                                          'w[0]=Nomura': 1,
                                          ...
                                          }),
                                 ...
                    }
               modelfeatures_codebook:
                   {('B-PP','w[0]=at'): 1,
                    ('B-PP','w[0]=by'): 2,
                    ('B-PP','w[0]=for'): 3,
                    ...
                   }
                   
        """
        modelfeatures = self.modelfeatures
        codebook = {}
        code = 0
        for y_patt, featuresum in modelfeatures.items():
            for feature in featuresum:
                #fkey = y_patt + "&&" + feature
                codebook[(y_patt, feature)] = code
                code += 1
        return(codebook)


    def get_modelstates_codebook(self, states):
        """create states codebook by mapping each state to a unique code/number
        
           Args:
               states: set of tags identified in training sequences
           
           Example::
           
               states = {'B-PP', 'B-NP', ...}
               states_codebook = {'B-PP':1, 'B-NP':2 ...}
        """
        return({s:i for (i, s) in enumerate(states)})
        
    def get_Z_pattern(self):
        """create a codebook from set Z by mapping each element to unique number/code
        
           Z is set of y patterns used in the model features
           
           Example::
           
               Z = {'O|B-VP|B-NP', 'O|B-VP', 'O', 'B-VP', 'B-NP', ...}
               Z_codebook = {'O|B-VP|B-NP':1, 'O|B-VP':2, 'O':3, 'B-VP':5, 'B-NP':4, ...}
        """
        modelfeatures = self.modelfeatures
        Z_codebook = {y_patt:index for index, y_patt in enumerate(modelfeatures)}
        return(Z_codebook)
    
    def get_Z_info(self):
        """get the properties of Z set
        """
        Z_codebook = self.Z_codebook
        Z_len = {}
        Z_elems = {}
        Z_numchar = {}
        for z in Z_codebook:
            elems = z.split("|")
            Z_len[z] = len(elems)
            Z_elems[z] = elems
            Z_numchar[z] = len(z)            
        return(Z_len, Z_elems, Z_numchar)
    

    def get_inverted_modelfeatures(self):
        r"""invert :attr:`modelfeatures` instance variable
        
            Example::
            
               modelfeatures_inverted = 
               {'w[0]=take': {1: {'I-VP'}, 2: {'I-VP|I-VP'}, 3: {'I-VP|I-VP|I-VP'}},
                'w[0]=the': {1: {'B-NP'},
                             2: {'B-PP|B-NP', 'I-VP|B-NP'},
                             3: {'B-NP|B-PP|B-NP', B-VP|I-VP|B-NP', ...}
                            },
                            ...
               }
               
               ypatt_features = {'B-NP', 'B-PP|B-NP', ..}
        """
        modelfeatures = self.modelfeatures
        Z_len = self.Z_len
        inverted_features = {}
        ypatt_features = set()
        
        for y_patt, featuredict in modelfeatures.items():
            z_len = Z_len[y_patt]
            # get features that are based only on y_patts
            if(y_patt in featuredict):
                ypatt_features.add(y_patt)
            for feature in featuredict:
                if(feature in inverted_features):
                    if(z_len in inverted_features[feature]):
                        inverted_features[feature][z_len].add(y_patt)
                    else:
                        s = set()
                        s.add(y_patt)                      
                        inverted_features[feature][z_len] = s
                else:
                    s = set()
                    s.add(y_patt)
                    inverted_features[feature] = {z_len:s}
        return(inverted_features, ypatt_features)
        
    def keep_longest_elems(self, s):
        """used to figure out longest suffix and prefix on sets 
        """
        longest_elems = {}
        for tup, l in s.items():
            longest_elems[tup] = max(l, key = len)
        return(longest_elems)
                  
    def check_suffix(self, token, ref_str):
        # check if ref_str ends with the token
#         return(ref_str[len(ref_str)-len(token):] == token)
        return(ref_str.endswith(token))
    
    def check_prefix(self, token, ref_str):
        # check if ref_str starts with a token
#         return(ref_str[:len(token)] == token)
        return(ref_str.startswith(token))
    
    def get_num_features(self):
        """return total number of features in the model
        """
        return(len(self.modelfeatures_codebook))
    
    def get_num_states(self):
        """return total number of states identified by the model in the training set
        """
        return(len(self.Y_codebook)) 
    
    def represent_globalfeatures(self, seq_featuresum):
        """represent features extracted from sequences using :attr:`modelfeatures_codebook`
        
           Args:
               seq_featuresum: dictionary of sequence global features representing F(X,Y)
        """
        modelfeatures_codebook = self.modelfeatures_codebook
        windx_fval = {}
        for y_patt, seg_features in seq_featuresum.items():
            for featurename in seg_features: 
                #fkey = y_patt + "&&" + featurename
                fkey = (y_patt, featurename)
                if(fkey in modelfeatures_codebook):
                    windx_fval[modelfeatures_codebook[fkey]] = seg_features[featurename]
        count = len(windx_fval)
        return(numpy.fromiter(windx_fval.keys(), numpy.uint32, count), 
               numpy.fromiter(windx_fval.values(), numpy.float64, count))
        
    def represent_activefeatures(self, activefeatures):
        windx_fval = {}
        for z_patt in activefeatures:
            count = len(activefeatures[z_patt])
            windx_fval[z_patt] = (numpy.fromiter(activefeatures[z_patt].keys(), numpy.uint32, count), 
                                  numpy.fromiter(activefeatures[z_patt].values(), numpy.float64, count))
        return(windx_fval)
    
    def accumulate_activefeatures(self, activefeatures, accumfeatures):
        for z_patt in activefeatures:
                if(z_patt in accumfeatures):
                    accumfeatures[z_patt].update(activefeatures[z_patt])
                else:
                    accumfeatures[z_patt] = activefeatures[z_patt]
                   
    def join_segfeatures_filteredstates(self, seg_features, filtered_states):
        """represent detected active features while parsing sequences
         
           Args:
               activestates: dictionary of the form {'patt_len':{patt_1, patt_2, ...}}
               seg_features: dictionary of the observation features. It has the form 
                             {featureA_name:value, featureB_name:value, ...} 
        """
        modelfeatures_codebook = self.modelfeatures_codebook   
        activefeatures = {}
        for z_len in filtered_states:
            z_patt_set = filtered_states[z_len]
            for z_patt in z_patt_set:
                windx_fval = {}
                for seg_featurename in seg_features:
                    fkey = (z_patt, seg_featurename)
#                     print("filtering ...")
#                     print("zpatt ", z_patt)
#                     print("fkey ", fkey)
                    if(fkey in modelfeatures_codebook):
                        windx_fval[modelfeatures_codebook[fkey]] = seg_features[seg_featurename]     
                if(windx_fval):
                    activefeatures[z_patt] = windx_fval
        return(activefeatures)
    
    def represent_ypatt_filteredstates(self, filtered_states):
        """represent detected active features while parsing sequences
         
           Args:
               activestates: dictionary of the form {'patt_len':{patt_1, patt_2, ...}}
               seg_features: dictionary of the observation features. It has the form 
                             {featureA_name:value, featureB_name:value, ...} 
        """
        modelfeatures = self.modelfeatures
        modelfeatures_codebook = self.modelfeatures_codebook   
        activefeatures = {}
        for z_len in filtered_states:
            z_patt_set = filtered_states[z_len]
            for z_patt in z_patt_set:
                windx_fval = {}
                if(z_patt in modelfeatures[z_patt]):
                    fkey = (z_patt, z_patt)
#                     print("filtering ...")
#                     print("zpatt ", z_patt)
#                     print("fkey ", fkey)
                    windx_fval[modelfeatures_codebook[fkey]] = 1
                if(windx_fval):
                    activefeatures[z_patt] = windx_fval
        return(activefeatures)
    
    def find_seg_activefeatures(self, seg_features, allowed_z_len):
        """finds active features based on the observation/segment features
        
           Args:
               seg_features:
               allowed_z_len:
        """
        modelfeatures_codebook = self.modelfeatures_codebook   
        modelfeatures_inverted = self.modelfeatures_inverted
        activefeatures = {}
        # use segment features plus the activated states
        for seg_featurename in seg_features:
            if(seg_featurename in modelfeatures_inverted):
                for z_len in allowed_z_len:
                    if(z_len in modelfeatures_inverted[seg_featurename]):
                        for zpatt in modelfeatures_inverted[seg_featurename][z_len]:
                            fkey = (zpatt, seg_featurename)
#                             print("zpatt ", zpatt)
#                             print("fkey ", fkey)
                            if(zpatt in activefeatures):
                                activefeatures[zpatt][modelfeatures_codebook[fkey]] = seg_features[seg_featurename]
                            else:
                                activefeatures[zpatt] = {modelfeatures_codebook[fkey]:seg_features[seg_featurename]}
        return(activefeatures)
   
    def find_ypatt_activefeatures(self, allowed_z_len):
        """finds the label and state transition features (if applicable -- in case it is modeled)
        
           Args:
               allowed_z_len:
        """
        modelfeatures_codebook = self.modelfeatures_codebook   
        ypatt_activestates = self.ypatt_activestates
        activefeatures = {}
        # check if ypattern features are modeled
        for z_len in allowed_z_len:
            if(z_len in ypatt_activestates):
                for zpatt in ypatt_activestates[z_len]:
                    fkey = (zpatt, zpatt)
#                     print("zpatt ", zpatt)
#                     print("fkey ", fkey)
                    if(zpatt in activefeatures):
                        activefeatures[zpatt][modelfeatures_codebook[fkey]] = 1
                    else:
                        activefeatures[zpatt] = {modelfeatures_codebook[fkey]:1}

        return(activefeatures)

    def find_activated_states(self, seg_features, allowed_z_len):
        """identify possible activated y patterns/features using the observation features
        
           Args:
               seg_features: dictionary of the observation features. It has the form 
                             {featureA_name:value, featureB_name:value, ...} 
               allowed_z_len: set of permissible order/length of y features
                             {1,2,3} -> means up to third order y features are allowed
        """ 
        modelfeatures_inverted = self.modelfeatures_inverted
        active_states = {}
        for feature in seg_features:
            if(feature in modelfeatures_inverted):
                factivestates = modelfeatures_inverted[feature]
                for z_len in factivestates:
                    if(z_len in allowed_z_len):
                        if(z_len in active_states):
                            active_states[z_len].update(factivestates[z_len])
                        else:
                            active_states[z_len] = set(factivestates[z_len])
                #print("active_states from func ", active_states)
        return(active_states)
    
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
    
    def save(self, folder_dir):
        """save main model data structures
        """
        model_info = {'MR_modelfeatures':self.modelfeatures, 
                      'MR_modelfeaturescodebook':self.modelfeatures_codebook, 
                      'MR_Ycodebook':self.Y_codebook,
                      'MR_L':self.L
                      }
        for name in model_info:
            ReaderWriter.dump_data(model_info[name], os.path.join(folder_dir, name))
                      
class LCRF(object):
    """linear chain CRF model 
    
      Args:
          model: an instance of :class:`LCRFModelRepresentation` class
          seqs_representer: an instance of :class:`SeqsRepresenter` class
          seqs_info: dictionary holding sequences info
       
      Keyword Args:
          load_info_fromdisk: integer from 0 to 5 specifying number of cached data 
                              to be kept in memory. 0 means keep everything while
                              5 means load everything from disk
                               
      Attributes:
          model: an instance of :class:`LCRFModelRepresentation` class
          weights: a numpy vector representing feature weights
          seqs_representer: an instance of :class:`SeqsRepresenter` class
          seqs_info: dictionary holding sequences info
          beam_size: determines the size of the beam for state pruning
          fun_dict: a function map
          def_cached_entities: a list of the names of cached entities sorted (descending)
                               based on estimated space required in memory 
                               
    """
    def __init__(self, model, seqs_representer, seqs_info, load_info_fromdisk = 5):

        self.model = model
        self.weights = numpy.zeros(model.num_features, dtype= "longdouble")
        self.seqs_representer = seqs_representer
        self.seqs_info = seqs_info
        self.func_dict = {"alpha": self._load_alpha,
                          "beta": self._load_beta,
                          "activated_states": self.load_activatedstates,
                          "seg_features": self.load_segfeatures,
                          "globalfeatures": self.load_globalfeatures,
                          "globalfeatures_per_boundary": self.load_globalfeatures,
                          "activefeatures": self.load_activefeatures,
                          "Y":self._load_Y}
        
        self.def_cached_entities = self.cached_entitites(load_info_fromdisk)
        # default beam size covers all available states
        self.beam_size = len(self.model.Y_codebook)

    def cached_entitites(self, load_info_fromdisk):
        """construct list of names of cached entities in memory
        """
        ondisk_info = ["activefeatures", "seg_features", "activated_states", "globalfeatures_per_boundary", "globalfeatures", "Y"]
        def_cached_entities = ondisk_info[:load_info_fromdisk]
        return(def_cached_entities)
    
    @property
    def seqs_info(self):
        return self._seqs_info
    @seqs_info.setter
    def seqs_info(self, info_dict):
        # make a copy of the passed seqs_info dictionary
        self._seqs_info = deepcopy(info_dict)     

    def identify_activefeatures(self, seq_id, boundary, accum_activestates, apply_filter = True):
        """determine model active features for a given sequence at defined boundary
           
           Main task:
               - determine model active features in a given boundary
               - update the accum_activestates dictionary   
                     
           Args:
               seq_id: integer representing unique id assigned to the sequence
               boundary: tuple (u,v) defining the boundary under consideration
               accum_activestates: dictionary of the form {(u,v):{state_1, state_2, ...}}
                                   it keeps track of the active states in each boundary
        """
        
        model = self.model
        max_patt_len = model.max_patt_len
        patts_len = model.patts_len
        ypatt_features = model.ypatt_features
        # default length of a state/tag
        state_len = 1
        seg_features = self.seqs_info[seq_id]['seg_features'][boundary]
        
        start_state_flag = False
        if('__START__' in model.Y_codebook): # first order model is used with max_patt_len = 2
            start_state_flag = True
            apply_filter = True
            
        u, __ = boundary
        if(u == 1 and start_state_flag):
            accum_activestates[0,0] = {'__START__'}
#         print("boundary ", boundary)
#         print('seg_features ', seg_features)
#         print("accum_activestates ", accum_activestates)
        if(u < max_patt_len):
            # case when we use first-order CRF model -- max_patt_len = 2
            if(start_state_flag):
                max_len = max_patt_len
            else:
                max_len = u
        else:
            max_len = max_patt_len
        # determine allowed z patterns length (i.e. pattern order)
        allowed_z_len = {z_len for z_len in patts_len if z_len <= max_len}
        
#         print("apply filter ", apply_filter)
        if(not apply_filter): # case of no filtering
            seg_activefeatures = model.find_seg_activefeatures(seg_features, allowed_z_len)
            ypatt_activefeatures = model.find_ypatt_activefeatures(allowed_z_len)
            # combine both
            accumfeatures = seg_activefeatures
            model.accumulate_activefeatures(ypatt_activefeatures, accumfeatures)  
                 
        else: # case of filtering
            seg_activefeatures = model.find_seg_activefeatures(seg_features, {state_len})
            ypatt_activefeatures = model.find_ypatt_activefeatures({state_len})
            # determine activate states with order 0 (i.e. length =1)
            zero_order_activatedstates = set(seg_activefeatures.keys())
            zero_order_activatedstates.update(set(ypatt_activefeatures.keys()))
            accum_activestates[boundary] = zero_order_activatedstates
            # remove states with zero order (i.e. length = 1)
            allowed_z_len.remove(state_len)
            seg_activated_states = model.find_activated_states(seg_features, allowed_z_len)
            seg_filtered_states = model.filter_activated_states(seg_activated_states, accum_activestates, boundary)
            seg_activefeatures_addendum = model.join_segfeatures_filteredstates(seg_features, seg_filtered_states)
            
            ypatt_activated_states = model.find_activated_states(ypatt_features, allowed_z_len)
            ypatt_filtered_states = model.filter_activated_states(ypatt_activated_states, accum_activestates, boundary)
            ypatt_activefeatures_addendum = model.represent_ypatt_filteredstates(ypatt_filtered_states) 
            
            # join all the active features
            accumfeatures = seg_activefeatures
            model.accumulate_activefeatures(ypatt_activefeatures, accumfeatures)
            model.accumulate_activefeatures(seg_activefeatures_addendum, accumfeatures)
            model.accumulate_activefeatures(ypatt_activefeatures_addendum, accumfeatures)

        activefeatures = model.represent_activefeatures(accumfeatures)

        return(activefeatures)  
    
    def generate_activefeatures(self, seq_id):
        """construct a dictionary of model active features identified given a sequence
        
           Main task:
               - generate active features for every boundary of the sequence 

           Args:
               seq_id: integer representing unique id assigned to the sequence
               
        """
        # to be used when using gradient-based methods for learning
        T = self.seqs_info[seq_id]["T"]
        L = self.model.L
        accum_activestates = {}
        activefeatures_perboundary = {}
        ypatt_activestates = self.model.ypatt_activestates
        # zero-order state/tag has state_len = 1 (i.e. using only one state)
        state_len = 1
        apply_filter = True
        # check if we are modeling label bias terms or having categorical features
        if(state_len in ypatt_activestates or self.seqs_representer.attr_scaler):
            apply_filter = False
            
        for j in range(1, T+1):
            for d in range(L):
                u = j - d
                if(u <= 0):
                    break
                v = j
                boundary = (u, v)
                # identify active features
                active_features = self.identify_activefeatures(seq_id, boundary, accum_activestates, apply_filter=apply_filter)
                activefeatures_perboundary[boundary] = active_features
        return(activefeatures_perboundary)

               
    def compute_forward_vec(self, w, seq_id):
        """compute the forward matrix (alpha matrix)
           
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
              
           .. warning::
           
              implementation of this method is in the child class
        """
        # to be implemented in the child class
        pass

    def compute_backward_vec(self, w, seq_id):
        """compute the backward matrix (beta matrix)
        
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
            
           .. warning::
           
              implementation of this method is in the child class
        """
        # to be implemented in the child class
        pass  
    
    def compute_marginals(self, seq_id):
        """compute the marginal (i.e. probability of each y pattern at each position)
        
           Args:
               seq_id: integer representing unique id assigned to the sequence
            
           .. warning::
           
              implementation of this method is in the child class
        """
        # to be implemented in the child class
        pass
    
    
    def compute_feature_expectation(self, seq_id, P_marginals):
        """compute the features expectations (i.e. expected count of the feature based on learned model)
        
           Args:
               seq_id: integer representing unique id assigned to the sequence
               P_marginals: probability matrix for y patterns at each position in time
            
           .. warning::
           
              implementation of this method is in the child class
        """       
        # to be implemented in the child class
        pass     

    def compute_seq_loglikelihood(self, w, seq_id):
        """computes the conditional log-likelihood of a sequence (i.e. :math:`p(Y|X;w)`) 
           
           it is used as a cost function for the single sequence when trying to estimate parameters w
           
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
            
        """
#         print("-"*40)
#         print("... Evaluating compute_seq_loglikelihood() ...")
        
        # we need global features and alpha matrix to be ready -- order is important
        l = OrderedDict()
        l['globalfeatures'] = (seq_id, False)
        l['activefeatures'] = (seq_id, )
        l['alpha'] = (w, seq_id)
        
        self.check_cached_info(seq_id, l)
        # get the p(X;w) -- probability of the sequence under parameter w
        Z = self.seqs_info[seq_id]["Z"]
        w_indx, f_val = self.seqs_info[seq_id]["globalfeatures"]
        # log(p(Y|X;w))
        loglikelihood = numpy.dot(w[w_indx], f_val) - Z 
        self.seqs_info[seq_id]["loglikelihood"] = loglikelihood

        return(loglikelihood)
    
    
    def compute_seq_gradient(self, w, seq_id, grad):
        r"""compute the gradient of conditional log-likelihood with respect to the parameters vector w (:math:`\frac{\partial p(Y|X;w)}{\partial w}`)
           
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
        """
#         print("-"*40)
#         print("... Evaluating compute_seq_gradient() ...")

        # we need alpha, beta, global features and active features  to be ready
        l = OrderedDict()
        l['globalfeatures'] = (seq_id, False)
        l['activefeatures'] = (seq_id, )
        l['alpha'] = (w, seq_id)
        l['beta'] = (w, seq_id)
        self.check_cached_info(seq_id, l)
        # compute marginal probability of y patterns at every position
        P_marginal = self.compute_marginals(seq_id)
        # compute features expectation
        self.compute_feature_expectation(seq_id, P_marginal, grad)
        target_indx = numpy.where(grad!=0)[0]
        # get global features count of the reference sequence
        gwindx, gfval = self.seqs_info[seq_id]["globalfeatures"]
        grad[target_indx] *= -1
        grad[gwindx] += gfval
        # update target_indx
        target_indx = numpy.unique(numpy.concatenate((target_indx, gwindx)))
        #target_indx = numpy.where(grad!=0)[0]
        return(target_indx)
    
    def compute_seqs_loglikelihood(self, w, seqs_id):
        """computes the conditional log-likelihood of training sequences 
           
           it is used as a cost/objective function for the whole training sequences when trying to estimate parameters w
           
           Args:
               w: weight vector (numpy vector)
               seqs_id: list of integer representing unique ids of sequences used for training
            
        """
        seqs_loglikelihood = 0
        for seq_id in seqs_id:
            seqs_loglikelihood += self.compute_seq_loglikelihood(w, seq_id)
        return(seqs_loglikelihood)
    
    def compute_seqs_gradient(self, w, seqs_id):
        """compute the gradient of conditional log-likelihood with respect to the parameters vector w
                      
           Args:
               w: weight vector (numpy vector)
               seqs_id: list of integer representing unique ids of sequences used for training
            
        """
        seqs_grad = numpy.zeros(len(w))
        seq_grad = numpy.zeros(len(w))
        for seq_id in seqs_id:
            target_indx = self.compute_seq_gradient(w, seq_id, seq_grad) 
            seqs_grad[target_indx] += seq_grad[target_indx]
            seq_grad.fill(0)
        return(seqs_grad)
    
    def _load_alpha(self, w, seq_id):
        """compute and load the alpha matrix in :attr:`seqs_info`
         
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
                
           .. note::
            
              - seg_features (per boundary) dictionary should be available in :attr:`seqs.info`
              - activated_states (per boundary) dictionary should be available in :attr:`seqs.info`

        """
        seq_info = self.seqs_info[seq_id]
        seq_info["alpha"] = self.compute_forward_vec(w, seq_id)
        seq_info["Z"] = vectorized_logsumexp(seq_info["alpha"][-1,:])
        #print("... Computing alpha probability ...")
    
    def _load_beta(self, w, seq_id):
        """compute and load the beta matrix in :attr:`seqs_info`
        
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
                
           .. note:: 
           
              - fpotential per boundary dictionary should be available in :attr:`seqs.info`
        """
        seq_info = self.seqs_info[seq_id]
        seq_info["beta"] = self.compute_backward_vec(w, seq_id)
        #print("... Computing beta probability ...")

    def _load_Y(self, seq_id):
        """load the Y sequence and the boundaries in :attr:`seqs_info`
        
           Args:
               seq_id: integer representing unique id assigned to the sequence
                
        """
        seq = self._load_seq(seq_id, target="seq")
        self.seqs_info[seq_id]['Y'] = {'flat_y':seq.flat_y, 'boundaries':seq.y_sboundaries}
        #print("... loading Y ...")

    def load_activatedstates(self, seq_id):
        """load sequence activated states in :attr:`seqs_info`

           Args:
               seq_id: integer representing unique id assigned to the sequence
                
        """
        seqs_info = self.seqs_info
        seqs_representer = self.seqs_representer
        activated_states = seqs_representer.get_seq_activatedstates(seq_id, seqs_info)
        seqs_info[seq_id]["activated_states"] = activated_states
        #print("... loading activated states ...")
    
    def load_segfeatures(self, seq_id):
        """load sequence observation features in :attr:`seqs_info`
        
           Args:
               seq_id: integer representing unique id assigned to the sequence
                
        """        
        seqs_info = self.seqs_info
        seqs_representer = self.seqs_representer
        seg_features = seqs_representer.get_seq_segfeatures(seq_id, seqs_info)
        self.seqs_info[seq_id]["seg_features"] = seg_features
        #print("... loading segment features ...")
        
    def load_activefeatures(self, seq_id):
        """load sequence model identified active features in :attr:`seqs_info`

            Args:
                seq_id: integer representing unique id assigned to the sequence
                
        """        
        seqs_representer = self.seqs_representer
        activefeatures = seqs_representer.get_seq_activefeatures(seq_id, self.seqs_info)
        if(not activefeatures):
            # check if activated_states and seg_features are loaded
            l = {}
            #l['activated_states'] = (seq_id, )
            l['seg_features'] = (seq_id, )
            self.check_cached_info(seq_id, l)
            activefeatures = self.generate_activefeatures(seq_id)
            seq_dir = self.seqs_info[seq_id]['activefeatures_dir']
            ReaderWriter.dump_data(activefeatures, os.path.join(seq_dir, 'activefeatures'))
        self.seqs_info[seq_id]["activefeatures"] = activefeatures 
           
    def load_globalfeatures(self, seq_id, per_boundary=True):
        """load sequence global features in :attr:`seqs_info`
        
           Args:
               seq_id: integer representing unique id assigned to the sequence
                
           Keyword Args:
               per_boundary: boolean representing if the required global features dictionary 
                             is represented by boundary (i.e. True) or aggregated (i.e. False)
                
        """ 
        seqs_representer = self.seqs_representer
        gfeatures, exception_fired = seqs_representer.get_seq_globalfeatures(seq_id, self.seqs_info, per_boundary=per_boundary)
#         print("per_boundary ", per_boundary)
#         print(gfeatures_perboundary)
        if(per_boundary):
            fname = "globalfeatures_per_boundary"
        else:
            fname = "globalfeatures"
            if(exception_fired):
                gfeatures = self.model.represent_globalfeatures(gfeatures)
                seq_dir = self.seqs_info[seq_id]['globalfeatures_dir']
                ReaderWriter.dump_data(gfeatures, os.path.join(seq_dir, 'globalfeatures_repr'))
        self.seqs_info[seq_id][fname] = gfeatures
#         print(self.seqs_info[seq_id][fname])
        #print("loading globalfeatures")
        
    def load_imposter_globalfeatures(self, seq_id, y_imposter, seg_other_symbol):
        """load imposter sequence global features in :attr:`seqs_info`

           Args:
               seq_id: integer representing unique id assigned to the sequence
               y_imposter: the imposter sequence generated using viterbi decoder
               seg_other_sybmol: If it is specified, then the task is a segmentation problem 
                                  (in this case we need to specify the non-entity/other element)
                                  else if it is None (default), then it is considered as sequence labeling problem

        """ 
        seqs_representer = self.seqs_representer
        imposter_gfeatures_perboundary, y_imposter_boundaries = seqs_representer.get_imposterseq_globalfeatures(seq_id, self.seqs_info, y_imposter, seg_other_symbol)
        return(imposter_gfeatures_perboundary, y_imposter_boundaries)
    
    def represent_globalfeature(self, gfeatures, boundaries):
        """represent extracted sequence global features 
        
           two representation could be applied:
               - (1) features identified by boundary (i.e. f(X,Y))
               - (2) features identified and aggregated across all positions in the sequence (i.e. F(X, Y))
                

           Args:
               gfeatures: dictionary representing the extracted sequence features (i.e F(X, Y))
               boundaries: if specified (i.e. list of boundaries), then the required representation
                           is global features per boundary (i.e. option (1))
                           else (i.e. None or empty list), then the required representation is the
                           aggregated global features (option(2))
        """ 
        seqs_representer = self.seqs_representer
        windx_fval = seqs_representer.represent_gfeatures(gfeatures, self.model, boundaries=boundaries)        
        return(windx_fval)  
    
    
    def _load_seq(self, seq_id, target = "seq"):
        """load/return components of the sequence which is an instance of :class:`SequenceStruct`
        
           Args:
               seq_id: integer representing unique id assigned to the sequence
                
           Keyword Args:
               target: string from {'seq', 'Y', 'X'}
                
        """ 
        seqs_representer = self.seqs_representer
        seq = seqs_representer.load_seq(seq_id, self.seqs_info)
        if(target == "seq"):
            return(seq)
        elif(target == "Y"):
            return(seq.Y)
        elif(target == "X"):
            return(seq.X)
        
    def check_cached_info(self, seq_id, entity_names):
        """check and load required data elements/entities for every computation step

           Args:
               seq_id: integer representing unique id assigned to the sequence
               entity_name: list of names of the data elements need to be loaded in :attr:`seqs.info` dictionary
                            needed while performing computation
                             
           .. note::
                
              order of elements in the entity_names list is **important**
                
        """ 
        seq_info = self.seqs_info[seq_id]
        func_dict = self.func_dict
        none_type = type(None) 
        for varname, args in entity_names.items():
            if(type(seq_info.get(varname)) == none_type):
                func_dict[varname](*args)

    def clear_cached_info(self, seqs_id, cached_entities = []):
        """clear/clean loaded data elements/entities in :attr:`seqs.info` dictionary

           Args:
               seqs_id: list of integers representing the unique ids of the training sequences

           Keyword Args:
               cached_entities: list of data entities to be cleared for the :attr:`seqs.info` dictionary
                             
           .. note::
           
              order of elements in the entity_names list is **important**
                
        """ 
        args = self.def_cached_entities + cached_entities
        for seq_id in seqs_id:
            seq_info = self.seqs_info[seq_id]
            for varname in args:
                if(varname in seq_info):
                    seq_info[varname] = None
                    
    
    def save_model(self, folder_dir):
        """save model data structures
           
           Args:
               folder_dir: string representing directory where files are pickled/dumped
        """
        # to clean things before pickling the model
        #print(self.seqs_info)
        self.seqs_info.clear() 
        self.seqs_representer.save(folder_dir)
        self.model.save(folder_dir)
        # save weights
        ReaderWriter.dump_data(self.weights, os.path.join(folder_dir, "weights"))
        # write classes used into a file
        class_desc = []
        class_desc.append(str(self.model.__class__).split(".")[-1].split("'")[0])
        class_desc.append(str(self.__class__).split(".")[-1].split("'")[0])
        class_desc.append(str(self.seqs_representer.__class__).split(".")[-1].split("'")[0])
        class_desc.append(str(self.seqs_representer.feature_extractor.__class__).split(".")[-1].split("'")[0])
        class_desc.append(str(self.seqs_representer.attr_extractor.__class__).split(".")[-1].split("'")[0])
        if(self.seqs_representer.attr_scaler):
            class_desc.append(str(self.seqs_representer.attr_scaler.__class__).split(".")[-1].split("'")[0])
        else:
            class_desc.append('None')
        with open(os.path.join(folder_dir, 'class_desc.txt'), 'a') as f:
            f.write("\n".join(class_desc))
            
        
        #print('seqs_info from LCRF ',  self.seqs_info)

    def decode_seqs(self, decoding_method, out_dir, **kwargs):
        r"""decode sequences (i.e. infer labels of sequence of observations)
           
            Args:
                decoding_method: a string referring to type of decoding {viterbi, per_state_decoding}
                out_dir: string representing the working directory (path) where sequence processing will take place
               
            Keyword Arguments:
                file_name: the name of the file in case decoded sequences are required to be written
                sep: separator (default '\t') between the columns when writing decoded sequences to file
                procseqs_foldername: string representing the folder name where intermediary data and parsing would take place
                beam_size: integer determining the size of the beam while decoding
                seqs: a list comprising of sequences that are instances of :class:`SequenceStruct` class to be decoded
                     (used for decoding test data or any new/unseen data -- sequences)
                seqs_info: dictionary containing the info about the sequences to decode 
                          (used for decoding training sequences)
                seqs_dict: a dictionary comprising of sequence ids as keys and corresponding sequences that are instances of :class:`SequenceStruct` class to be decoded
                           as values
            .. note:: 
            
               for keyword arguments only one of {``seqs`` , ``seqs_info``, ``seqs_dict``} option need to be specified
        
        """
        
        w = self.weights

        if(decoding_method == "perstate_decoding"):
            decoder = self.perstate_posterior_decoding
        else:
            decoder = self.viterbi
            
        file_name = kwargs.get('file_name')
        if(file_name):
            # file to write the sequences with their predicted labels
            corpus_fname = "decoding_seqs"
            out_file = os.path.join(create_directory(corpus_fname, out_dir), file_name)
            if(kwargs.get("sep")):
                sep = kwargs['sep']
            else:
                # default separator is tab
                sep = "\t"

        beam_size = kwargs.get('beam_size')
        if(not beam_size):
            beam_size = self.beam_size
            
        unique_id = False
        procseqs_foldername = kwargs.get('procseqs_foldername')
        if(not procseqs_foldername):
            unique_id = True
            procseqs_foldername = "processed_seqs"

        if(kwargs.get("seqs_info")):
            self.seqs_info = kwargs["seqs_info"]
            N = len(self.seqs_info)
        else:
            if(kwargs.get("seqs")): 
                seqs = kwargs["seqs"]           
                seqs_dict = {i+1:seqs[i] for i in range(len(seqs))}
            elif(kwargs.get("seqs_dict")):
                seqs_dict = kwargs['seqs_dict'] 
            else:
                raise('You need to specify one of the following keyword arguments {``seqs`` , ``seqs_info``, ``seqs_dict``} ')
            seqs_id = list(seqs_dict.keys())
            N = len(seqs_id)
            seqs_info = self.seqs_representer.prepare_seqs(seqs_dict, procseqs_foldername, out_dir, unique_id = unique_id)
            self.seqs_representer.scale_attributes(seqs_id, seqs_info)
            self.seqs_representer.extract_seqs_modelactivefeatures(seqs_id, seqs_info, self.model, "processed_seqs", learning=False)
            self.seqs_info = seqs_info

        seqs_pred = {}
        seqs_info = self.seqs_info
        counter = 0
        for seq_id in seqs_info:
            Y_pred, __ = decoder(w, seq_id, beam_size)
            seq = ReaderWriter.read_data(os.path.join(seqs_info[seq_id]["globalfeatures_dir"], "sequence"))
            if(file_name):
                self.write_decoded_seqs([seq], [Y_pred], out_file, sep)
            seqs_pred[seq_id] = {'seq': seq,'Y_pred': Y_pred}
            # clear added info per sequence
            self.clear_cached_info([seq_id])
            counter += 1
            print("sequence decoded -- {} sequences are left".format(N-counter))
        
        # clear seqs_info
        self.seqs_info.clear()
        return(seqs_pred)

            
    def write_decoded_seqs(self, ref_seqs, Y_pred_seqs, out_file, sep = "\t"):
        """write inferred sequences on file
        
           Args:
               ref_seqs: list of sequences that are instances of :class:`SequenceStruct`
               Y_pred_seqs: list of list of tags decoded for every reference sequence
               out_file: string representing out file where data is written
               sep: separator used while writing on out file
        """
        for i in range(len(ref_seqs)):
            Y_pred_seq = Y_pred_seqs[i]
            ref_seq = ref_seqs[i]
            T = ref_seq.T
            line = ""
            for t in range(1, T+1):
                for field_name in ref_seq.X[t]:
                    line += ref_seq.X[t][field_name] + sep
                if(ref_seq.flat_y):
                    line += ref_seq.flat_y[t-1] + sep
                line += Y_pred_seq[t-1]
                line += "\n" 
            line += "\n"
            ReaderWriter.log_progress(line,out_file)  
            
    def prune_states(self, j, delta, beam_size):
        """prune states that fall off the specified beam size
        
           Args:
               j: current position (integer) in the sequence
               delta: score matrix 
               beam_size: specified size of the beam (integer)
    
           .. warning::
           
              implementation of this method is in the child class
        """
        pass
    

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
               
           .. warning::
           
              implementation of this method is in the child class
                          
        """
        pass

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
        Z_beta = numpy.min(beta[1, :])
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
        
    def check_gradient(self, w, seq_id):
        """implementation of finite difference method similar to ``scipy.optimize.check_grad()``
        
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
        
        """
        print("checking gradient...")
        self.clear_cached_info([seq_id])
        epsilon = 1e-4
        w_dim = len(w)
        w = numpy.random.randn(w_dim)
        # basis vector
        ei = numpy.zeros(w_dim, dtype="longdouble")
        grad = numpy.zeros(w_dim, dtype="longdouble")
        for i in range(len(w)):
            ei[i] = epsilon
            l_wplus = self.compute_seq_loglikelihood(w + ei, seq_id)
            self.clear_cached_info([seq_id])
            l_wminus = self.compute_seq_loglikelihood(w - ei, seq_id)
            self.clear_cached_info([seq_id])
            grad[i] = (l_wplus - l_wminus) / (2*epsilon)
            ei[i] = 0
        estimated_grad = self.compute_seqs_gradient(w, [seq_id])
        diff = numpy.abs(-grad + estimated_grad)
        avg_diff = numpy.mean(diff)
        print("difference between both gradients: \n {}".format(diff))
        print("average difference = {}".format(avg_diff))
        # clear seq_id info
        self.clear_cached_info([seq_id])
        return(avg_diff)
    
    def validate_gradient(self, w, seq_id):
        print("checking gradient using approach mentioned in (Bottou, 2012) 'Stochastic Gradient Descent Tricks' paper...")
        self.clear_cached_info([seq_id])
        epsilons = [1e-6, 1e-8, 1e-10]
        rounds = 5
        res = {}
        # generate a random initial weight w
        for __ in range(rounds):
            for epsilon in epsilons: 
                w0 = numpy.random.rand(len(w))
                l0 = self.compute_seq_loglikelihood(w0, seq_id)
                self.clear_cached_info([seq_id])
                g = self.compute_seqs_gradient(w0, [seq_id])
                self.clear_cached_info([seq_id])
                delta = -epsilon*g
                w_prime = w0 + delta
                l_prime = self.compute_seq_loglikelihood(w_prime, seq_id)
                # clear seq_id info
                self.clear_cached_info([seq_id])
                # verify that l_prime = l0 + epsilon*g
                diff = numpy.abs(l0 + numpy.dot(delta, g) - l_prime)
                if(epsilon in res):
                    res[epsilon].append(diff)
                else:
                    res[epsilon] = [diff]
        diff_concat = []
        for eps, diff_array in res.items():
            print("epsilon = ", eps)
            print("difference across 5 random initializations of w ", diff_array)
            diff_concat += diff_array
        avg_diff = numpy.mean(numpy.asarray(diff_concat))
        print("Average gradient difference across all epsilons and initializations is ", avg_diff)
        return(avg_diff)
    
    def validate_expected_featuresum(self, w, seqs_id):
        """validate expected feature computation
        
           Args:
               w: weight vector (numpy vector)
               seqs_id: list of integers representing unique id assigned to the sequences
        """
        self.clear_cached_info(seqs_id)
        grad = self.compute_seqs_gradient(w, seqs_id)
        abs_grad = numpy.abs(grad)
        avg_diff = numpy.mean(abs_grad)
        print("difference between empirical feature sum and model's expected feature sum: \n {}".format(avg_diff))
        print("average difference is {}".format(avg_diff))
        self.clear_cached_info(seqs_id)
        return(avg_diff)
    
if __name__ == "__main__":
    pass