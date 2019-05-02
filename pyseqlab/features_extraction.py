'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''

import os
from copy import deepcopy
from datetime import datetime
from collections import Counter
import numpy
from .utilities import ReaderWriter, create_directory, generate_datetime_str
from .attributes_extraction import AttributeScaler

class FeatureExtractor(object):
    """Generic feature extractor class that contains feature functions/templates 
    
       Args:
           templateX: dictionary specifying template to follow for observation features extraction.
                      It has the form: ``{attr_name: {x_offset:tuple(y_offsets)}}``.
                      e.g. ``{'w': {(0,):((0,), (-1,0), (-2,-1,0))}}``
                                            
           templateY: dictionary specifying template to follow for y pattern features extraction.
                      It has the form: ``{Y: tuple(y_offsets)}``.  
                      e.g. ``{'Y': ((0,), (-1,0), (-2,-1,0))}``
           attr_desc: dictionary containing description and the encoding of the attributes/observations
                      e.g. attr_desc['w'] = {'description':'the word/token','encoding':'categorical'}
                      for more details/info check the :attr:`attr_desc` of the :class:`NERSegmentAttributeExtractor`
            
       Attributes:
           template_X: dictionary specifying template to follow for observation features extraction.
                       It has the form: ``{attr_name: {x_offset:tuple(y_offsets)}}``
                       e.g. ``{'w': {(0,):((0,), (-1,0), (-2,-1,0))}}``
           template_Y: dictionary specifying template to follow for y pattern features extraction.
                       It has the form: ``{Y: tuple(y_offsets)}``  
                       e.g. ``{'Y': ((0,), (-1,0), (-2,-1,0))}``
           attr_desc: dictionary containing description and the encoding of the attributes/observations.
                      e.g. ``attr_desc['w'] = {'description':'the word/token','encoding':'categorical'}``.
                      For more details/info check the :attr:`attr_desc` of the :class:`NERSegmentAttributeExtractor`
            
    """
    def __init__(self, templateX, templateY, attr_desc):
        self.template_X = templateX
        self.template_Y = templateY
        self.attr_desc = attr_desc
        self.attr_represent_func = self.attr_represent_funcmapper()
    
    def attr_represent_funcmapper(self):
        """assign a representation function based on the encoding (i.e. categorical or continuous) of each attribute name
        """
        attr_represent_func = {}
        attr_desc = self.attr_desc
        for attr_name in attr_desc:
            if(attr_desc[attr_name]["encoding"] == "categorical"):
                attr_represent_func[attr_name] = self._represent_categorical_attr
            else:
                attr_represent_func[attr_name] = self._represent_continuous_attr
        return(attr_represent_func)
    
    @property
    def template_X(self):
        return self._template_X
    @template_X.setter
    def template_X(self, template):
        r"""setup/verify template_X
        
           Args:
               template: dictionary specifying template to follow for observation features extraction
        
           Example:: 
           
               template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
                          = {attr_name: {x_offset:tuple(y_offsets)}}
        """
        if(type(template) == dict):
            self._template_X = {}
            self.y_offsets = set()
            self.x_featurenames = {}
            for attr_name, templateX in template.items():
                self._template_X[attr_name] = {}
                self.x_featurenames[attr_name] = {}
                for offset_x, offsets_y in templateX.items():
                    s_offset_x = tuple(sorted(offset_x))
                    feature_name = '|'.join([attr_name + "[" + str(ofst_x) + "]"  for ofst_x in s_offset_x])
                    self.x_featurenames[attr_name][offset_x] = feature_name
                    unique_dict = {}
                    for offset_y in offsets_y:
                        s_offset_y = tuple(sorted(offset_y))
                        check = self._validate_template(s_offset_y)
                        if(check):
                            unique_dict[s_offset_y] = 1
                            self.y_offsets.add(s_offset_y)
                    if(unique_dict):
                        self._template_X[attr_name][s_offset_x] = tuple(unique_dict.keys())

    @property
    def template_Y(self):
        return self._template_Y
    @template_Y.setter
    def template_Y(self, template):
        r"""setup/verify template_X
        
           Args:
               template: dictionary specifying template to follow for y pattern features extraction
        
           Example:
           
           :: 
           
               template_Y = {'Y': ((0,), (-1,0), (-2,-1,0))}
                          = {Y: tuple(y_offsets)}

        """
        if(type(template) == dict):
            self._template_Y = {}
            unique_dict = {}
            offsets_y = template['Y']
            for offset_y in offsets_y:
                s_offset_y = tuple(sorted(offset_y))
                check = self._validate_template(s_offset_y)
                if(check):
                    unique_dict[s_offset_y] = 1
            if(unique_dict):
                self._template_Y['Y'] = tuple(unique_dict.keys())
            else:
                self._template_Y['Y'] = ()

    def _validate_template(self, template):
        """validate passed template
        
           Args:
               template: a tuple comprising the order of y pattern (i.e. (-2,-1,0))
               
        """
        check = True
        if(len(template) > 1):
            for i in range(len(template)-1):
                curr_elem = template[i]
                next_elem = template[i+1]
                diff = curr_elem - next_elem
                if(diff != -1):
                    check = False
                    break
        else:
            if(template[0] != 0):
                check = False
        return(check)
                    
                
    def extract_seq_features_perboundary(self, seq, seg_features=None):
        """extract features (observation and y pattern features) per boundary
        
           Args:
               seq: a sequence instance of :class:`SequenceStruct`
               
           Keywords Arguments:
               seg_features: optional dictionary of observation features
               
        """
        # this method is used to extract features from sequences with known labels
        # (i.e. we know the Y labels and boundaries)
        Y = seq.Y
        features = {}
        for boundary in Y:
            xy_feat = self.extract_features_XY(seq, boundary, seg_features)
            y_feat = self.extract_features_Y(seq, boundary, self.template_Y)
            y_feat = y_feat['Y']
            #print("boundary {}".format(boundary))
            #print("boundary {}".format(boundary))
            #print("y_feat {}".format(y_feat))
            #print("xy_feat {}".format(xy_feat))
            for offset_tup_y in y_feat:
                for y_patt in y_feat[offset_tup_y]:
                    if(y_patt in xy_feat):
                        xy_feat[y_patt].update(y_feat[offset_tup_y])
                    else:
                        xy_feat[y_patt] = y_feat[offset_tup_y]
            features[boundary] = xy_feat
#             #print("features {}".format(features[boundary]))
#             #print("*"*40)
        return(features)

    
    def aggregate_seq_features(self, features, boundaries):
        """aggregate features across all boundaries
        
           it is usually used to aggregate features in the dictionary obtained from
           :func:`extract_seq_features_perboundary()` function
           
           Args:
               features: dictionary of sequence features per boundary
               boundaries: list of boundaries where detected features are aggregated
               
        """
        # summing up all local features across all boundaries
        seq_features = {}
        for boundary in boundaries:
            xy_features = features[boundary]
            for y_patt in xy_features:
                if(y_patt in seq_features):
                    seq_features[y_patt].update(xy_features[y_patt])
#                     seq_features[y_patt] + xy_features[y_patt]
                else:
                    seq_features[y_patt] = Counter(xy_features[y_patt])
        return(seq_features)
    
#     def extract_seq_features(self, seq):
#         features_per_boundary = self.extract_seq_features_perboundary(seq)
#         seq_features = self.agggregate_features(features_per_boundary, boundaries=seq.Y)
#         return(seq_features)
    
    def extract_features_Y(self, seq, boundary, templateY):
        """extract y pattern features for a given sequence and template Y
        
           Args:
               seq: a sequence instance of :class:`SequenceStruct`
               boundary: tuple (u,v) representing current boundary
               templateY: dictionary specifying template to follow for extraction.
                          It has the form: {Y: tuple(y_offsets)}  
                          e.g. ``{'Y': ((0,), (-1,0), (-2,-1,0))}``
        """
        # to remove y_range and substitute it by checking if pos is within 0 and seq.T
        
        template_Y = templateY['Y']

        if(template_Y):
            Y = seq.Y
            y_sboundaries = seq.y_sboundaries
            y_boundpos_map = seq.y_boundpos_map
            curr_pos = y_boundpos_map[boundary]
            range_y = seq.y_range

            y_patt_features = {}
            feat_template = {}
            for offset_tup_y in template_Y:
                y_pattern = []
                for offset_y in offset_tup_y:
                    # offset_y should be always <= 0
                    pos = curr_pos + offset_y
                    if(pos in range_y):
                        b = y_sboundaries[pos]
                        y_pattern.append(Y[b])
                    else:
                        y_pattern = []
                        break 
                if(y_pattern):
                    feat_template[offset_tup_y] = {"|".join(y_pattern):1}
    
            y_patt_features['Y'] = feat_template
            
        else:
            y_patt_features = {'Y':{}}
        
#         #print("X"*40)
#         #print("boundary {}".format(boundary))
#         for attr_name, f_template in y_patt_features.items():
#             for offset, features in f_template.items():
#                 #print("{} -> {}".format(offset, features))
#         #print("X"*40)
        
        return(y_patt_features)
    
    def extract_features_X(self, seq, boundary):
        """extract observation features for a given sequence at a specified boundary
        
           Args:
               seq: a sequence instance of :class:`SequenceStruct`
               boundary: tuple (u,v) representing current boundary
        
        """
        # get template X
        template_X = self.template_X
        x_featurenames = self.x_featurenames
        # current boundary begin and end
        u, v = boundary

#         #print("positions {}".format(positions))
        seg_features = {}
        for attr_name in template_X:
            attr_represent_func = self.attr_represent_func[attr_name]
#             #print("attr_name {}".format(attr_name))
            # check the type of attribute
            # to use instead a function mapper -- check the init method     
            feat_template = {}
            for offset_tup_x in template_X[attr_name]:
                attributes = []
#                 #print("feature_name {}".format(feature_name))
                for offset_x in offset_tup_x:
#                     #print("offset_x {}".format(offset_x))
                    if(offset_x > 0):
                        pos = (v + offset_x, v + offset_x)
                    elif(offset_x < 0):
                        pos = (u + offset_x, u + offset_x)
                    else:
                        pos = (u, v)
                   
                    if(pos in seq.seg_attr):
                        attributes.append(seq.seg_attr[pos][attr_name])
#                         #print("attributes {}".format(attributes))
                    else:
                        attributes = []
                        break
                if(attributes):
#                     feat_template[offset_tup_x] = represent_attr(attributes, feature_name)
                    feat_template[offset_tup_x] = attr_represent_func(attributes, x_featurenames[attr_name][offset_tup_x])
            seg_features[attr_name] = feat_template
#         
#         #print("X"*40)
#         #print("boundary {}".format(boundary))
#         for attr_name, f_template in seg_features.items():
#             for offset, features in f_template.items():
#                 #print("{} -> {}".format(offset, features))
#         #print("X"*40)

        return(seg_features)

    
    def extract_features_XY(self, seq, boundary, seg_features = None):
        """extract/join observation features with y pattern features as specified :attr:`template_X` 
        
           Args:
               seq: a sequence instance of :class:`SequenceStruct`
               boundary: tuple (u,v) representing current boundary
               
           Keywords Arguments:
               seg_features: optional dictionary of observation features
               
           Example::
           
               template_X = {'w': {(0,):((0,), (-1,0), (-2,-1,0))}}
               Using template_X the function will extract all unigram features of the observation 'w' (0, ) 
               and join it with:
                   - zero-order y pattern features (0,)
                   - first-order y pattern features (-1, 0)
                   - second-order y pattern features (-2, -1, 0)
               template_Y = {'Y': ((0,), (-1,0), (-2,-1,0))}
        """
        if(not seg_features):
            seg_feat_templates = self.extract_features_X(seq, boundary)
        else:
            seg_feat_templates = seg_features[boundary]
        y_feat_template = self.extract_features_Y(seq, boundary, {'Y':self.y_offsets})
#         print(y_feat_template)
#         print(self.y_offsets)
        y_feat_template = y_feat_template['Y']
        templateX = self.template_X

#         #print("seg_feat_templates {}".format(seg_feat_templates))
        xy_features = {}
        for attr_name, seg_feat_template in seg_feat_templates.items():
            for offset_tup_x in seg_feat_template:
                for offset_tup_y in templateX[attr_name][offset_tup_x]:
                    if(offset_tup_y in y_feat_template):
                        for y_patt in y_feat_template[offset_tup_y]:
                            if(y_patt in xy_features):
                                xy_features[y_patt].update(seg_feat_template[offset_tup_x])
                            else:
                                xy_features[y_patt] = dict(seg_feat_template[offset_tup_x])
#                         #print("xy_features {}".format(xy_features))
        return(xy_features)
    
    def lookup_features_X(self, seq, boundary):
        """lookup observation features for a given sequence using varying boundaries (i.e. g(X, u, v))
        
           Args:
               seq: a sequence instance of :class:`SequenceStruct`
               boundary: tuple (u,v) representing current boundary
               
        """
        # get template X
        template_X = self.template_X
        x_featurenames = self.x_featurenames
        # current boundary begin and end
        u = boundary[0]
        v = boundary[-1]
        
#         #print("positions {}".format(positions))
        seg_features = {}
        for attr_name in template_X:
#             #print("attr_name {}".format(attr_name))
            # check the type of attribute
            attr_represent_func = self.attr_represent_func[attr_name]        
            # the offset_tup_x is sorted tuple -- this is helpful in case of out of boundary tuples    
            for offset_tup_x in template_X[attr_name]:
                attributes = []
#                 feature_name = '|'.join(['{}[{}]'.format(attr_name, offset_x) for offset_x in offset_tup_x])
#                 #print("feature_name {}".format(feature_name))
                for offset_x in offset_tup_x:
#                     #print("offset_x {}".format(offset_x))
                    if(offset_x > 0):
                        pos = (v + offset_x, v + offset_x)
                    elif(offset_x < 0):
                        pos = (u + offset_x, u + offset_x)
                    else:
                        pos = (u, v)
#                     #print("pos {}".format(pos))
                    
                    if(pos in seq.seg_attr):
                        attributes.append(seq.seg_attr[pos][attr_name])
#                         #print("attributes {}".format(attributes))
                    else:
                        attributes = []
                        break
                if(attributes):
                    seg_features.update(attr_represent_func(attributes, x_featurenames[attr_name][offset_tup_x]))

#         #print("seg_features lookup {}".format(seg_features))
        return(seg_features)

    def flatten_segfeatures(self, seg_features):
        """flatten observation features dictionary
        
           Args:
               seg_features: dictionary of observation features

        """
        flat_segfeatures = {}
        for attr_name in seg_features:
            for offset in seg_features[attr_name]:
                flat_segfeatures.update(seg_features[attr_name][offset])
        return(flat_segfeatures)
        
    def lookup_seq_modelactivefeatures(self, seq, model, learning=False):
        """lookup/search model active features for a given sequence using varying boundaries (i.e. g(X, u, v))
           
           Args:
               seq: a sequence instance of :class:`SequenceStruct`
               model: a model representation instance of the CRF class (i.e. the class having `ModelRepresentation` suffix)
            
           Keyword Arguments:
               learning: optional boolean indicating if this function is used wile learning model parameters
        """
        # segment length
        L = model.L
        T = seq.T
        seg_features = {}
        l_segfeatures = {}
            
        for j in range(1, T+1):
            for d in range(L):
                if(j-d <= 0):
                    break
                # start boundary
                u = j-d
                # end boundary
                v = j
                boundary = (u, v)
                # used in the case of model training
                if(learning):
                    l_segfeatures[boundary] = self.extract_features_X(seq, boundary)
                    seg_features[boundary] = self.flatten_segfeatures(l_segfeatures[boundary])
                else:
                    seg_features[boundary] = self.lookup_features_X(seq, boundary)

        return(seg_features, l_segfeatures)
    
    
    ########################################################
    # functions used to represent continuous and categorical attributes
    ########################################################

    def _represent_categorical_attr(self, attributes, feature_name):
        """function to represent categorical attributes 
        """
#         #print("attributes ",attributes)
#         #print("featurename ", feature_name)
        feature_val = '|'.join(attributes)
#         feature = '{}={}'.format(feature_name, feature_val)
        feature = feature_name + "=" + feature_val
        return({feature:1})

    def _represent_continuous_attr(self, attributes, feature_name):
        """function to represent continuous attributes
        """
        feature_val = sum(attributes) 
        return({feature_name:feature_val})
    
    def save(self, folder_dir):
        """store the templates used -- templateX and templateY"""
        save_info = {'FE_templateX': self.template_X,
                     'FE_templateY': self.template_Y
                    }
        for name in save_info:
            ReaderWriter.dump_data(save_info[name], os.path.join(folder_dir, name))


class HOFeatureExtractor(FeatureExtractor):
    """Feature extractor class for higher order CRF models """
    def __init__(self, templateX, templateY, attr_desc):
        super().__init__(templateX, templateY, attr_desc)

class FOFeatureExtractor(FeatureExtractor):
    r"""Feature extractor class for first order CRF models 
    
        it supports the addition of start_state and **potentially** stop_state in the future release
        
        Args:
            templateX: dictionary specifying template to follow for observation features extraction.
                       It has the form: ``{attr_name: {x_offset:tuple(y_offsets)}}``
                       e.g. ``{'w': {(0,):((0,), (-1,0), (-2,-1,0))}}``
                                            
            templateY: dictionary specifying template to follow for y pattern features extraction.
                       It has the form: ``{Y: tuple(y_offsets)}``  
                       e.g. ``{'Y': ((0,), (-1,0), (-2,-1,0))}``
                       
            attr_desc: dictionary containing description and the encoding of the attributes/observations
                       e.g. ``attr_desc['w'] = {'description':'the word/token','encoding':'categorical'}``.
                       For more details/info check the :attr:`attr_desc` of the :class:`NERSegmentAttributeExtractor`
            
            start_state: boolean indicating if __START__ state is required in the model
            
        Attributes:
            templateX: dictionary specifying template to follow for observation features extraction.
                       It has the form: ``{attr_name: {x_offset:tuple(y_offsets)}}``
                       e.g. ``{'w': {(0,):((0,), (-1,0), (-2,-1,0))}}``
                                            
            templateY: dictionary specifying template to follow for y pattern features extraction.
                       It has the form: ``{Y: tuple(y_offsets)}``  
                       e.g. ``{'Y': ((0,), (-1,0), (-2,-1,0))}``
                       
            attr_desc: dictionary containing description and the encoding of the attributes/observations
                       e.g. ``attr_desc['w'] = {'description':'the word/token','encoding':'categorical'}``.
                       For more details/info check the :attr:`attr_desc` of the :class:`NERSegmentAttributeExtractor`
            
            start_state: boolean indicating if __START__ state is required in the model
    
        .. note::
       
           The addition of this class is to add support for __START__ and potentially __STOP__ states
          
    """
    def __init__(self, templateX, templateY, attr_desc, start_state = True):
        super().__init__(templateX, templateY, attr_desc)
        self.start_state = start_state
                
    def _validate_template(self, template):
        """validate passed template
        
           Args:
               template: a tuple comprising the order of y pattern (i.e. (-2,-1,0))
        """
        valid_offsets = {(0,), (-1,0)}
        if(template in valid_offsets):
            check = True
        else:
            check = False
        
        return(check)
    
    def extract_features_Y(self, seq, boundary, templateY):
        """extract y pattern features for a given sequence and template Y
        
           Args:
               seq: a sequence instance of :class:`SequenceStruct`
               boundary: tuple (u,v) representing current boundary
               templateY: dictionary specifying template to follow for extraction.
                          It has the form: ``{Y: tuple(y_offsets)}``  
                          e.g. ``{'Y': ((0,), (-1,0)}``
        """
        template_Y = templateY['Y']

        if(template_Y):
            Y = seq.Y
            y_sboundaries = seq.y_sboundaries
            y_boundpos_map = seq.y_boundpos_map
            curr_pos = y_boundpos_map[boundary]
            range_y = seq.y_range
            startstate_flag = self.start_state
            
            y_patt_features = {}
            feat_template = {}
            for offset_tup_y in template_Y:
                y_pattern = []
                for offset_y in offset_tup_y:
                    # offset_y should be always <= 0
                    pos = curr_pos + offset_y
                    if(pos in range_y):
                        b = y_sboundaries[pos]
                        y_pattern.append(Y[b])
                    else:
                        if(startstate_flag and pos == -1):
                            y_pattern.append("__START__")
                        else:
                            y_pattern = []
                            break
                if(y_pattern):
                    feat_template[offset_tup_y] = {"|".join(y_pattern):1}
    
            y_patt_features['Y'] = feat_template
            
        else:
            y_patt_features = {'Y':{}}

        return(y_patt_features)        

class SeqsRepresenter(object):
    """Sequence representer class that prepares, pre-process and transform sequences for learning/decoding tasks
    
       Args:
           attr_extractor: instance of attribute extractor class such as :class:`NERSegmentAttributeExtractor`
                           it is used to apply defined observation functions generating features for the observations
           fextractor: instance of feature extractor class such as :class:`FeatureExtractor`
                       it is used to extract features from the observations and generated observation features using the observation functions
       
       Attributes:
           attr_extractor: instance of attribute extractor class such as :class:`NERSegmentAttributeExtractor`
           fextractor: instance of feature extractor class such as :class:`FeatureExtractor`
           attr_scaler: instance of scaler class :class:`AttributeScaler`
                        it is used for scaling features that are continuous --not categorical (using standardization or rescaling) 
                        
    """
    def __init__(self, attr_extractor, fextractor):
        self.attr_extractor = attr_extractor
        self.feature_extractor = fextractor
        self.attr_scaler = None
        
    @property
    def feature_extractor(self):
        return self._feature_extractor
    @feature_extractor.setter
    def feature_extractor(self, fextractor):
        # make a copy to preserve the template_X and template_Y used in the extractor
        self._feature_extractor = deepcopy(fextractor)
    
    def prepare_seqs(self, seqs_dict, corpus_name, working_dir, unique_id=True, log_progress=True):
        r"""prepare sequences to be used in the CRF models
        
           Main task:
               - generate attributes (i.e. apply observation functions) on the sequence
               - create a directory for every sequence where we save the relevant data
               - create and return seqs_info dictionary comprising info about the prepared sequences
        
           Args:
               seqs_dict: dictionary containing  sequences and corresponding ids where 
                          each sequence is an instance of the :class:`SequenceStruct` class
               corpus_name: string specifying the name of the corpus that will be used as corpus folder name
               working_dir: string representing the directory where the parsing and saving info on disk will occur
               unique_id: boolean indicating if the generated corpus folder will include a generated id
               
           Return:
               seqs_info (dictionary): dictionary comprising the the info about the prepared sequences

           Example::
           
               seqs_info = {'seq_id':{'globalfeatures_dir':directory,
                                      'T': length of sequence
                                      'L': length of the longest segment
                                     }
                                    ....
                            }
        """
        attr_extractor = self.attr_extractor
        
        if(unique_id):
            corpus_folder = "{}_{}".format(corpus_name, generate_datetime_str())
        else:
            corpus_folder = corpus_name
            
        target_dir = create_directory("global_features", create_directory(corpus_folder, working_dir))
        seqs_info = {}
         
        start_time = datetime.now()
        for seq_id, seq in seqs_dict.items():
            # boundaries of X generate segments of length equal 1
            x_boundaries = seq.get_x_boundaries()
            # this will update the seg_attr of the sequence 
            attr_extractor.generate_attributes(seq, x_boundaries)
            # create a folder for every sequence
            seq_dir = create_directory("seq_{}".format(seq_id), target_dir)
            ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "wb")
            seqs_info[seq_id] = {'globalfeatures_dir': seq_dir, 'T':seq.T, 'L':seq.L}
            
        end_time = datetime.now()
        
        # log progress
        if(log_progress):
            log_file = os.path.join(target_dir, "log.txt")
            line = "---Preparing/parsing sequences--- starting time: {} \n".format(start_time)
            line +=  "Number of sequences prepared/parsed: {}\n".format(len(seqs_dict))
            line += "Corpus directory of the parsed sequences is: {} \n".format(target_dir)
            line += "---Preparing/parsing sequences--- end time: {} \n".format(end_time)
            line += "\n \n"
            ReaderWriter.log_progress(line, log_file)
        
        return(seqs_info)

    def preprocess_attributes(self, seqs_id, seqs_info, method = "rescaling"):
        r"""preprocess sequences by generating attributes for segments with L >1 
        
           Main task:
               - generate attributes (i.e. apply observation functions) on segments (i.e. L>1)
               - scale continuous attributes and building the relevant scaling info needed
               - create a directory for every sequence where we save the relevant data
        
           Args:
               seqs_id: list of sequence ids to be processed
               seqs_info: dictionary comprising the the info about the prepared sequences

           Keyword Arguments:
                method: string determining the type of scaling (if applicable)
                        it supports {standardization, rescaling}
        """
        attr_extractor = self.attr_extractor
        grouped_attr = attr_extractor.group_attributes()
        if(grouped_attr.get("continuous")):
            active_attr = list(self.feature_extractor.template_X.keys())
            active_continuous_attr = [attr for attr in active_attr if attr in grouped_attr['continuous']]
        else:
            active_continuous_attr = {}
            
        attr_dict = {}
        
        seq_dir = None
        start_time = datetime.now()
        for seq_id in seqs_id:
            # length of longest entity in a sequence
            seq_L = seqs_info[seq_id]['L']
            if(seq_L > 1 or active_continuous_attr):
                seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
                seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb")
                y_boundaries = seq.y_sboundaries
            # generate attributes for segments 
            if(seq_L>1):
                # this will update the value of the seg_attr of the sequence 
                new_boundaries = attr_extractor.generate_attributes(seq, y_boundaries)
                # this condition might be redundant -- consider to remove and directly dump the sequence
                if(new_boundaries):
                    ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "wb")
            
            # gather stats for rescaling/standardizing continuous variables
            if(active_continuous_attr):
                for attr_name in active_continuous_attr:
                    for y_boundary in y_boundaries:
                        attr_val = seq.seg_attr[y_boundary][attr_name]
                        if(attr_name in attr_dict):
                            attr_dict[attr_name].append(attr_val)
                        else:
                            attr_dict[attr_name] = [attr_val]  
                            
        # generate attribute scaler object
        if(attr_dict):                      
            scaling_info = {}
            if(method == "rescaling"):
                for attr_name in attr_dict:
                    scaling_info[attr_name] = {}
                    scaling_info[attr_name]['max'] = numpy.max(attr_dict[attr_name])
                    scaling_info[attr_name]['min'] = numpy.min(attr_dict[attr_name])
            elif(method == "standardization"):
                for attr_name in attr_dict:
                    scaling_info[attr_name] = {}
                    scaling_info[attr_name]['mean'] = numpy.mean(attr_dict[attr_name])
                    scaling_info[attr_name]['sd'] = numpy.std(attr_dict[attr_name])

            attr_scaler = AttributeScaler(scaling_info, method)
            self.attr_scaler = attr_scaler
            # scale the attributes
            self.scale_attributes(seqs_id, seqs_info)
        end_time = datetime.now()
                    
        # any sequence would lead to the parent directory of prepared/parsed sequences
        # using the last sequence id and corresponding sequence directory
        if(seq_dir):
            target_dir = os.path.dirname(seq_dir)
            log_file = os.path.join(target_dir, "log.txt")
            line = "---Rescaling continuous features--- starting time: {} \n".format(start_time)
            line +=  "Number of instances/training data processed: {}\n".format(len(seqs_id))
            line += "---Rescaling continuous features--- end time: {} \n".format(end_time)
            line += "\n \n"
            ReaderWriter.log_progress(line, log_file)
        
    def scale_attributes(self, seqs_id, seqs_info):
        """scale continuous attributes 
        
           Args:
               seqs_id: list of sequence ids to be processed
               seqs_info: dictionary comprising the the info about the prepared sequences
        """
        attr_scaler = self.attr_scaler
        if(attr_scaler):
            for seq_id in seqs_id:
                seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
                seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb") 
                boundaries = list(seq.seg_attr.keys())
                attr_scaler.scale_continuous_attributes(seq, boundaries)
                ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "wb")

    def extract_seqs_globalfeatures(self, seqs_id, seqs_info, dump_gfeat_perboundary=False):
        r"""extract globalfeatures (i.e. F(X,Y)) from every sequence
        
            Main task:
                - parses each sequence and generates global feature :math:`F_j(X,Y) = \sum_{t=1}^{T}f_j(X,Y)` 
                - for each sequence we obtain a set of generated global feature functions where each
                  :math:`F_j(X,Y)` represents the sum of the value of its corresponding low-level/local feature function
                  :math:`f_j(X,Y)` (i.e. :math:`F_j(X,Y) = \sum_{t=1}^{T+1} f_j(X,Y)`)
                - saves all the results on disk
        
            Args:
                seqs_id: list of sequence ids to be processed
                seqs_info: dictionary comprising the the info about the prepared sequences
               
            .. note::
             
               it requires that the sequences have been already parsed and preprocessed (if applicable)
        """
        feature_extractor = self.feature_extractor
        
        start_time = datetime.now()
        counter = 0 
        for seq_id in seqs_id:
            seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
            seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb")
            # extract the sequence global features per boundary
            gfeatures_perboundary = feature_extractor.extract_seq_features_perboundary(seq)   
            y_boundaries = seq.y_sboundaries
            # gfeatures has this format {'Y_patt':Counter(features)}
            gfeatures = feature_extractor.aggregate_seq_features(gfeatures_perboundary, y_boundaries)                 
            # store the features' sum (i.e. F_j(X,Y) for every sequence on disk)
            ReaderWriter.dump_data(gfeatures, os.path.join(seq_dir, "globalfeatures"))
            # case of perceptron/search based training with pruned beam
            if(dump_gfeat_perboundary):
                ReaderWriter.dump_data(gfeatures_perboundary, os.path.join(seq_dir, "globalfeatures_per_boundary"))
            counter+=1
            print("dumping globalfeatures -- processed seqs: ", counter)

        end_time = datetime.now()
        
        # any sequence would lead to the parent directory of prepared/parsed sequences
        # using the last sequence id and corresponding sequence directory
        target_dir = os.path.dirname(seq_dir)
        log_file = os.path.join(target_dir, "log.txt")
        line = "---Generating Global Features F_j(X,Y)--- starting time: {} \n".format(start_time)
        line +=  "Number of instances/training data processed: {}\n".format(len(seqs_id))
        line += "---Generating Global Features F_j(X,Y)--- end time: {} \n".format(end_time)
        line += "\n \n"
        ReaderWriter.log_progress(line, log_file)
        
    def create_model(self, seqs_id, seqs_info, model_repr_class, filter_obj = None):
        r"""aggregate all identified features in the training sequences to build one model
        
           Main task:
               - use the sequences assigned  in the training set to build the model
               - takes the union of the detected global feature functions :math:`F_j(X,Y)` for each chosen parsed sequence
                 from the training set to form the set of model features
               - construct the tag set Y_set (i.e. possible tags assumed by y_t) using the chosen parsed sequences
                 from the training data set
               - determine the longest segment length (if applicable)
               - apply feature filter (if applicable)
               
           Args:
               seqs_id: list of sequence ids to be processed
               seqs_info: dictionary comprising the the info about the prepared sequences
               model_repr_class: class name of model representation (i.e. class that has suffix
                                 `ModelRepresentation` such as :class:`HOCRFModelRepresentation`)
                                 
           Keyword Arguments:
               filter_obj: optional instance of :class:`FeatureFilter` class to apply filter
             
           .. note::
             
              it requires that the sequences have been already parsed and global features were generated
              using :func:`extract_seqs_globalfeatures`
              
        """
        Y_states = {}
        modelfeatures = {}
        # length of default entity in a segment
        L = 1
        counter = 0
        start_time = datetime.now()
        for seq_id in seqs_id:
            seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
            # get the largest length of an entity in the segment
            seq_L = seqs_info[seq_id]['L']
            if(seq_L > L):
                L = seq_L
                
            gfeatures = ReaderWriter.read_data(os.path.join(seq_dir, "globalfeatures"))
            # generate a global vector for the model    
            for y_patt, featuresum in gfeatures.items():
                if(y_patt in modelfeatures):
                    modelfeatures[y_patt].update(featuresum)
                else:
                    modelfeatures[y_patt] = featuresum
                # record all encountered states/labels
                parts = y_patt.split("|")
                for state in parts:
                    Y_states[state] = 1 
            counter+=1      
            print("constructing model -- processed seqs: ", counter)
        # apply a filter 
        if(filter_obj):
            # this will trim unwanted features from modelfeatures dictionary
            modelfeatures = filter_obj.apply_filter(modelfeatures)
            #^print("modelfeatures ", modelfeatures)
            
        # create model representation
        model = model_repr_class()
        model.setup_model(modelfeatures, Y_states, L)

        end_time = datetime.now()

        # any sequence would lead to the parent directory
        target_dir = os.path.dirname(seq_dir)
        # log progress
        log_file = os.path.join(target_dir, "log.txt")
        line = "---Constructing model--- starting time: {} \n".format(start_time)
        line += "Number of instances/training data processed: {}\n".format(len(seqs_id))
        line += "Number of features: {} \n".format(model.num_features)
        line += "Number of labels: {} \n".format(model.num_states)
        line += "---Constructing model--- end time: {} \n".format(end_time)
        line += "\n \n"
        ReaderWriter.log_progress(line, log_file)
        
        return(model)

    def extract_seqs_modelactivefeatures(self, seqs_id, seqs_info, model, output_foldername, learning=False):
        """identify for every sequence model active states and features
        
           Main task:
               - generate attributes for all segments with length 1 to maximum length defined in the model
                 it is an optional step and only applied in case of having segmentation problems
               - generate segment features, potential activated states and a representation of segment features
                 to be used potentially while learning
               - dump all info on disk
               
           Args:
               seqs_id: list of sequence ids to be processed
               seqs_info: dictionary comprising the the info about the prepared sequences
               model: an instance of model representation class (i.e. class that has suffix
                      `ModelRepresentation` such as :class:`HOCRFModelRepresentation`)
               output_foldername: string representing the name of the root folder to be created 
                                  for containing all saved info
                                 
           Keyword Arguments:
               learning: boolean indicating if this function used for the purpose of learning (model weights optimization)
             
           .. note::
             
              seqs_info dictionary will be updated by including the directory of the saved generatd info
              
        """
        # get the root_dir
        seq_id = seqs_id[0]
        seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
        root_dir = os.path.dirname(os.path.dirname(seq_dir))
        output_dir = create_directory("model_activefeatures_{}".format(output_foldername), root_dir)
        L = model.L
        f_extractor = self.feature_extractor
        counter = 0
        start_time = datetime.now()
        for seq_id in seqs_id:
            counter += 1
            # lookup active features for the current sequence and store them on disk
            print("identifying model active features -- processed seqs: ", counter)
            seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
            seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"))
            if(L > 1):
                self._lookup_seq_attributes(seq, L)
                ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "wb")
            seg_features, l_segfeatures = f_extractor.lookup_seq_modelactivefeatures(seq, model, learning=learning)

            # dump model active features data
            activefeatures_dir = create_directory("seq_{}".format(seq_id), output_dir)
            seqs_info[seq_id]["activefeatures_dir"] = activefeatures_dir
            ReaderWriter.dump_data(seg_features, os.path.join(activefeatures_dir, "seg_features"))
            # to add condition regarding learning
            ReaderWriter.dump_data(l_segfeatures, os.path.join(activefeatures_dir, "l_segfeatures"))

            
        end_time = datetime.now()
        
 
        log_file = os.path.join(output_dir, "log.txt")
        line = "---Finding sequences' model active-features--- starting time: {} \n".format(start_time)
        line += "Total number of parsed sequences: {} \n".format(len(seqs_id))
        line += "---Finding sequences' model active-features--- end time: {} \n".format(end_time)
        line += "\n \n"
        ReaderWriter.log_progress(line, log_file)
    
    def _lookup_seq_attributes(self, seq, L):
        """generate the missing attributes if the segment length is greater than 1
        
           Args:
               seq: a sequence instance of :class:`SequenceStruct`
               L: longest segment defined in the model
               
           .. note::
              
              sequence :attr:`seg_attr` attribute might be update if L > 1  
        """
        # 
        attr_extractor = self.attr_extractor
        attr_scaler = self.attr_scaler
        T = seq.T
        for j in range(1, T+1):
            for d in range(L):
                if(j-d <= 0):
                    break
                boundary = (j-d, j)
                if(boundary not in seq.seg_attr):
                    # this will update the value of the seg_attr of the sequence 
                    attr_extractor.generate_attributes(seq, [boundary])
                    if(attr_scaler):
                        attr_scaler.scale_continuous_attributes(seq, [boundary])
            
    
    def get_seq_activatedstates(self, seq_id, seqs_info):
        """retrieve identified activated states that were saved on disk using `seqs_info`
        
           Args:
               seqs_id: list of sequence ids to be processed
               seqs_info: dictionary comprising the the info about the prepared sequences
    
           .. note::
           
              this data was generated using :func:`extract_seqs_modelactivefeatures`
               
        """
        seq_dir = seqs_info[seq_id]["activefeatures_dir"]
        activated_states = ReaderWriter.read_data(os.path.join(seq_dir,"activated_states"))
        return(activated_states)
    
    def get_seq_segfeatures(self, seq_id, seqs_info):
        """retrieve segment features that were extracted and saved on disk using `seqs_info`
        
           Args:
               seqs_id: list of sequence ids to be processed
               seqs_info: dictionary comprising the the info about the prepared sequences
               
           .. note::
           
              this data was generated using :func:`extract_seqs_modelactivefeatures`
               
        """
        seq_dir = seqs_info[seq_id]["activefeatures_dir"]
        seg_features = ReaderWriter.read_data(os.path.join(seq_dir, "seg_features"))
        return(seg_features)
    
    def get_seq_lsegfeatures(self, seq_id, seqs_info):
        """retrieve segment features that were extracted with a modified representation for the purpose of parameter learning
        
           Args:
               seqs_id: list of sequence ids to be processed
               seqs_info: dictionary comprising the the info about the prepared sequences
               
           .. note::
           
              this data was generated using :func:`extract_seqs_modelactivefeatures`
               
        """
        seq_dir = seqs_info[seq_id]["activefeatures_dir"]
        seg_features = ReaderWriter.read_data(os.path.join(seq_dir, "l_segfeatures"))
        return(seg_features)
    
    def get_seq_activefeatures(self, seq_id, seqs_info):
        """retrieve sequence model active features that are identified
        
           Args:
               seqs_id: list of sequence ids to be processed
               seqs_info: dictionary comprising the the info about the prepared sequences
        """

        seq_dir = seqs_info[seq_id]["activefeatures_dir"]
        try:
            activefeatures = ReaderWriter.read_data(os.path.join(seq_dir, "activefeatures"))
        except FileNotFoundError:
            # consider logging the error
            #print("activefeatures_per_boundary file does not exist yet !!")
            activefeatures = None
        finally:
            return(activefeatures)
        
    def get_seq_globalfeatures(self, seq_id, seqs_info, per_boundary=True):
        r"""retrieves the global features available for a given sequence (i.e. :math:`F(X,Y)` for all :math:`j \in [1...J]` ) 
        
           Args:
               seqs_id: list of sequence ids to be processed
               seqs_info: dictionary comprising the the info about the prepared sequences
           
           Keyword Arguments:
               per_boundary: boolean specifying if the global features representation
                             should be per boundary or aggregated across the whole sequence
        """
        seq_dir = seqs_info[seq_id]['globalfeatures_dir']
        if(per_boundary):
            fname = "globalfeatures_per_boundary"
        else:
            fname = "globalfeatures_repr"
        try:
            exception_fired = False
            gfeatures = ReaderWriter.read_data(os.path.join(seq_dir, fname))
        except FileNotFoundError:
            # read the saved globalfeatures on disk
            gfeatures = ReaderWriter.read_data(os.path.join(seq_dir, "globalfeatures"))
            exception_fired = True
        finally:
            return(gfeatures, exception_fired)
    
    def aggregate_gfeatures(self, gfeatures, boundaries):
        """aggregate global features using specified list of boundaries
        
           Args:
               gfeatures: dictionary representing the extracted sequence features (i.e F(X, Y))
               boundaries: list of boundaries to use for aggregating global features
        """
        feature_extractor = self.feature_extractor
        # gfeatures is assumed to be represented by boundaries
        gfeatures = feature_extractor.aggregate_seq_features(gfeatures, boundaries)
        return(gfeatures)
    
    def represent_gfeatures(self, gfeatures, model, boundaries=None):
        """represent extracted sequence global features 
        
           two representation could be applied:
               - (1) features identified by boundary (i.e. f(X,Y))
               - (2) features identified and aggregated across all positions in the sequence (i.e. F(X, Y))
                

           Args:
               gfeatures: dictionary representing the extracted sequence features (i.e F(X, Y))
               model: an instance of model representation class (i.e. class that has suffix
                      ModelRepresentation such as :class:`HOCRFModelRepresentation`)
                      
           Keyword Args:
               boundaries: if specified (i.e. list of boundaries), then the required representation
                           is global features per boundary (i.e. option (1))
                           else (i.e. None or empty list), then the required representation is the
                           aggregated global features (option(2))
        """ 
        feature_extractor = self.feature_extractor
        # if boundaries is specified, then gfeatures is assumed to be represented by boundaries
        if(boundaries):
            gfeatures = feature_extractor.aggregate_seq_features(gfeatures, boundaries)
        #^print("gfeatures ", gfeatures)
        windx_fval = model.represent_globalfeatures(gfeatures)
        return(windx_fval)
    
    @staticmethod
    def load_seq(seq_id, seqs_info):
        """load dumped sequences on disk
        
           Args:
               seqs_info: dictionary comprising the the info about the prepared sequences
        """
        seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
        seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb")
        return(seq)
            
    def get_imposterseq_globalfeatures(self, seq_id, seqs_info, y_imposter, seg_other_symbol = None):
        """get an imposter decoded sequence 
        
           Main task:
               - to be used for processing a sequence, generating global features and 
                 return back without storing/saving intermediary results on disk
        
           Args:
               seqs_id: list of sequence ids to be processed
               seqs_info: dictionary comprising the the info about the prepared sequences
               y_imposter: list of labels (y tags) decoded using a decoder
            
           Keyword Arguments:
               seg_other_symbol: in case of segmentation, this represents the non-entity symbol 
                                 label used. Otherwise, it is None (default) which translates to 
                                 be a sequence labeling problem.
        

        """
        feature_extractor = self.feature_extractor
        attr_extractor = self.attr_extractor
        attr_scaler = self.attr_scaler
        ##print("seqs_info {}".format(seqs_info))
        seq_dir = seqs_info[seq_id]["globalfeatures_dir"]
        ##print("seq_dir {}".format(seq_dir))
        seq = ReaderWriter.read_data(os.path.join(seq_dir, "sequence"), mode = "rb")
        
        y_ref = list(seq.flat_y)        
        # update seq.Y with the imposter Y
        seq.Y = (y_imposter, seg_other_symbol)
        y_imposter_boundaries = seq.y_sboundaries
        #^print("y_imposter_boundaries ", y_imposter_boundaries)
        # this will update the value of the seg_attr of the sequence 
        new_boundaries = attr_extractor.generate_attributes(seq, y_imposter_boundaries)
        #^print("new_boundaries ", new_boundaries)
        if(new_boundaries):
            attr_scaler.scale_continuous_attributes(seq, new_boundaries)
            
        activefeatures_dir =  seqs_info[seq_id]["activefeatures_dir"]

        l_segfeatures = ReaderWriter.read_data(os.path.join(activefeatures_dir, "l_segfeatures"), mode = "rb")

        imposter_gfeatures = feature_extractor.extract_seq_features_perboundary(seq, l_segfeatures)
        #^print("imposter_gfeatures ", imposter_gfeatures)
        # put back the original Y
        seq.Y = (y_ref, seg_other_symbol) 
        if(new_boundaries):
            # write back the sequence on disk given the segment attributes have been updated
            ReaderWriter.dump_data(seq, os.path.join(seq_dir, "sequence"), mode = "rb")
        
        return(imposter_gfeatures, y_imposter_boundaries)

    def save(self, folder_dir):
        """save essential info about feature extractor
        
           Args:
               folder_dir: string representing directory where files are pickled/dumped
        """
        self.feature_extractor.save(folder_dir)
        if(self.attr_scaler):
            self.attr_scaler.save(folder_dir)
        
        

class FeatureFilter(object):
    r"""class for applying filters by y pattern or feature counts
    
       Args:
           filter_info: dictionary that contains type of filter to be applied 
           
       Attributes:
           filter_info: dictionary that contains type of filter to be applied 
           rel_func: dictionary of function map
       
       Example::
       
           filter_info dictionary has three keys:
               - `filter_type` to define the type of filter either {count or pattern}
               - `filter_val` to define either the y pattern or threshold count
               - `filter_relation` to define how the filter should be applied
               
            
           *count filter*: 
               - ``filter_info = {'filter_type': 'count', 'filter_val':5, 'filter_relation':'<'}``
                 this filter would delete all features that have count less than five
                 
           *pattern filter*:
               - ``filter_info = {'filter_type': 'pattern', 'filter_val': {"O|L", "L|L"}, 'filter_relation':'in'}``
                 this filter would delete all features that have associated y pattern ["O|L", "L|L"]

    """
    
    def __init__(self, filter_info):

        self.filter_info = filter_info
        self.rel_func = {"=":self._equal_rel,
                         "<=":self._lequal_rel,
                         "<":self._less_rel,
                         ">=":self._gequal_rel,
                         ">":self._greater_rel,
                         "in":self._in_rel,
                         "not in":self._notin_rel}
        
    def apply_filter(self, featuresum_dict):
        """apply define filter on model features dictionary
        
           Args:
               featuresum_dict: dictoinary that represents model features
                                similar to `modelfeatures` attribute in one of 
                                model representation instances
                                
        """
        filtered_dict = deepcopy(featuresum_dict)
        filter_info = self.filter_info
        rel_func = self.rel_func
        if(filter_info['filter_type'] == "count"):
            threshold = filter_info['filter_val']
            relation = filter_info['filter_relation']
            # filter binary/categorical features that have counts less than specified threshold
            for z in featuresum_dict:
                for fname, fsum in featuresum_dict[z].items():
                    # apply filtering only to binary/categorical features if the threshold is of type int
                    if(type(threshold) == int and type(fsum) == int):
                        rel_func[relation](fsum, threshold, filtered_dict[z], fname)
                    elif(type(threshold)==float): # threshold is of type float -- apply to both categorical and continuous
                        rel_func[relation](fsum, threshold, filtered_dict[z], fname)

                            
        elif(filter_info['filter_type'] == "pattern"):
            filter_pattern = filter_info['filter_val']
            relation = filter_info['filter_relation']
            # filter based on specific patterns
            for z in featuresum_dict:
                #^print("z ", z)
                rel_func[relation](z, filter_pattern, filtered_dict)
        #^print("filtered_dict ", filtered_dict)
        return(filtered_dict)
    
    @staticmethod
    def _equal_rel(x, y, f, z):
        if(x==y): del f[z]

    @staticmethod
    def _lequal_rel(x, y, f, z):
        if(x<=y): del f[z]
        
    @staticmethod
    def _less_rel(x, y, f, z):
        if(x<y): del f[z]
        
    @staticmethod
    def _gequal_rel(x, y, f, z):
        if(x>=y): del f[z]
        
    @staticmethod
    def _greater_rel(x, y, f, z):
        if(x>y): del f[z]

    @staticmethod
    def _in_rel(x, y, z):
        if(x in y): del z[x]
    @staticmethod
    def _notin_rel(x, y, z):
        if(x not in y): 
            #^print("{} not in {}".format(x, y))
            #^print("deleting ", z[x])
            del z[x]
        
def main():
    pass

if __name__ == "__main__":
    main()
    