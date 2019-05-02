'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''
import os
from collections import defaultdict
from pyseqlab.utilities import SequenceStruct, ReaderWriter


class AttributeScaler(object):
    """attribute scalar class to scale/standardize continuous attributes/features
    
       Args:
           scaling_info: dictionary comprising the relevant info for performing standardization
           method: string defining the method of scaling {rescaling, standardization}
           
       Attributes:
           scaling_info: dictionary comprising the relevant info for performing standardization
           method: string defining the method of scaling {rescaling, standardization}    
           
       Example::
       
           in case of *standardization*:
               - scaling_info has the form: scaling_info[attr_name] = {'mean':value,'sd':value}
           in case of *rescaling*
               - scaling_info has the form: scaling_info[attr_name] = {'max':value, 'min':value}
           
                        
    """
    def __init__(self, scaling_info, method):
        self.scaling_info = scaling_info
        self.method = method
        
    def scale_continuous_attributes(self, seq, boundaries):
        """scale continuous attributes of a sequence for a list of boundaries
        
           Args:
               seq: a sequence instance of :class:`SequenceStruct`
               boundaries: list of boundaries ``[(1,1), (2,2),...,]``
               
        """
        scaling_info = self.scaling_info
        method = self.method
        seg_attr = seq.seg_attr
        try:
            if(method == "standardization"):
                for attr_name in scaling_info:
                    attr_mean = scaling_info[attr_name]['mean']
                    attr_sd = scaling_info[attr_name]['sd']
                    for boundary in boundaries:
                        seg_attr[boundary][attr_name]= (seg_attr[boundary][attr_name] - attr_mean)/(attr_sd)
            elif(method == "rescaling"):
                for attr_name in scaling_info:
                    attr_max = scaling_info[attr_name]['max']
                    attr_min = scaling_info[attr_name]['min']
                    diff = attr_max - attr_min
                    if(diff == 0):        
                        for boundary in boundaries:
                            seg_attr[boundary][attr_name]= 0
                    else:
                        for boundary in boundaries:
                            seg_attr[boundary][attr_name]= self.transform_scale(seg_attr[boundary][attr_name], attr_min, attr_max)

#                         seg_attr[boundary][attr_name]= (seg_attr[boundary][attr_name] - attr_min)/(diff)
        except Exception as e:
            print("one of the features is either constant or zero. Division by zero error...")
            print(e)
            
    def transform_scale(self, x, xref_min, xref_max):
        """transforms feature value to scale from [-1,1]"""
        x_new = 2*(x-xref_min)/(xref_max-xref_min) - 1
        return(x_new)
    
    def save(self, folder_dir):
        """save relevant info about the scaler on disk
        
           Args:
               folder_dir: string representing directory where files are pickled/dumped
        """
        save_info = {'AS_scalinginfo': self.scaling_info,
                     'AS_method':self.method
                    }
        for name in save_info:
            ReaderWriter.dump_data(save_info[name], os.path.join(folder_dir, name))   

class GenericAttributeExtractor(object):
    """Generic attribute extractor class implementing observation functions that generates attributes from tokens/observations
       
       Args:
           attr_desc: dictionary defining the atomic observation/attribute names including
                      the encoding of such attribute (i.e. {continuous, categorical}}    
       Attributes:
           attr_desc: dictionary defining the atomic observation/attribute names including
                      the encoding of such attribute (i.e. {continuous, categorical}}
           seg_attr:  dictionary comprising the extracted attributes per each boundary of a sequence

    """
    def __init__(self, attr_desc):
        self.attr_desc = attr_desc
        self.determine_attr_encoding(attr_desc)
        self.seg_attr = {}
    
    def determine_attr_encoding(self, attr_desc):
        for attr in attr_desc:
            if(attr_desc[attr]['encoding'] == 'categorical'):
                attr_desc[attr]['repr_func'] = self._represent_categorical_attr
            else:
                attr_desc[attr]['repr_func'] = self._represent_continuous_attr
                
    def group_attributes(self):
        """function to group attributes based on the encoding type (i.e. continuous vs. categorical)"""
        attr_desc = self.attr_desc
        grouped_attr = {}
        for attr_name in attr_desc:
            encoding_type = attr_desc[attr_name]['encoding']
            if(encoding_type in grouped_attr):
                grouped_attr[encoding_type].append(attr_name)
            else:
                grouped_attr[encoding_type] = [attr_name]
        return(grouped_attr)
       
    def generate_attributes(self, seq, boundaries):
        X = seq.X  
        observed_attrnames = list(X[1].keys())
        # segment attributes dictionary
        self.seg_attr = {}
        new_boundaries = []
        # create segments from observations using the provided boundaries
        for boundary in boundaries:
            if(boundary not in seq.seg_attr):
                self._create_segment(X, boundary, observed_attrnames)
                new_boundaries.append(boundary)
#         print("seg_attr {}".format(self.seg_attr))
#         print("new_boundaries {}".format(new_boundaries))
        if(self.seg_attr):
            # save generated attributes in seq
            seq.seg_attr.update(self.seg_attr)
#             print('saved attribute {}'.format(seq.seg_attr))
            # clear the instance variable seg_attr
            self.seg_attr = {}
        return(new_boundaries)
        
    def _create_segment(self, X, boundary, attr_names, sep = " "):
        self.seg_attr[boundary] = {}
        attr_desc = self.attr_desc
        for attr_name in attr_names:
            segment_value = self._get_segment_value(X, boundary, attr_name)
            self.seg_attr[boundary][attr_name] = attr_desc[attr_name]['repr_func'](segment_value, sep)
            
    def _get_segment_value(self, X, boundary, target_attr):
        u = boundary[0]
        v = boundary[1]
        segment = []
        for i in range(u, v+1):
            segment.append(X[i][target_attr])
        return(segment)
    
    def _represent_categorical_attr(self, attributes, sep):
        """function to represent categorical attributes
        """
        return(sep.join(attributes))

    def _represent_continuous_attr(self, attributes, sep=None):
        """function to represent continuous attributes
        """
        return(sum(float(attr) for attr in attributes))

class NERSegmentAttributeExtractor(GenericAttributeExtractor):
    """class implementing observation functions that generates attributes from word tokens/observations
       
       Args:
           attr_desc: dictionary defining the atomic observation/attribute names including
                      the encoding of such attribute (i.e. {continuous, categorical}}
    
       Attributes:
           attr_desc: dictionary defining the atomic observation/attribute names including
                      the encoding of such attribute (i.e. {continuous, categorical}}
           seg_attr:  dictionary comprising the extracted attributes per each boundary of a sequence

    """
    def __init__(self):
        attr_desc = self.generate_attributes_desc()
        super().__init__(attr_desc)
    
    def generate_attributes_desc(self):
        """define attributes by including description and encoding of each observation or observation feature  
        """
        attr_desc = {}
        attr_desc['w'] = {'description':'the word/token',
                          'encoding':'categorical'
                         }
        attr_desc['shape'] = {'description':'the shape of the word',
                              'encoding':'categorical'
                             }
        attr_desc['shaped'] = {'description':'the compressed/degenerated form/shape of the word',
                               'encoding':'categorical'
                              }
        attr_desc['seg_numchars'] = {'description':'number of characters in a segment',
                                     'encoding':'continuous'
                                    }
        attr_desc['seg_len'] = {'description':'the length of a segment',
                                'encoding':'continuous'
                               }
        return(attr_desc)
 
    def generate_attributes(self, seq, boundaries):
        """generate attributes of the sequence observations in a specified list of boundaries
        
           Args:
               seq: a sequence instance of :class:`SequenceStruct`
               boundaries: list of boundaries [(1,1), (2,2),...,]
               
           .. note::
           
              the generated attributes are saved first in :attr:`seg_attr` and then passed to 
              the **`seq.seg_attr`**. In other words, at the end :attr:`seg_att` is always cleared
              
        
        """
        X = seq.X
        observed_attrnames = list(X[1].keys() & self.attr_desc.keys())
        # segment attributes dictionary
        self.seg_attr = {}
        new_boundaries = []
        # create segments from observations using the provided boundaries
        for boundary in boundaries:
            if(boundary not in seq.seg_attr):
                self._create_segment(X, boundary, observed_attrnames)
                new_boundaries.append(boundary)
#         print("seg_attr {}".format(self.seg_attr))
#         print("new_boundaries {}".format(new_boundaries))
        if(self.seg_attr):
            attr_names_boa = ('w', 'shaped')
            for boundary in new_boundaries:
                self.get_shape(boundary)
                self.get_degenerateshape(boundary)
                self.get_seg_length(boundary)
                self.get_num_chars(boundary)
                # generate bag of attributes properties in every segment
                self.get_seg_bagofattributes(boundary, attr_names_boa)
            
            # save generated attributes in seq
            seq.seg_attr.update(self.seg_attr)
#             print('saved attribute {}'.format(seq.seg_attr))
            # clear the instance variable seg_attr
            self.seg_attr = {}
        return(new_boundaries)
            
    def get_shape(self, boundary):
        """get shape of segment
        
           Args:
               boundary: tuple (u,v) that marks beginning and end of a segment
        """
        segment = self.seg_attr[boundary]['w']
        res = ''
        for char in segment:
            if char.isupper():
                res += 'A'
            elif char.islower():
                res += 'a'
            elif char.isdigit():
                res += 'D'
            else:
                res += '_'

        self.seg_attr[boundary]['shape'] = res
            
    def get_degenerateshape(self, boundary):
        """get degenerate shape of segment
        
           Args:
               boundary: tuple (u,v) that marks beginning and end of a segment
        """
        segment = self.seg_attr[boundary]['shape']
        track = ''
        for char in segment:
            if not track or track[-1] != char:
                track += char
        self.seg_attr[boundary]['shaped'] = track
        
    def get_seg_length(self, boundary):
        """get the length of a segment
        
           Args:
               boundary: tuple (u,v) that marks beginning and end of a segment
        """
        # begin and end of a boundary
        u = boundary[0]
        v = boundary[-1]
        seg_len = v - u + 1
        self.seg_attr[boundary]['seg_len'] = seg_len
            
    def get_num_chars(self, boundary, filter_out = " "):
        """get the number of characters of a segment
        
           Args:
               boundary: tuple (u,v) that marks beginning and end of a segment
               filter_out: string the default separator between attributes
        """
        segment = self.seg_attr[boundary]['w']
        filtered_segment = segment.split(sep = filter_out)
        num_chars = 0
        for entry in filtered_segment:
            num_chars += len(entry)
        self.seg_attr[boundary]['seg_numchars'] = num_chars
            
    def get_seg_bagofattributes(self, boundary, attr_names, sep = " "):
        """implements the bag-of-attributes concept within a segment 

           Args:
               boundary: tuple (u,v) representing current boundary
               attr_names: list of names of the atomic observations/attributes
               sep: separator (by default is the space)
               
           .. note::
               it can be used **only** with attributes that have binary_encoding type set equal True
           
        """
        prefix = 'bag_of_attr'
        attr_desc = self.attr_desc
        # generate bag of attributes properties in every segment
        for attr_name in attr_names:
            segment = self.seg_attr[boundary][attr_name]
            split_segment = segment.split(sep)
            count_dict = defaultdict(int)
            for elem in split_segment:
                count_dict[elem] += 1
            
            for attr_value, count in count_dict.items():
                fkey = prefix + '_' + attr_name + '_' + attr_value
                self.seg_attr[boundary][fkey] = count
                # adding dynamically the description and the encoding of the new bag of attributes property
                if(fkey not in attr_desc):
                    attr_desc[fkey] = {'description':'{} -- bag of attributes property'.format("fkey"),
                                       'encoding':'continuous'
                                      }
    
if __name__ == "__main__":
    # sequence example is from `Cuong et al. paper <>`_
    X = [{'w':'Peter'}, {'w':'goes'}, {'w':'to'}, {'w':'Britain'}, {'w':'and'}, {'w':'France'}, {'w':'annually'},{'w':'.'}]
    Y = ['P', 'O', 'O', 'L', 'O', 'L', 'O', 'O']
    seq = SequenceStruct(X, Y)
    attr_extractor = NERSegmentAttributeExtractor()
    print("attr_desc {}".format(attr_extractor.attr_desc))
    attr_extractor.generate_attributes(seq, seq.get_y_boundaries())
    for boundary, seg_attr in seq.seg_attr.items():
        print("boundary {}".format(boundary))
        print("attributes {}".format(seg_attr))
    print("seg_attr {}".format(seq.seg_attr))