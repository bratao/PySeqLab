'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''    
import os
from pyseqlab.features_extraction import FOFeatureExtractor, HOFeatureExtractor
from pyseqlab.ho_crf_ad import HOCRFAD, HOCRFADModelRepresentation
from pyseqlab.fo_crf import FirstOrderCRF, FirstOrderCRFModelRepresentation
from pyseqlab.ho_crf import HOCRF, HOCRFModelRepresentation
from pyseqlab.hosemi_crf import HOSemiCRF, HOSemiCRFModelRepresentation
from pyseqlab.hosemi_crf_ad import HOSemiCRFAD, HOSemiCRFADModelRepresentation
from pyseqlab.workflow import GenericTrainingWorkflow
from pyseqlab.utilities import ReaderWriter, SequenceStruct, TemplateGenerator, \
                               create_directory, generate_trained_model
from pyseqlab.attributes_extraction import GenericAttributeExtractor
import numpy as np

# define frequently used directories
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

class SeqGenerator(object):
    def __init__(self, seq_len, num_labels, num_features, scaling_type, perc_categorical):
        self.seq_len = seq_len
        self.num_labels = num_labels
        self.num_features = num_features
        self.scaling_type = scaling_type
        self.perc_categorical = perc_categorical
        
    def generate_data(self):
        num_labels = self.num_labels
        seq_len = self.seq_len
        num_features = self.num_features
        percent_categorical = self.perc_categorical
        len_perlabel = int(np.ceil(seq_len/num_labels))
        num_cont_features = num_features
        if(percent_categorical):
            num_cont_features = int(np.ceil(num_features*(100-percent_categorical)/100))
        step_size = 50
        labels = []
        for j in range(num_labels):
            labels += [j]*len_perlabel
        # generate continuous features
        features = []
        for __ in range(num_cont_features):
            step_min = np.random.randint(1, 100, 1)
            step_max = step_min + step_size
            feature = np.array([])
            for j in range(num_labels):
                feature = np.concatenate([feature,
                                          np.random.randint(step_min,step_max, len_perlabel) + np.random.randn(len_perlabel)])
                step_min = step_max + step_size + np.random.randint(1, step_size, 1)
                step_max = step_min + step_size
            features.append(feature.tolist())
        # generate categorical features (if applicable)
        for __ in range(num_features-num_cont_features):
            step_min = np.random.randint(1, 100, 1)
            step_max = step_min + 10
            feature = np.array([], dtype='int32')
            for j in range(num_labels):
                feature = np.concatenate([feature, np.random.randint(step_min,step_max, len_perlabel)])
                step_min = step_max + step_size + np.random.randint(1, step_size, 1)
                step_max = step_min + 10
            features.append(feature.tolist())
        return(features, labels)

    def prepare_data(self, features, labels):
        num_features = len(features)
        X = []
        flag = False
    #     print("features ", features)
        for i in range(num_features):
            feat_name = 'f_{}'.format(i)
    #         print("feat_name ", feat_name)
            for j, elem in enumerate(features[i]):
    #             print("j ", j)
    #             print("elem ", elem)
                if(flag):
                    X[j][feat_name] = str(elem)
                else:
                    X.append({feat_name:str(elem)})
            flag = True
        labels = [str(elem) for elem in labels]
        return(X, labels)

    def generate_seqs(self, num_seqs):
        seqs = []
        for __ in range(num_seqs):
            features, labels= self.generate_data()
            X, Y = self.prepare_data(features, labels)
            seq = SequenceStruct(X, Y)
            seqs.append(seq)
        return(seqs)

# create an instance of SeqGenerator
# this would be a global object to manipulate if different generating options is required
seq_generator = SeqGenerator(100, 3, 100, "rescaling", 0)

class AttributeExtractor(GenericAttributeExtractor):
    """class implementing observation functions that generates attributes from observations"""

    def __init__(self):
        self.num_features = seq_generator.num_features
        self.num_cont_features = int(np.ceil(self.num_features*(100-seq_generator.perc_categorical)/100))
        attr_desc = self.generate_attributes_desc()
        super().__init__(attr_desc)
    
    def generate_attributes_desc(self):
        attr_desc = {}
        for i in range(self.num_cont_features):
            track_attr_name = "f_{}".format(i)
            attr_desc[track_attr_name] = {'description': '{} track'.format(track_attr_name),
                                          'encoding':'continuous'}
        for j in range(self.num_cont_features, self.num_features):
            track_attr_name = "f_{}".format(j)
            attr_desc[track_attr_name] = {'description': '{} track'.format(track_attr_name),
                                          'encoding':'categorical'}  
        return(attr_desc)

def template_config_1():
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for tracks
    track_attr_names = ["f_{}".format(i) for i in range(seq_generator.num_features)]
    for track_attr_name in track_attr_names:
        template_generator.generate_template_XY(track_attr_name, ('1-gram:2-gram', range(-3,4)), '1-state:2-states', templateXY)
    templateY = {'Y':()}
    return(templateXY, templateY)

def template_config_2():
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for tracks
    track_attr_names = ["f_{}".format(i) for i in range(seq_generator.num_features)]
    for track_attr_name in track_attr_names:
        template_generator.generate_template_XY(track_attr_name, ('1-gram:2-gram', range(-3,4)), '1-state:2-states', templateXY)
    templateY = template_generator.generate_template_Y('1-state:2-states:3-states')
    return(templateXY, templateY)

def template_config_3():
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for tracks
    track_attr_names = ["f_{}".format(i) for i in range(seq_generator.num_features)]
    for track_attr_name in track_attr_names:
        template_generator.generate_template_XY(track_attr_name, ('1-gram:2-gram', range(-5,6)), '1-state:2-states', templateXY)
    templateY = template_generator.generate_template_Y('1-state:2-states:3-states')
    return(templateXY, templateY)

def template_config_4():
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for tracks
    track_attr_names = ["f_{}".format(i) for i in range(seq_generator.num_features)]
    for track_attr_name in track_attr_names:
        template_generator.generate_template_XY(track_attr_name, ('1-gram', range(0,1)), '1-state', templateXY)
    templateY = {'Y':()}
    return(templateXY, templateY)

def revive_learnedmodel(model_dir):
    modelpart_dir = os.path.join(model_dir, 'model_parts')
    lmodel = generate_trained_model(modelpart_dir, AttributeExtractor)
    return(lmodel)
            
def build_model(model_type, template_config, num_seqs):
    if(model_type == 'HOCRFAD'):
        modelrepr_class = HOCRFADModelRepresentation
        model_class = HOCRFAD
        fextractor_class = HOFeatureExtractor
    elif(model_type == 'HOCRF'):
        modelrepr_class = HOCRFModelRepresentation
        model_class = HOCRF
        fextractor_class = HOFeatureExtractor
    elif(model_type == 'HOSemiCRFAD'):
        modelrepr_class = HOSemiCRFADModelRepresentation
        model_class = HOSemiCRFAD
        fextractor_class = HOFeatureExtractor
    elif(model_type == 'HOSemiCRF'):
        modelrepr_class = HOSemiCRFModelRepresentation
        model_class = HOSemiCRF
        fextractor_class = HOFeatureExtractor
    elif(model_type == 'FirstOrderCRF'):
        modelrepr_class = FirstOrderCRFModelRepresentation
        model_class = FirstOrderCRF
        fextractor_class = FOFeatureExtractor
    
    # init attribute extractor
    attr_extractor = AttributeExtractor()
    # load templates
    template_XY, template_Y = template_config()    
    # init feature extractor
    fextractor = fextractor_class(template_XY, template_Y, attr_extractor.attr_desc)
    # generate data 
    seqs = seq_generator.generate_seqs(num_seqs)
    # use all passed data as training data -- no splitting
    data_split_options = {'method':'none'}
    # no feature filter 
    fe_filter = None
    working_dir = create_directory('wd', current_dir)
    workflow = GenericTrainingWorkflow(attr_extractor, fextractor, fe_filter, 
                                       modelrepr_class, model_class,
                                       working_dir)
    
    
    # since we are going to train using perceptron based methods
    full_parsing = True

    data_split = workflow.seq_parsing_workflow(data_split_options,
                                               seqs=seqs,
                                               full_parsing = full_parsing)


    # build and return a CRFs model
    # folder name will be f_0 as fold 0
    trainseqs_id = data_split[0]['train']
    crf_m = workflow.build_crf_model(trainseqs_id,
                                     "f_0", 
                                     full_parsing=full_parsing)
    
    
    return(workflow, crf_m, data_split)
    
if __name__ == "__main__":
    pass