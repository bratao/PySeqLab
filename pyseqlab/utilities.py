'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''
import os
import pickle
import shutil
from datetime import datetime
from copy import deepcopy
from itertools import combinations
import heapq
import numpy

class SequenceStruct(object):
    r"""class for representing each sequence/segment
    
       Args:
           Y: list containing the sequence of states/labels (i.e. ['P','O','O','L','L'])
           X: list containing dictionary elements of observation sequences and/or features of the input
           seg_other_symbol: string or None (default), if specified then the task is a segmentation problem 
                             where it represents the non-entity symbol else (None) then it is considered 
                             as sequence labeling problem
        
       Attributes:
           Y: list containing the sequence of states/labels (i.e. ['P','O','O','L','L'])
           X: list containing dictionary elements of observation sequences and/or features of the input
           seg_other_symbol: string or None(default), if specified then the task is a segmentation problem 
                             where it represents the non-entity symbol else (None) then it is considered 
                             as sequence labeling problem
           T: int, length of a sequence (i.e. len(X))
           seg_attr: dictionary comprising the extracted attributes per each boundary of a sequence
           L: int, longest length of an identified segment in the sequence
           flat_y: list of labels/tags 
           y_sboundaries: sorted list of boundaries of the :attr:`Y` of the sequence
           y_range: range of the sequence
    
    """
    def __init__(self, X, Y, seg_other_symbol = None):
        self.seg_attr = {}
        self.X = X
        self.Y = (Y, seg_other_symbol)


    @property
    def X(self):
        return(self._X)
    @X.setter
    def X(self, l):
        """setup the observation sequence 
           
           Args:
               l: a list of elements (i.e. ``X =  [{'w':'Michael'}, {'w':'is'}, {'w':'in'}, {'w':'New'}, {'w':'Haven'}]``)
           
           
           Example::
           
               the output X becomes:
                            {1:{'w':'Michael'},
                             2:{'w':'is'}, 
                             3:{'w':'in'}, 
                             4:{'w':'New'},
                             5:{'w':'Haven'}
                            }
        """
        self._X = {}
        T = len(l)
        for i in range(T):
            self._X[i+1] = l[i]

        # new assignment clear seg_attr
        if(self.seg_attr):
            self.seg_attr.clear()
        self.T = T
        
    @property
    def Y(self):
        return(self._Y)
    @Y.setter
    def Y(self, elmtup):
        """setup the label sequence
        
           Args:
               elmtup: tuple consisting of:
                       - **Y** a list of elements (i.e. ``Y = ['P','O','O','L','L']``) 
                         representing the labels of the elements in X
                       - **non_entity_symbol** which represents the Other category (i.e. non entity element which is 'O' in above example)
           
           Example:
           
               Y after the transformation becomes ``{(1, 1): 'P', (2,2): 'O', (3, 3): 'O', (4, 5): 'L'}``
        """
        try:
            Y_ref, non_entity_symb = elmtup
        except ValueError:
            raise ValueError("tuple containing Y and non-entity symbol must be passed")
        else:
            self._Y = {}
            # length of longest entity in a segment
            L = 1
            if(non_entity_symb):
                label_indices = {}
                for i in range(len(Y_ref)):
                    label = Y_ref[i]
                    if(label in label_indices):
                        label_indices[label].append(i+1)
                    else:
                        label_indices[label] = [i+1]
 
                for label, indices_list in label_indices.items():
                    if(label == non_entity_symb or len(indices_list) == 1):
                        for indx in indices_list:
                            boundary = (indx, indx)
                            self._Y[boundary] = label

                    else:
                        indx_stack = []
                        for indx in indices_list:
                            if(not indx_stack):
                                indx_stack.append(indx)
                            else:
                                diff = indx - indx_stack[-1]
                                if(diff > 1):
                                    boundary = (indx_stack[0], indx_stack[-1])
                                    self._Y[boundary] = label
                                    l = indx_stack[-1] - indx_stack[0] + 1
                                    if(l > L):
                                        L = l
                                    indx_stack = [indx]
                                else:
                                    indx_stack.append(indx)
                        if(indx_stack):
                            boundary = (indx_stack[0], indx_stack[-1])
                            self._Y[boundary] = label
                            l = indx_stack[-1] - indx_stack[0] + 1
                            if(l > L):
                                L = l
                            indx_stack = [indx]
    
            else:
                for i in range(len(Y_ref)):
                    label = Y_ref[i]
                    boundary = (i+1, i+1)
                    self._Y[boundary] = label
                    
            # store the length of longest entity
            self.L = L
            # keep a copy of Y in as flat list (i.e. ['P','O','O','L','L'])
            self.flat_y = Y_ref
            
            # construct a map from the yboundaries to the pos in the list
            y_sboundaries = self.get_y_boundaries()
            self.y_sboundaries = y_sboundaries

            self.y_boundpos_map = {}
            pos = 0 
            for boundary in y_sboundaries:
                self.y_boundpos_map[boundary] = pos
                pos += 1
            self.y_range = set(range(0, pos))
            
#     def update_boundaries(self):
#         self.y_boundaries = self.get_y_boundaries()
#         self.x_boundaries = self.get_x_boundaries()

    def flatten_y(self, Y):
        r"""flatten the :attr:`Y` attribute 
        
           Args:
               Y: dictionary of this form ``{(1, 1): 'P', (2,2): 'O', (3, 3): 'O', (4, 5): 'L'}``
           
           Example:
               
               flattened y becomes ``['P','O','O','L','L']``
        """
        s_boundaries = sorted(Y)
        flat_y = []
        for u, v in s_boundaries:
            for _ in range(u, v+1):
                flat_y.append(Y[(u,v)])
        return(flat_y)
  
    def get_y_boundaries(self):
        """return the sorted boundaries of the labels of the sequence"""
        return(sorted(self.Y.keys()))
    
    def get_x_boundaries(self):
        """return the boundaries of the observation sequence"""
        boundaries = []
        for u in self.X:
            boundaries.append((u, u))
        return(boundaries)
    
    def __str__(self):
        """return string representation of the parsed sequence"""
        out_str = "Y sequence:\n {}\nX sequence:\n {}\n{}".format(self.flat_y, self.X, "-"*40)
        return(out_str)
            
class DataFileParser(object):
    """class to parse a data file comprising the training/testing data
    
       Attributes:
           seqs: list comprising of sequences that are instances of :class:`SequenceStruct` class
           header: list of attribute names read from the file

    """
    def __init__(self):
        self.header = []
        
    def read_file(self, file_path, header, y_ref = True, seg_other_symbol = None, column_sep = " "):
        r"""read and parse a file the contains the sequences following a predefined format
        
            the file should contain label and observation tracks each separated in a column 
           
            .. note::
        
               label column is the **LAST** column in the file (i.e. X_a X_b Y)
                
            Args:
                file_path: string representing the file path to the data file
                header: specifies how the header is reported in the file containing the sequences
                        options include:
                        - 'main' -> one header in the beginning of the file
                        - 'per_sequence' -> a header for every sequence
                        - list of keywords as header (i.e. ['w', 'part_of_speech'])
                           
            Keyword Arguments:
                y_ref: boolean specifying if the reference label column in the data file
                seg_other_sybmol: string or None(default), if specified then the task is a segmentation problem 
                                  where `seg_other_symbol` represents the non-entity symbol. In this case semi-CRF models
                                  are used. Else (i.e. `seg_other_symbol` is not None) then it is considered 
                                  as sequence labeling problem.
                column_sep: string, separator used between the columns in the file

        """
        if(y_ref):
            update_seq = self.update_XY
        else:
            update_seq = self.update_X
            
        with open(file_path) as file_obj:
            counter = 0
            X = []
            Y = []
            for line in file_obj:
                counter += 1
                line = line.rstrip()
#                 print(line)
                if line:
#                     print(line)
                    if(y_ref):
                        *x_arg, y = line.split(column_sep)
                        self._xarg = x_arg
                        self._y = y
                    else:
                        x_arg = line.split(column_sep)
                        self._xarg = x_arg

#                     print(x_arg)
                    # first line of a sequence
                    if(counter == 1):
                        if(header == "main"):
                            if(self.header):
                                update_seq(X, Y)
#                                 X.append(self.parse_line(x_arg))
#                                 Y.append(y)
                            else:
                                self.parse_header(x_arg)
                                
                        elif(header == "per_sequence"):
                            if(not self.header):
                                self.parse_header(x_arg)
                        else:
                            if(self.header):
                                update_seq(X, Y)
#                                 X.append(self.parse_line(x_arg))
#                                 Y.append(y)
                            else:   
                                self.parse_header(header)
                                update_seq(X, Y)
#                                 X.append(self.parse_line(x_arg))
#                                 Y.append(y)
                    else:
                        update_seq(X, Y)
#                         X.append(self.parse_line(x_arg))
#                         Y.append(y)

                else:
                    seq = SequenceStruct(X, Y, seg_other_symbol)
                    # reset counter for filling new sequence
                    counter = 0
                    X = []
                    Y = []
                    self._xarg = None
                    self._y = None
                    yield seq
                    
            if(X and Y):
                seq = SequenceStruct(X, Y, seg_other_symbol)
                # reset counter for filling new sequence
                counter = 0
                X = []
                Y = []
                self._xarg = None
                self._y = None 
                yield seq

    def update_XY(self, X, Y):
        """update sequence observations and corresponding labels"""
        X.append(self.parse_line(self._xarg))
        Y.append(self._y)
    
    def update_X(self, X, Y):
        """update sequence observations"""
        X.append(self.parse_line(self._xarg))
                
    def parse_line(self, x_arg):
        """parse the read line
        
           Args:
               x_arg: tuple of observation columns
        """
        # fill the sequences X and Y with observations and tags respectively
        header = self.header
        x = {}
        for i in range(len(x_arg)):
            x[header[i]] = x_arg[i]
        return(x)

    def parse_header(self, x_arg):
        """parse header
        
           Args:
               x_arg: tuple of attribute/observation names 
        """
        seq_header = [input_src for input_src in x_arg]
        self.header = seq_header

class ReaderWriter(object):
    """class for dumping, reading and logging data"""
    def __init__(self):
        pass
    @staticmethod
    def dump_data(data, file_name, mode = "wb"):
        """dump data by pickling 
        
           Args:
               data: data to be pickled
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            pickle.dump(data, f, protocol = 4) 
    @staticmethod  
    def read_data(file_name, mode = "rb"):
        """read dumped/pickled data
        
           Args:
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            data = pickle.load(f)
        return(data)
    
    @staticmethod
    def log_progress(line, outfile, mode="a"):
        """write data to a file
        
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(outfile, mode) as f:
            f.write(line)



class AStarNode(object):
    """class representing A* node to be used with A* searcher and viterbi for generating k-decoded list
    
       Args:
           cost: float representing the score/unnormalized probability of a sequence up to given position
           position: integer representing the current position in the sequence
           pi_c: prefix or state code of the label
           label: label of the current position in a sequence
           frwdlink: a link to :class:`AStarNode` node
    
       Attributes:
           cost: float representing the score/unnormalized probability of a sequence up to given position
           position: integer representing the current position in the sequence
           pi_c: prefix or state code of the label
           label: label of the current position in a sequence
           frwdlink: a link to :class:`AStarNode` node
           
    """
    def __init__(self, cost, position, pi_c, label, frwdlink):
        self.cost = cost
        self.position = position
        self.pi_c = pi_c
        self.label = label
        self.frwdlink = frwdlink
        
    def print_node(self):
        """print the info about a node"""
        statement = "cost: {}, position: {}, pi_code: {}, label: {}, ".format(self.cost, self.position, self.pi_c, self.label)
        if(self.frwdlink):
            statement += "forward_link: {}".format(self.frwdlink)
        else:
            statement += "forward_link: None"
        print(statement)
        
class AStarAgenda(object):
    """class containing a heap where instances of :class:`AStarNode` class will be pushed 
    
       the push operation will use the score matrix (built using viterbi algorithm)
       representing the unnormalized probability of the sequences ending at every position 
       with the different available prefixes/states
    
       Attributes:
           qagenda: queue where instances of :class:`AStarNode` are pushed
           entry_count: counter that keeps track of the entries and associate each entry(node)
                        with a unique number. It is useful for resolving nodes with equal costs
        
    """
    def __init__(self):
        self.qagenda = []
        self.entry_count = 0

    def push(self, astar_node, cost):
        """push instance of :class:`AStarNode` with its associated cost to the heap
        
           Args:
               astar_node: instance of :class:`AStarNode` class
               cost: float representing the score/unnormalized probability of a sequence up to given position
        """
        heapq.heappush(self.qagenda, (-cost, self.entry_count, astar_node))
        self.entry_count += 1

    def pop(self):
        """pop nodes with highest score from the heap
        """
        astar_node = heapq.heappop(self.qagenda)[-1]
        return(astar_node)
    
class FO_AStarSearcher(object):
    """A* star searcher associated with first-order CRF model such as :class:`FirstOrderCRF`
       
       Args:
           Y_codebook_rev: a reversed version of dictionary comprising the set of states each assigned a unique code
           
       Attributes:
           Y_codebook_rev: a reversed version of dictionary comprising the set of states each assigned a unique code
    """
    def __init__(self, Y_codebook_rev):
        self.Y_codebook_rev = Y_codebook_rev

    def infer_labels(self, top_node, back_track):
        """decode sequence by inferring labels
        
           Args:
               top_node: instance of :class:`AStarNode` class
               back_track: dictionary containing back pointers built using dynamic programming algorithm
        """
        Y_codebook_rev = self.Y_codebook_rev
        # decoding the sequence
        #print("we are decoding")
        #top_node.print_node()
        y_c = top_node.pi_c
        pos = top_node.position
        Y_decoded = []
        Y_decoded.append(y_c)
        t = pos - 1
        while t>0:
            y_c_tplus1 = Y_decoded[-1]
            y_c_t = back_track[t+1, y_c_tplus1]
            Y_decoded.append(y_c_t)
            t -= 1
        Y_decoded.reverse()
        Y_decoded = [Y_codebook_rev[y_code] for y_code in Y_decoded]
        
        while(top_node.frwdlink):
            y = top_node.frwdlink.label
            Y_decoded.append(y)
            top_node = top_node.frwdlink
#         print(Y_decoded)
        return(Y_decoded)
    
    def search(self, alpha, back_track, T, K):
        """A* star searcher uses the score matrix  (built using viterbi algorithm) to decode top-K list of sequences
        
           Args:
               alpha: score matrix build using the viterbi algorithm
               back_track: back_pointers dictionary tracking the best paths to every state
               T: last decoded position of a sequence (in this context, it is the alpha.shape[0])
               K: number of top decoded sequences to be returned
               
           Returns:
               topk_list: top-K list of decoded sequences
           
        
        """
        # push the best astar nodes to the queue (i.e. the states at time T)
        q = AStarAgenda()
        r = set()
        c = 0
        Y_codebook_rev = self.Y_codebook_rev
        # create nodes from the states at time T
        for y_c in Y_codebook_rev:
            cost = alpha[T, y_c]
            pos = T
            frwdlink = None
            label = Y_codebook_rev[y_c]
            node = AStarNode(cost, pos, y_c, label, frwdlink)
#             node.print_node()
            q.push(node, cost)
            
        track = []
        topk_list = []
        try:
            while c < K:
                #print("heap size ", len(q.qagenda))
                top_node = q.pop()
                track.append(top_node)
        
                for i in reversed(range(2, top_node.position+1)):
                    # best previous state at pos = i-1
                    curr_y_c = top_node.pi_c
                    bestprev_y_c = back_track[i, curr_y_c]
                    pos = i - 1
                    for prev_y_c in Y_codebook_rev:
                        # create a new astar node
                        if(prev_y_c != bestprev_y_c):
                            label = Y_codebook_rev[prev_y_c]
                            cost = alpha[pos, prev_y_c]
                            s = AStarNode(cost, pos, prev_y_c, label, top_node)
                            q.push(s, cost)
                    
                    # create the backlink of the previous top_node (i.e. create a node from the best_y_c) 
                    cost = alpha[pos, bestprev_y_c]
                    label = Y_codebook_rev[bestprev_y_c]
                    top_node = AStarNode(cost, pos, y_c, label, top_node)
                    
                # decode and check if it is not saved already in topk list
                y_labels = self.infer_labels(track[-1], back_track)
#                 print(y_labels)
                signature = "".join(y_labels)
                if(signature not in r):
                    r.add(signature)
                    topk_list.append(y_labels)
                    c += 1
                track.pop()
        except (KeyError, IndexError) as e:
            # consider logging the error
            print(e)
    
        finally:
            #print('r ', r)
            #print('topk ', topk_list)
            return(topk_list)

class HO_AStarSearcher(object):
    """A* star searcher associated with higher-order CRF model such as :class:`HOCRFAD`
       
       Args:
          P_codebook_rev: reversed codebook of set of proper prefixes in the `P` set
                          e.g. ``{0:'', 1:'P', 2:'L', 3:'O', 4:'L|O', ...}``
          P_elems: dictionary comprising the composing elements of every prefix in the `P` set
                   e.g. ``{'':('',), 'P':('P',), 'L':('L',), 'O':('O',), 'L|O':('L','O'), ...}``

       Attributes:
          P_codebook_rev: reversed codebook of set of proper prefixes in the `P` set
                          e.g. ``{0:'', 1:'P', 2:'L', 3:'O', 4:'L|O', ...}``
          P_elems: dictionary comprising the composing elements of every prefix in the `P` set
                   e.g. ``{'':('',), 'P':('P',), 'L':('L',), 'O':('O',), 'L|O':('L','O'), ...}``
    """
    def __init__(self, P_codebook_rev, P_elems):
        self.P_codebook_rev = P_codebook_rev
        self.P_elems = P_elems
        
    def get_node_label(self, pi_code):
        """get the the label/state given a prefix code
        
           Args:
               pi_code: prefix code which is an element of :attr:`P_codebook_rev`
        """
        
        pi = self.P_codebook_rev[pi_code]
        y =  self.P_elems[pi][-1]
        return(y)

    def infer_labels(self, top_node, back_track):
        """decode sequence by inferring labels
        
           Args:
               top_node: instance of :class:`AStarNode` class
               back_track: dictionary containing back pointers tracking the best paths to every state
        """
        # decoding the sequence
        #print("we are decoding")
        #top_node.print_node()
        y = top_node.label
        pi_c = top_node.pi_c
        pos = top_node.position
        Y_decoded = []
        Y_decoded.append((pi_c, y))
        #print("t={}, p_T_code={}, p_T={}, y_T ={}".format(T, p_T_code, p_T, y_T))
        t = pos - 1
        while t>0:
            p_tplus1_c = Y_decoded[-1][0]
            p_t_c, y_t = back_track[t+1, p_tplus1_c]
            #print("t={}, (t+1, p_t_code)=({}, {})->({},{})".format(t, t+1, P_codebook[p_tplus1], p_t, y_t))
            Y_decoded.append((p_t_c, y_t))
            t -= 1
        Y_decoded.reverse()
        Y_decoded = [y for (__, y) in Y_decoded]
        
        while(top_node.frwdlink):
            y = top_node.frwdlink.label
            Y_decoded.append(y)
            top_node = top_node.frwdlink
#         print(Y_decoded)
        return(Y_decoded)
    
    def search(self, alpha, back_track, T, K):
        """A* star searcher uses the score matrix  (built using viterbi algorithm) to decode top-K list of sequences
        
           Args:
               alpha: score matrix build using the viterbi algorithm
               back_track: back_pointers dictionary tracking the best paths to every state
               T: last decoded position of a sequence (in this context, it is the alpha.shape[0])
               K: number of top decoded sequences to be returned
               
           Returns:
               topk_list: top-K list of decoded sequences
           
        
        """
        # push the best astar nodes to the queue (i.e. the pi's at time T)
        q = AStarAgenda()
        r = set()
        c = 0
        P_codebook_rev = self.P_codebook_rev
        # create nodes from the pi's at time T
        for pi_c in P_codebook_rev:
            cost = alpha[T, pi_c]
            pos = T
            frwdlink = None
            label = self.get_node_label(pi_c)
            node = AStarNode(cost, pos, pi_c, label, frwdlink)
#             node.print_node()
            q.push(node, cost)
            
        track = []
        topk_list = []
        try:
            while c < K:
                #print("heap size ", len(q.qagenda))
                top_node = q.pop()
                track.append(top_node)
                
                for i in reversed(range(2, top_node.position+1)):
                    best_prev_pi_c, best_y = back_track[i, top_node.pi_c]
                    pos = i - 1
                    for prev_pi_c in P_codebook_rev:
                        # create a new astar node
                        if(prev_pi_c != best_prev_pi_c):
                            label = self.get_node_label(prev_pi_c)
                            cost = alpha[pos, prev_pi_c]
                            s = AStarNode(cost, pos, prev_pi_c, label, top_node)
                            q.push(s, cost)
                            
                    # create the backlink of the top_node 
                    cost = alpha[pos, best_prev_pi_c]
                    top_node = AStarNode(cost, pos, best_prev_pi_c, best_y, top_node)
                    
                # decode and check if it is not saved already in topk list
                y_labels = self.infer_labels(track[-1], back_track)
#                 print(y_labels)
                sig = "".join(y_labels)
                if(sig not in r):
                    r.add(sig)
                    topk_list.append(y_labels)
                    c += 1
                    track.pop()
        except (KeyError, IndexError) as e:
            # consider logging the error
            print(e)
    
        finally:
            #print('r ', r)
            #print('topk ', topk_list)
            return(topk_list)
        
class HOSemi_AStarSearcher(object):
    """A* star searcher associated with higher-order CRF model such as :class:`HOSemiCRFAD`
       
       Args:
          P_codebook_rev: reversed codebook of set of proper prefixes in the `P` set
                          e.g. ``{0:'', 1:'P', 2:'L', 3:'O', 4:'L|O', ...}``
          P_elems: dictionary comprising the composing elements of every prefix in the `P` set
                   e.g. ``{'':('',), 'P':('P',), 'L':('L',), 'O':('O',), 'L|O':('L','O'), ...}``

       Attributes:
          P_codebook_rev: reversed codebook of set of proper prefixes in the `P` set
                          e.g. ``{0:'', 1:'P', 2:'L', 3:'O', 4:'L|O', ...}``
          P_elems: dictionary comprising the composing elements of every prefix in the `P` set
                   e.g. ``{'':('',), 'P':('P',), 'L':('L',), 'O':('O',), 'L|O':('L','O'), ...}``
    """
    def __init__(self, P_codebook_rev, pi_elems):
        self.P_codebook_rev = P_codebook_rev
        self.pi_elems = pi_elems
        
    def get_node_label(self, pi_code):
        """get the the label/state given a prefix code
        
           Args:
               pi_code: prefix code which is an element of :attr:`P_codebook_rev`
        """
        pi = self.P_codebook_rev[pi_code]
        y =  self.pi_elems[pi][-1]
        return(y)

    def infer_labels(self, top_node, back_track):
        """decode sequence by inferring labels
        
           Args:
               top_node: instance of :class:`AStarNode` class
               back_track: dictionary containing back pointers tracking the best paths to every state
        """
        # decoding the sequence
        #print("we are decoding")
        #top_node.print_node()
        y = top_node.label
        pi_c = top_node.pi_c
        pos = top_node.position
        Y_decoded = []
        
        d, pt_c, yt = back_track[pos, pi_c]
        for _ in range(d+1):
            Y_decoded.append(y)
            
        t = pos - d - 1
        while t>0:
            new_d, new_pt_c, new_yt = back_track[t, pt_c]
            for _ in range(new_d+1):
                Y_decoded.append(yt)
            t = t - new_d -1
            pt_c = new_pt_c
            yt = new_yt
        Y_decoded.reverse()   
        
        while(top_node.frwdlink):
            y = top_node.frwdlink.label
            Y_decoded.append(y)
            top_node = top_node.frwdlink
#         print(Y_decoded)
        return(Y_decoded)
    
    def search(self, alpha, back_track, T, K):
        """A* star searcher uses the score matrix  (built using viterbi algorithm) to decode top-K list of sequences
        
           Args:
               alpha: score matrix build using the viterbi algorithm
               back_track: back_pointers dictionary tracking the best paths to every state
               T: last decoded position of a sequence (in this context, it is the alpha.shape[0])
               K: number of top decoded sequences to be returned
               
           Returns:
               topk_list: top-K list of decoded sequences
           
        
        """
        # push the best astar nodes to the queue (i.e. the pi's at time T)
        q = AStarAgenda()
        r = set()
        c = 0
        P_codebook_rev = self.P_codebook_rev

        # create nodes from the pi's at time T
        for pi_c in P_codebook_rev:
            cost = alpha[T, pi_c]
            pos = T
            frwdlink = None
            label = self.get_node_label(pi_c)
            node = AStarNode(cost, pos, pi_c, label, frwdlink)
#             node.print_node()
            q.push(node, cost)
            
        track = []
        topk_list = []
        try:
            while c < K:
                #print("heap size ", len(q.qagenda))
                top_node = q.pop()
                track.append(top_node)
                while(True):
                    curr_pos = top_node.position
                    if(curr_pos == 1):
                        break
                    d, best_prev_pi_c, best_prev_y = back_track[curr_pos, top_node.pi_c]
                    prev_pos = curr_pos - d - 1
                    for prev_pi_c in P_codebook_rev:
                        # create a new astar node
                        if(prev_pi_c != best_prev_pi_c):
                            label = self.get_node_label(prev_pi_c)
                            cost = alpha[prev_pos, prev_pi_c]
                            s = AStarNode(cost, prev_pos, prev_pi_c, label, top_node)
                            q.push(s, cost)
                            
                    # create the backlink of the top_node 
                    cost = alpha[prev_pos, best_prev_pi_c]
                    top_node = AStarNode(cost, prev_pos, best_prev_pi_c, best_prev_y, top_node)
                    
                # decode and check if it is not saved already in topk list
                y_labels = self.infer_labels(track[-1], back_track)
#                 print(y_labels)
                sig = "".join(y_labels)
                if(sig not in r):
                    r.add(sig)
                    topk_list.append(y_labels)
                    c += 1
                    track.pop()
        except (KeyError, IndexError) as e:
            # consider logging the error
            print(e)
    
        finally:
            #print('r ', r)
            #print('topk ', topk_list)
            return(topk_list)

class TemplateGenerator(object):
    """template generator class for feature/function template generation
    """

    def __init__(self):
        pass
    
    def generate_template_XY(self, attr_name, x_spec, y_spec, template):
        r"""generate template XY for the feature extraction
        
           Args:
               attr_name: string representing the attribute name of the atomic observations/tokens
               x_spec: tuple of the form  (n-gram, range)
                       that is we can specify the n-gram features required in a specific range/window
                       for an observation token ``attr_name``
               y_spec: string specifying how to join/combine the features on the X observation level
                       with labels on the Y level. 
                       
                       Example of passed options would be:
                           - one state (i.e. current state) by passing ``1-state`` or 
                           - two states (i.e. current and previous state) by passing ``2-states`` or
                           - one and two states (i.e. mix/combine observation features with one state model and two states models)
                             by passing ``1-state:2-states``. Higher order models support models with states > 2 such as ``3-states`` and above. 
               template: dictionary that accumulates the generated feature template for all attributes
               
           Example:
           
               suppose we have `word` attribute referenced by 'w' and we need to use the current word
               with the current label (i.e. unigram of words with the current label) in a range of (0,1)
            
               ::
               
                   templateXY = {}
                   generate_template_XY('w', ('1-gram', range(0, 1)), '1-state', templateXY)
               
               we can also specify a two states/labels features at the Y level
               
               ::
               
                   generate_template_XY('w', ('1-gram', range(0, 1)), '1-state:2-states', templateXY)
               
           .. note ::
              this can be applied for every attribute name and accumulated in the `template` dictionary
        """
        ngram_options, wsize = x_spec
        templateX = self._traverse_x(attr_name, ngram_options, wsize)
        templateY = self.generate_template_Y(y_spec)
        templateXY = self._mix_template_XY(templateX, templateY)
        #update the template we are building
        self._update_template(template, templateXY)
        
    def _update_template(self, template, templateXY):  
        """update the accumulated template with the current generated templateXY
        
           Args:
               template: dictionary of the accumulated template for the different offsets
                         and attribute names
               templateXY: dictionary of the form ``{attr_name:{x_offset:(y_offsets)}}``
        """
        for attr_name in templateXY:
            if(attr_name in template):
                for x_offset in templateXY[attr_name]:
                    template[attr_name][x_offset] = templateXY[attr_name][x_offset]
            else:
                template[attr_name] = templateXY[attr_name] 
                             
    def _traverse_x(self, attr_name, ngram_options, wsize):
        """generate template on the X observation level only
        
           Args:
               attr_name: string representing the attribute name of the atomic observations/tokens
               ngram_options: string specifying the n-grams (i.e. ``1-gram``) it also supports multiple
                              specification such as ``1-gram:2-gram`` where each is separated by a colon
               wsize: a range specifying the window size where the template operates
               
        """
        options = ngram_options.split(":")
        l = list(wsize)
        template = {attr_name:{}}
        for option in options:
            n = int(option.split("-")[0])
            ngram_list = self.generate_ngram(l, n)
            for offset in ngram_list:
                template[attr_name][offset] = None
        return(template)
    
    def generate_template_Y(self, ngram_options):
        """generate template on the Y labels level
        
           Args:
               ngram_options: string specifying the number of states to be use (i.e. ``1-state``).
                              It also supports multiple specification such as ``1-state:2-states`` 
                              where each is separated by a colon
               
        """
        template = {'Y':[]}
        options = ngram_options.split(":")
        for option in options:
            max_order = int(option.split("-")[0])
            template['Y'] += self._traverse_y(max_order, accumulative = False)['Y']
        return(template)
    
    @staticmethod
    def _traverse_y(max_order, accumulative = True):
        """generate the y template"""
        attr_name = 'Y'
        template = {attr_name:[]}
        if(accumulative):
            for j in range(max_order):
                offsets_y = [-i for i in range(j+1)]
                offsets_y = tuple(reversed(offsets_y))
                template[attr_name].append(offsets_y)
        else:
            offsets_y = [-i for i in range(max_order)]
            offsets_y = tuple(reversed(offsets_y))
            template[attr_name].append(offsets_y) 
    
        return(template)
    
    @staticmethod
    def _mix_template_XY(templateX, templateY):
        """mix and join the template on the X observation level with the Y level
           
           Args:
               templateX: dictionary of the form ``{attr_name:{x_offset:None}}``
                          e.g. ``{'w': {(0,): None}}``
               templateY: dictionary of the form ``{'Y':[y_offset]}``
                          e.g. ``{'Y': [(0,), (-1, 0)]}``
           .. note::
           
              - x_offset is a tuple of offsets representing the ngram options needed 
                such as (0,) for unigram and (-1,0) for bigram
                
              - y_offset is a tuple of offsets representing the number of states options needed 
                such as (0,) for 1-state and (-1,0) for 2-states and (-2,-1,0) for 3-states
        """
        template_XY = deepcopy(templateX)
        for attr_name in template_XY:
            for offset_x in template_XY[attr_name]:
                template_XY[attr_name][offset_x] = tuple(templateY['Y'])
        return(template_XY)
    
    @staticmethod
    def generate_ngram(l, n):
        """n-gram generator based on the length of the window and the ngram option
        
           Args:
               l: list of positions of the range representing the window size (i.e. list(wsize))
               n: integer representing the n-gram option (i.e. 1 for unigram, 2 for bigram, etc..)
        """
        ngram_list = []
        for i in range(0, len(l)):
            elem = tuple(l[i:i+n])
            if(len(elem) != n):
                break
            ngram_list.append(elem)
            
        return(ngram_list)
    
    @staticmethod
    def generate_combinations(n):
        """generates all possible combinations based on the maximum number of ngrams n
        
           Args:
               n: integer specifying the maximum/greatest ngram option
               
        """
        option_names = []
        start = 1
        for i in range(start, n+1):
            option_names.append("{}-gram".format(i))
            
        config = {}
        for i in range(start, n+1):
            config[i] = list(combinations(option_names, i))
            
        config_combinations = {}
        for c_list in config.values():
            for c_tup in c_list:
                key_name = ":".join(c_tup)
                config_combinations[key_name] = set()
        elemkeys = config_combinations.keys()
        for option_i in config_combinations:
            s = config_combinations[option_i]
            for option_j in elemkeys:
                s.add(option_j)
            config_combinations[option_i] = s
        return(config_combinations)
    
class BoundNode(object):
    """boundary entity class used when generating all possible partitions within specified constraint
    
       Args:
           parent: instance of :class:`BoundNode` 
           boundary: tuple (u,v) representing the current boundary
    """
    def __init__(self, parent, boundary):
        self.parent = parent
        self.boundary = boundary
        self.children = []
        
    def add_child(self, child):
        """add link to the child nodes"""
        self.children.append(child)
    
    def get_child(self):
        """retrieve child nodes"""
        return(self.children.pop())
    
    def get_signature(self):
        """retrieve the id of the node"""
        return(id(self))
    
def generate_partitions(boundary, L, patt_len, bound_node_map, depth_node_map, parent_node, depth=1):
    """generate all possible partitions within the range of segment length and model order
    
       it transforms the partitions into a tree of nodes starting from the root node
       that uses `boundary` argument in its construction
       
       Args:
           boundary: tuple (u,v) representing the current boundary in a sequence
           L: integer representing the maximum length a segment could be constructed
           patt_len: integer representing the maximum model order
           bound_node_map: dictionary that keeps track of all possible partitions represented as
                           instances of :class:`BoundNode`
           depth_node_map: dictionary that arranges the generated nodes by their depth in the tree
           parent_node: instance of :class:`BoundNode` or None in case of the root node
           depth: integer representing the maximum depth of the tree to be reached before stopping 
    """
    if(depth >= patt_len):
        return

    if(parent_node):
        if(boundary in bound_node_map):
            curr_node = bound_node_map[boundary]
        else:
            curr_node = BoundNode(parent_node, boundary)
            bound_node_map[boundary] = curr_node
            if(depth in depth_node_map):
                depth_node_map[depth].append(curr_node)
            else:
                depth_node_map[depth] = [curr_node]
    else:
        # setup root node
        curr_node = BoundNode(None, boundary)
        bound_node_map[boundary] = curr_node
        depth_node_map[depth] = [curr_node]

    u= boundary[0]-1
    v= u
    depth += 1

    for d in range(L):
        if(u-d < 1):
            break
        upd_boundary = (u-d, v)
        if(upd_boundary in bound_node_map):
            child = bound_node_map[upd_boundary]
        else:
            child = BoundNode(curr_node, upd_boundary)
            bound_node_map[upd_boundary] = child
            if(depth in depth_node_map):
                depth_node_map[depth].append(child)
            else:
                depth_node_map[depth] = [child]
        curr_node.add_child(child)
        generate_partitions(upd_boundary, L, patt_len, bound_node_map, depth_node_map, child, depth)
        
def generate_partition_boundaries(depth_node_map):
    """generate partitions of the boundaries generated in :func:`generate_partitions` function
    
       Args:
           depth_node_map: dictionary that arranges the generated nodes by their depth in the tree
                           it is constructed using :func:`generate_partitions` function
    """
    g = {}
    depths = sorted(depth_node_map, reverse=True)
    
    for depth in depths:
        g[depth] = []
        nodes = depth_node_map[depth]
        for curr_node in nodes:
            l = []
            l.append(curr_node.boundary)
            while(True):
                curr_node = curr_node.parent
                if(curr_node):
                    l.append(curr_node.boundary)
                else:
                    g[depth].append(l)
                    break

    return(g)
        
def delete_directory(directory):
    if(os.path.isdir(directory)):
        shutil.rmtree(directory)
        
def delete_file(filepath):
    check = os.path.isfile(filepath)
    if(check):
        os.remove(filepath)
                
def create_directory(folder_name, directory = "current"):
    """create directory/folder (if it does not exist) and returns the path of the directory
    
       Args:
           folder_name: string representing the name of the folder to be created
       
       Keyword Arguments:
           directory: string representing the directory where to create the folder
                      if `current` then the folder will be created in the current directory
    """
    if directory == "current":
        path_current_dir = os.path.dirname(__file__)
    else:
        path_current_dir = directory
    path_new_dir = os.path.join(path_current_dir, folder_name)
    if not os.path.exists(path_new_dir):
        os.makedirs(path_new_dir)
    return(path_new_dir)

def generate_datetime_str():
    """generate string composed of the date and time"""
    datetime_now = datetime.now()
    datetime_str = "{}_{}_{}-{}_{}_{}_{}".format(datetime_now.year,
                                                 datetime_now.month,
                                                 datetime_now.day,
                                                 datetime_now.hour,
                                                 datetime_now.minute,
                                                 datetime_now.second,
                                                 datetime_now.microsecond)
    return(datetime_str)



# def vectorized_logsumexp(vec):
#     """vectorized version of log sum exponential operation
#     
#        Args:
#            vec: numpy vector where entries are in the log domain
#     """
#     with numpy.errstate(invalid='warn'):
#         max_a = numpy.max(vec)
#         try:
#             res = max_a + numpy.log(numpy.sum(numpy.exp(vec - max_a)))
#         except Warning:
#             res = max_a
#     return(res)

def vectorized_logsumexp(vec):
    """vectorized version of log sum exponential operation
     
       Args:
           vec: numpy vector where entries are in the log domain
    """
    max_a = numpy.max(vec)
    if(max_a != -numpy.inf):
        return(max_a + numpy.log(numpy.sum(numpy.exp(vec - max_a))))
    # case where max_a == -numpy.inf
    return(max_a)


def generate_updated_model(modelparts_dir, modelrepr_class,  
                           model_class, aextractor_obj, 
                           fextractor_class, seqrepresenter_class, ascaler_class=None):
    """update/regenerate CRF models using the saved parts/components
    
       Args:
           modelparts_dir: string representing the directory where model parts are saved
           modelrepr_class: name of the model representation class to be used which has 
                            suffix `ModelRepresentation` such as :class:`HOCRFADModelRepresentation`
           model_class: name of the CRF model class such as :class:`HOCRFAD`
           aextractor_class: name of the attribute extractor class such as :class:`NERSegmentAttributeExtractor`
           fextractor_class: name of the feature extractor class used such as :class:`HOFeatureExtractor`
           seqrepresenter_class: name of the sequence representer class such as :class:`SeqsRepresenter`
           ascaler_class: name of the attribute scaler class such as :class:`AttributeScaler`
           
       .. note::
       
          This function is equivalent to :func:`generate_trained_model` function. However, this function
          uses explicit specification of the arguments (i.e. specifying explicitly the classes to be used)
       
           
    """
    from pyseqlab.attributes_extraction import GenericAttributeExtractor

    
    ycodebook = ReaderWriter.read_data(os.path.join(modelparts_dir, "MR_Ycodebook"))
    mfeatures  = ReaderWriter.read_data(os.path.join(modelparts_dir, "MR_modelfeatures"))
    mfeatures_codebook  = ReaderWriter.read_data(os.path.join(modelparts_dir, "MR_modelfeaturescodebook"))
    L = ReaderWriter.read_data(os.path.join(modelparts_dir, "MR_L"))
    
    # generate model representation
    new_mrepr = modelrepr_class()
    new_mrepr.modelfeatures = mfeatures
    new_mrepr.modelfeatures_codebook = mfeatures_codebook
    new_mrepr.Y_codebook = ycodebook
    new_mrepr.L = L
    new_mrepr.generate_instance_properties()
    
    # generate attribute extractor
    if(type(aextractor_obj) == type(GenericAttributeExtractor)): # case it is a class
        new_attrextractor = aextractor_obj()
    else: # case it is an instance of a class 
        new_attrextractor = aextractor_obj

    # generate feature extractor
    templateX = ReaderWriter.read_data(os.path.join(modelparts_dir, "FE_templateX"))
    templateY = ReaderWriter.read_data(os.path.join(modelparts_dir, "FE_templateY"))
    new_fextractor = fextractor_class(templateX, templateY, new_attrextractor.attr_desc)
    
    # generate sequence representer
    new_seqrepr = seqrepresenter_class(new_attrextractor, new_fextractor)
    
    # generate attribute scaler if applicable
    if(ascaler_class):
        scaling_info = ReaderWriter.read_data(os.path.join(modelparts_dir, "AS_scalinginfo"))
        method = ReaderWriter.read_data(os.path.join(modelparts_dir, "AS_method"))
        new_attrscaler = ascaler_class(scaling_info, method)
        new_seqrepr.attr_scaler = new_attrscaler

    # generate crf instance
    new_crfmodel = model_class(new_mrepr, new_seqrepr, {})
    new_crfmodel.weights = ReaderWriter.read_data(os.path.join(modelparts_dir, "weights"))
    return(new_crfmodel)

def generate_trained_model(modelparts_dir, aextractor_obj):
    """regenerate trained CRF models using the saved trained model parts/components
    
       Args:
           modelparts_dir: string representing the directory where model parts are saved
           aextractor_class: name of the attribute extractor class such as :class:`NERSegmentAttributeExtractor`

    """
    # parse the class description file
    class_desc = []
    with open(os.path.join(modelparts_dir, 'class_desc.txt'), 'r') as f:
        for line in f:
            class_desc.append(line.strip())

    from pyseqlab.features_extraction import HOFeatureExtractor, FOFeatureExtractor, SeqsRepresenter
    seqrepresenter_class = SeqsRepresenter
    if(class_desc[1] == 'HOCRFAD'):
        from pyseqlab.ho_crf_ad import HOCRFAD, HOCRFADModelRepresentation
        modelrepr_class = HOCRFADModelRepresentation
        model_class = HOCRFAD
        fextractor_class = HOFeatureExtractor
    elif(class_desc[1] == 'HOCRF'):
        from pyseqlab.ho_crf import HOCRF, HOCRFModelRepresentation
        modelrepr_class = HOCRFModelRepresentation
        model_class = HOCRF
        fextractor_class = HOFeatureExtractor
    elif(class_desc[1] == 'HOSemiCRFAD'):
        from pyseqlab.hosemi_crf_ad import HOSemiCRFAD, HOSemiCRFADModelRepresentation
        modelrepr_class = HOSemiCRFADModelRepresentation
        model_class = HOSemiCRFAD
        fextractor_class = HOFeatureExtractor
    elif(class_desc[1] == 'HOSemiCRF'):
        from pyseqlab.hosemi_crf import HOSemiCRF, HOSemiCRFModelRepresentation
        modelrepr_class = HOSemiCRFModelRepresentation
        model_class = HOSemiCRF
        fextractor_class = HOFeatureExtractor
    elif(class_desc[1] == 'FirstOrderCRF'):
        from pyseqlab.fo_crf import FirstOrderCRF, FirstOrderCRFModelRepresentation
        modelrepr_class = FirstOrderCRFModelRepresentation
        model_class = FirstOrderCRF
        fextractor_class = FOFeatureExtractor
    
    # generate attribute scaler if applicable
    if(class_desc[-1] != 'None'):
        from pyseqlab.attributes_extraction import AttributeScaler
        ascaler_class = AttributeScaler
    else:
        ascaler_class = None

    trained_model = generate_updated_model(modelparts_dir, modelrepr_class, model_class,
                                           aextractor_obj, fextractor_class, seqrepresenter_class, ascaler_class)

    return(trained_model)


def split_data(seqs_id, options):
    r"""utility function for splitting dataset (i.e. training/testing and cross validation)
    
       Args:
           seqs_id: list of processed sequence ids
           options: dictionary comprising of the options on how to split data
           
       Example:
           To perform cross validation, we need to specify
               - cross-validation for the `method`
               - the number of folds for the `k_fold`
               
           ::
               
               options = {'method':'cross_validation',
                          'k_fold':number
                         }
                         
           To perform random splitting, we need to specify
               - random for the `method`
               - number of splits for the `num_splits`
               - size of the training set in percentage for the `trainset_size`
               
           ::
               
               options = {'method':'random',
                          'num_splits':number,
                          'trainset_size':percentage
                         }
    """
    N = len(seqs_id)
    data_split = {}
    method = options.get('method')
    if(method == None):
        method = 'cross_validation'
    if(method == "cross_validation"):
        k_fold = options.get("k_fold")
        if(type(k_fold) != int):
            # use 10 fold cross validation
            k_fold = 10
        elif(k_fold <= 0):
            k_fold = 10
        batch_size = int(numpy.ceil(N/k_fold))
        test_seqs = seqs_id.copy()
        seqs_len = len(test_seqs)
        #numpy.random.shuffle(test_seqs)
        indx = numpy.arange(0, seqs_len + 1, batch_size)
        if(indx[-1] < seqs_len):
            indx = numpy.append(indx, [seqs_len])
            
        for i in range(len(indx)-1):
            data_split[i] = {}
            current_test_seqs = test_seqs[indx[i]:indx[i+1]]
            data_split[i]["test"] = current_test_seqs
            data_split[i]["train"] = list(set(seqs_id)-set(current_test_seqs))

    elif(method == "random"):
        num_splits = options.get("num_splits")
        if(type(num_splits) != int):
            num_splits = 5
        trainset_size = options.get("trainset_size")
        if(type(trainset_size) != int):
            # 80% of the data set is training and 20% for testing
            trainset_size = 80
        elif(trainset_size <= 0 or trainset_size >=100):
            trainset_size = 80
        for i in range(num_splits):
            data_split[i] = {}
            current_train_seqs = numpy.random.choice(seqs_id, int(N*trainset_size/100), replace = False)
            data_split[i]["train"] = list(current_train_seqs)
            data_split[i]["test"] = list(set(seqs_id)-set(current_train_seqs))
            
    return(data_split)


"""split data based on sequences length
   we need to execute the three functions in order:
       (1) :func:`group_seqs_by_length`
       (2) :func:`weighted_sample`
       (3) :func:`aggregate_weightedsample`
"""
def group_seqs_by_length(seqs_info):
    """group sequences by their length
    
       Args:
           seqs_info: dictionary comprsing info about the sequences
                      it has this form {seq_id:{T:length of sequence}}
                      
       .. note::
       
          sequences that are with unique sequence length are grouped together as singeltons
    """
    grouped_seqs = {}
    for seq_id, seq_info in seqs_info.items():
        T = seq_info["T"]
        if(T in grouped_seqs):
            grouped_seqs[T].append(seq_id)
        else:
            grouped_seqs[T] = [seq_id]
    # loop to regroup single sequences
    singelton = [T for T, seqs_id in grouped_seqs.items() if len(seqs_id) == 1]
    singelton_seqs = []
    for T in singelton:
        singelton_seqs += grouped_seqs[T]
        del grouped_seqs[T]

    grouped_seqs["singleton"] = singelton_seqs
    return(grouped_seqs)
    
def weighted_sample(grouped_seqs, trainset_size):
    """get a random split of the grouped sequences
    
       Args:
           grouped_seqs: dictionary of the grouped sequences based on their length
                         it is obtained using :func:`group_seqs_by_length` function
           trainset_size: integer representing the size of the training set in percentage
           
    """
    options = {'method':'random', 'num_splits':1, 'trainset_size':trainset_size}
    wsample = {}
    for group_var, seqs_id in grouped_seqs.items():
#         quota = trainset_size*count_seqs[group_var]/total
        data_split = split_data(seqs_id, options)
        wsample[group_var] = data_split[0]
    return(wsample)

def aggregate_weightedsample(w_sample):
    """represent the random picked sample for training/testing
     
       Args:
           w_sample: dictionary representing a random split of the grouped sequences
                     by their length. it is obtained using :func:`weighted_sample` function
    """
    wdata_split= {"train":[],
                  "test": []}
    for grouping_var in w_sample:
        for data_cat in w_sample[grouping_var]:
            wdata_split[data_cat] += w_sample[grouping_var][data_cat]
    return({0:wdata_split})
##################################

def nested_cv(seqs_id, outer_kfold, inner_kfold):
    """generate nested cross-validation division of sequence ids
    """
    outer_split = split_data(seqs_id, {'method':'cross_validation', 'k_fold':outer_kfold})
    cv_hierarchy = {}
    for outerfold, outer_datasplit in outer_split.items():
        cv_hierarchy["{}_{}".format("outer", outerfold)] = outer_datasplit
        curr_train_seqs = outer_datasplit['train']
        inner_split = split_data(curr_train_seqs, {'method':'cross_validation', 'k_fold':inner_kfold}) 
        for innerfold, inner_datasplit in inner_split.items():
            cv_hierarchy["{}_{}_{}_{}".format("outer", outerfold, "inner", innerfold)] = inner_datasplit
    return(cv_hierarchy)

def get_conll00():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(current_dir)
    files_info = {'train_short_main.txt':('main', True, " "), 
                  'train_short_none.txt':(('w','pos'), True, " "),
                  'train_short_per_sequence.txt':('per_sequence', True, " ")
                  }
    for file_name in files_info:
        parser = DataFileParser()
        print(file_name)
        file_path = os.path.join(root_dir, "tests", "dataset","conll00",file_name)
        for seq in parser.read_file(file_path, header=files_info[file_name][0], y_ref = files_info[file_name][1], column_sep=files_info[file_name][2]):
            print(seq)
    
if __name__ == "__main__":
    pass
    #get_conll00()

