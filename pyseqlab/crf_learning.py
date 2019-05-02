'''
@author: Ahmed Allam <ahmed.allam@yale.edu>
'''

import os
from datetime import datetime
import numpy
from .utilities import ReaderWriter, create_directory, generate_datetime_str, vectorized_logsumexp

class Learner(object):
    """learner used for training CRF models supporting search- and gradient-based learning methods
    
       Args:
           crf_model: an instance of CRF models such as :class:`HOCRFAD`
       
       Keyword Arguments:
           crf_model: an instance of CRF models such as :class:`HOCRFAD`
           training_description: dictionary that will include the training specification
                                 of the model

    """
    def __init__(self, crf_model):
        self.crf_model = crf_model
        self.training_description = None
    
    def train_model(self, w0, seqs_id, optimization_options, working_dir, save_model = True):
        r"""the **MAIN** method for training models using the various options available
        
           Args:
               w0: numpy vector representing initial weights for the parameters
               seqs_id: list of integers representing the sequence ids
               optimization_options: dictionary specifying the training method
               working_dir: string representing the directory where the model data
                            and generated files will be saved
                            
           Keyword Arguments:
               save_model: boolean specifying if to save the final model
               
           Example:
               
               The available options for training are:
                   - `SGA` for stochastic gradient ascent
                   - `SGA-ADADELTA` for stochastic gradient ascent using ADADELTA approach
                   - `BFGS` or `L-BFGS-B` for optimization using second order information (hessian matrix)
                   - `SVRG` for stochastic variance reduced gradient method
                   - `COLLINS-PERCEPTRON` for structured perceptron
                   - `SAPO` for Search-based Probabilistic Online Learning Algorithm (SAPO) (an adapted version)
               
           For example possible specification of the optimization options are:
               
               ::
               
                    1) {'method': 'SGA-ADADELTA'
                       'regularization_type': {'l1', 'l2'}
                       'regularization_value': float
                       'num_epochs': integer
                       'tolerance': float
                       'rho': float
                       'epsilon': float
                      }
                      
                      
                   2) {'method': 'SGA' or 'SVRG'
                       'regularization_type': {'l1', 'l2'}
                       'regularization_value': float
                       'num_epochs': integer
                       'tolerance': float
                       'learning_rate_schedule': one of ("bottu", "exponential_decay", "t_inverse", "constant")
                       't0': float
                       'alpha': float
                       'eta0': float
                      }
                                   
                                  
                   3) {'method': 'L-BFGS-B' or 'BFGS'
                       'regularization_type': 'l2'
                       'regularization_value': float
                       'disp': False
                       'maxls': 20,
                       'iprint': -1,
                       'gtol': 1e-05,
                       'eps': 1e-08, 
                       'maxiter': 15000, 
                       'ftol': 2.220446049250313e-09, 
                       'maxcor': 10, 
                       'maxfun': 15000
                       }
                
                
                   4) {'method': 'COLLINS-PERCEPTRON'
                       'regularization_type': {'l1', 'l2'}
                       'regularization_value': float
                       'num_epochs': integer
                       'update_type':{'early', 'max-fast', 'max-exhaustive', 'latest'}
                       'shuffle_seq': boolean
                       'beam_size': integer
                       'avg_scheme': {'avg_error', 'avg_uniform'}
                       'tolerance': float
                      }
                      
                      
                   5) {'method': 'SAPO'
                       'regularization_type': {'l2'}
                       'regularization_value': float
                       'num_epochs': integer
                       'update_type':'early'
                       'shuffle_seq': boolean
                       'beam_size': integer
                       'topK': integer
                       'tolerance': float
                      }
                         
        """

        pop_keys = set()
        
        lambda_type = optimization_options.get("regularization_type")
        pop_keys.add("regularization_type")
        if(lambda_type not in {'l1', 'l2'}):
            # default regularization type is l2
            lambda_type = 'l2'
            #^print("regularization by default is l2")

        # get the regularization parameter value
        lambda_val = optimization_options.get("regularization_value")
        pop_keys.add("regularization_value")
        
        if(lambda_val == None):
            # assign default lambda value
            lambda_val = 0.0
        elif(lambda_val < 0):
            # regularization should be positive
            lambda_val = 0.0

        # initialization of weight vector w node
#         w0 = numpy.zeros(len(self.weights))
        method = optimization_options.get("method")
        pop_keys.add("method")
        if(method not in {"L-BFGS-B", "BFGS", "SGA","SGA-ADADELTA","SVRG","COLLINS-PERCEPTRON", "SAPO"}):
            # default weight learning/optimization method
            method = "SGA-ADADELTA"
            
        if(method in {"L-BFGS-B", "BFGS"}):
            # initialize the new optimization options
            option_keys = set(optimization_options.keys()) - pop_keys
            options = {elmkey:optimization_options[elmkey] for elmkey in option_keys}
            optimization_config = {'method':method,
                                   'regularization_value':lambda_val,
                                   'regularization_type':'l2',
                                   'options':options
                                   }
            estimate_weights = self._optimize_scipy

            
        elif(method in {"SGA", "SGA-ADADELTA", "SVRG", "COLLINS-PERCEPTRON", "SAPO"}):
            num_epochs = optimization_options.get("num_epochs")
            if(type(num_epochs) != int):
                # default number of epochs if not specified
                num_epochs = 3
            elif(num_epochs < 0):
                # num_epochs should be positive
                num_epochs = 3
                
            tolerance = optimization_options.get("tolerance")
            if(tolerance == None):
                # default value of tolerance if not specified
                tolerance = 1e-8
            elif(tolerance < 0):
                tolerance = 1e-8
                
            optimization_config = {'method': method,
                                   'regularization_type':lambda_type,
                                   'regularization_value': lambda_val,
                                   'num_epochs': num_epochs,
                                   'tolerance': tolerance
                                   }
            
            if(method in {"COLLINS-PERCEPTRON", "SAPO"}):

                # if segmentation problem the non-entity symbol is specified using this option else it is None
                seg_other_symbol = optimization_options.get("seg_other_symbol")
                optimization_config['seg_other_symbol'] = seg_other_symbol    
                # setting beam size
                beam_size = optimization_options.get("beam_size")
                # default beam size
                default_beam = len(self.crf_model.model.Y_codebook)
                if(type(beam_size) != int):
                    beam_size = default_beam
                elif(beam_size <= 0 or beam_size > default_beam):
                    beam_size = default_beam
                optimization_config["beam_size"] = beam_size
                self.crf_model.beam_size = beam_size

                # setting update type 
                update_type = optimization_options.get("update_type")
                if(update_type not in {'early', 'latest', 'max-exhaustive', 'max-fast'}):
                    update_type = 'early'
                optimization_config["update_type"] = update_type
                
                # setting shuffle_seq
                shuffle_seq = optimization_options.get("shuffle_seq")
                if(type(shuffle_seq) != bool):
                    shuffle_seq = False
                optimization_config["shuffle_seq"] = shuffle_seq
                if(method == "COLLINS-PERCEPTRON"):
                    # getting averaging scheme
                    avg_scheme = optimization_options.get("avg_scheme")
                    if(avg_scheme not in ("avg_uniform", "avg_error", "survival")):
                        avg_scheme = "avg_error"
                    optimization_config["avg_scheme"] = avg_scheme
                    estimate_weights = self._structured_perceptron
                else:
                    # getting gamma (i.e. learning rate)
                    gamma = optimization_options.get("gamma")
                    if(gamma == None):
                        # use default value
                        gamma = 1
                    elif(gamma < 0):
                        gamma = 1
                    optimization_config['gamma'] = gamma
                    # getting topK (i.e. top-K decoded sequences)
                    topK = optimization_options.get("topK")
                    if(topK == None):
                        # use default value
                        topK = 5
                    elif(topK < 0):
                        topK = 5
                    optimization_config['topK'] = topK
                    estimate_weights = self._sapo


            elif(method in {"SGA", "SVRG"}):
            
                # get the other parameters to be tuned such as t0 and alpha
                learning_rate_schedule = optimization_options.get("learning_rate_schedule")
                if(learning_rate_schedule not in {"bottu", "exponential_decay", "t_inverse", "constant"}):
                    # default learning rate schedule
                    learning_rate_schedule = "t_inverse"
                optimization_config["learning_rate_schedule"] = learning_rate_schedule
                
                t0 = optimization_options.get("t0")
                if(t0 == None):
                    # use default value
                    t0 = 0.1
                elif(t0 < 0):
                    t0 = 0.1
                optimization_config['t0'] = t0
    
                if(learning_rate_schedule in {"t_inverse", "exponential_decay"}):
                    # get the alpha parameter
                    a = optimization_options.get("a")
                    if(a == None):
                        # use a default value
                        a = 0.9
                    elif(a <= 0 or a >= 1):
                        a = 0.9
                    optimization_config['a'] = a

                if(method == "SGA"):
                    estimate_weights = self._sga_classic   
                else:
                    estimate_weights = self._sga_svrg     
                                
            elif(method == "SGA-ADADELTA"):
                estimate_weights = self._sga_adadelta
                
                p_rho = optimization_options.get("p_rho")
                if(p_rho == None):
                    # default value
                    p_rho = 0.95
                elif(p_rho < 0):
                    # num_epochs should be positive
                    p_rho = 0.95
                    
                epsilon = optimization_options.get("epsilon")
                if(epsilon == None):
                    # default value of tolerance if not specified
                    epsilon = 1e-6
                elif(epsilon < 0):
                    epsilon = 1e-6
                optimization_config['p_rho'] = p_rho
                optimization_config['epsilon'] = epsilon

        # save the training options
        self.training_description = optimization_config
        model_foldername = generate_datetime_str()
        model_dir = create_directory(model_foldername, create_directory("models", working_dir))
        model_name = model_foldername + ".model"
        self.training_description["model_dir"] = model_dir
        self.training_description["model_name"] = model_name
        self.training_description["train_seqs_id"] = seqs_id
        
        # if everything is defined correctly then estimate the parameters
        w_hat = estimate_weights(w0, seqs_id)
        # update model weights to w_hat
        self.crf_model.weights = w_hat
        if(save_model):
            # pickle the model
            modelparts_dir = create_directory("model_parts", model_dir)
            self.crf_model.save_model(modelparts_dir)
            
        # cleanup the instance variables
        self.cleanup()
   
            
    def _report_training(self):
        """report training by logging the description to a file"""
        method = self.training_description["method"]
        regularization_type = self.training_description["regularization_type"]
        # regularization parameter lambda
        C = self.training_description['regularization_value']
        model_dir = self.training_description["model_dir"]
        model_name = self.training_description["model_name"]
        # log file 
        log_file = os.path.join(model_dir, "crf_training_log.txt")
        line = "---Model training-- starting time {} \n".format(datetime.now())
        line += "model name: {} \n".format(model_name)
        line += "model directory: {} \n".format(model_dir)
        line += "model type: {} \n".format(self.crf_model.__class__)
        line += "training method: {} \n".format(method)
        if(C):
            line += "type of regularization: {} \n".format(regularization_type)
            line += "value of regularization: {} \n".format(C)
        
        if(method  == "SGA"):
            learning_rate_schedule = self.training_description["learning_rate_schedule"]
            t0 = self.training_description["t0"]
            line += "learning rate schedule: {} \n".format(learning_rate_schedule)
            line += "eta0: {} \n".format(t0)
            if(learning_rate_schedule in ("t_inverse", "exponential_decay")):
                # get the alpha parameter
                a = self.training_description["a"]
                line += "a: {} \n".format(a)
        elif(method == "SGA-ADADELTA"):
            rho = self.training_description["p_rho"]
            epsilon = self.training_description["epsilon"]
            line += "p_rho: {} \n".format(rho)
            line += "epsilon: {} \n".format(epsilon)
        elif(method in {"SAPO", "COLLINS-PERCEPTRON"}):
            update_type = self.training_description['update_type']
            beam_size = self.training_description['beam_size']
            shuffle_seq = self.training_description['shuffle_seq']
            line += "update_type: {} \n".format(update_type)
            line += "beam_size: {} \n".format(beam_size)
            line += "shuffle_seq: {} \n".format(shuffle_seq)
            if(method == "COLLINS-PERCEPTRON"):
                avg_scheme = self.training_description["avg_scheme"]
                line += "averaging scheme: {} \n".format(avg_scheme)
            else:
                gamma = self.training_description['gamma']
                topK = self.training_description['topK']
                line += "gamma (learning rate): {} \n".format(gamma)
                line += "topK (number of top decoded seqs): {} \n".format(topK)
                
        if(method not in ("L-BFGS-B", "BFGS")):
            line += "number of epochs: {} \n".format(self.training_description['num_epochs'])
        # write to file    
        ReaderWriter.log_progress(line, log_file)
        
    def _check_reldiff(self, x, y):
        """calculate relative difference between two numbers
           
           Ars:
               x: float
               y: float
        """
        tolerance = self.training_description["tolerance"]
        if(numpy.abs(y)<=tolerance):
            self._exitloop = True
        else:
            if(x != y):            
                reldiff = numpy.abs(x - y) / (numpy.abs(x) + numpy.abs(y))
                #print("reldiff = {}".format(reldiff))
                if(reldiff <= tolerance):
                    self._exitloop = True
                else:
                    self._exitloop = False 


    def _optscipy_seqs_loglikelihood(self, w, seqs_id):
        """compute seqs loglikelihood when using the BFGS and L-BFGS-B optimization options
        
           Args:
               w: weight vector (numpy vector)
               seqs_id: list of integers representing ids assigned to the sequence
        
        """
        crf_model = self.crf_model
        seqs_loglikelihood = crf_model.compute_seqs_loglikelihood(w, seqs_id)
        # clear cached info 
        crf_model.clear_cached_info(seqs_id)
        # check for regularization parameter
        l2 = self.training_description["regularization_value"]
        if(l2>0):
            # log(p(Y|X;w)) - lambda/2 * ||w||**2
            seqs_loglikelihood = seqs_loglikelihood - ((l2/2) * numpy.dot(w, w))
        # since the optimization will be based on minimization, hence we multiply by -1
        seqs_loglikelihood = seqs_loglikelihood * -1
        return(seqs_loglikelihood)


    def _optscipy_seqs_gradient(self, w, seqs_id):
        """compute seqs gradient when using the BFGS and L-BFGS-B optimization options
        
           Args:
               w: weight vector (numpy vector)
               seqs_id: list of integers representing ids assigned to the sequence
        
        """
        crf_model = self.crf_model
        seqs_grad = crf_model.compute_seqs_gradient(w, seqs_id)
        # clear cached info 
        crf_model.clear_cached_info(seqs_id)
        l2 = self.training_description["regularization_value"]
        if(l2>0):
            seqs_grad = seqs_grad - (l2 * w)
        # since the optimization will be based on minimization, hence we multiply by -1
        seqs_grad = seqs_grad * -1
        return(seqs_grad)
    
  
    def _optimize_scipy(self, w, train_seqs_id):
        """estimate the parameters w of the model using `scipy optimize function`
            
           it uses `optimize.minimize()` function from the scipy package
           
           Args:
               w: weight vector (numpy vector)
               train_seqs_id: list of integers representing ids of the training sequences
        
        """
        from scipy import optimize
        self._report_training() 
        objfunc = self._optscipy_seqs_loglikelihood
        gradfunc = self._optscipy_seqs_gradient
        method = self.training_description["method"]
        options = self.training_description['options']
  
        # to keep track of elapsed time between optimization iterations
        self._elapsed_time = datetime.now()
        self._iter_count = 0
        result = optimize.minimize(fun = objfunc,
                                   x0 = w,
                                   args = (train_seqs_id),
                                   method = method,
                                   jac = gradfunc,
                                   options = options,
                                   callback = self._track_scipy_optimizer)
          
        model_dir = self.training_description["model_dir"]
        # log file 
        log_file = os.path.join(model_dir, "crf_training_log.txt")
        line = "---Model training--- end time {} \n".format(datetime.now())
        line += "\n \n"
        ReaderWriter.log_progress(line, log_file)
          
#         print("results \n {}".format(result))
        print("success: ", result['success'])
#         print(result.keys())
 
        # estimated optimal weights
        w_hat = result.x
          
        return(w_hat)
    
    def _track_scipy_optimizer(self, w):
        """track scipy optimization by logging each iteration 
        
           Args:
               w: weight vector (numpy vector)

        """
        
        # increment iteration count
        self._iter_count += 1
        delta_time = datetime.now() - self._elapsed_time 
        crf_model = self.crf_model
        # approximate estimation of sum of loglikelihood -- using previous weights
        train_seqs_id = self.training_description["train_seqs_id"]
        seqs_loglikelihood = 0
        for seq_id in train_seqs_id:
            seq_loglikelihood = crf_model.seqs_info[seq_id]["loglikelihood"]
            seqs_loglikelihood += seq_loglikelihood
        seqs_loglikelihood *= -1 
          
        """ use the below command >> to compute the sum of sequences' loglikelihood using the updated/current weights
            the sum should be decreasing after each iteration for successful training (used as diagnostics)
            however it is expensive/costly to recompute
            
            >>> seqs_loglikelihood = crf_model.compute_seqs_loglikelihood(w, train_seqs_id)
        """
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")
        line = "--- Iteration {} --- \n".format(self._iter_count)
        line += "Estimated average negative loglikelihood is {} \n".format(seqs_loglikelihood)
        line += "Number of seconds spent: {} \n".format(delta_time.total_seconds())
        ReaderWriter.log_progress(line, log_file)
        self._elapsed_time = datetime.now()
        print("iteration ", self._iter_count)


    
    def _identify_violation_indx(self, viol_indx, y_ref_boundaries):
        """determine the index where the violation occurs
        
           violation means when the reference state falls off the specified beam while decoding
           
           Args:
               viol_indx: list of indices where violation occurred while decoding
               y_ref_boundaries: boundaries of the labels/tags in the reference sequence
        """
        # viol_index is 1-based indexing
        counter = 0
        for boundary in y_ref_boundaries:
            __, v = boundary
            if(v >= viol_indx):
                viol_pos = v
                viol_boundindex = counter + 1
                break
            counter+= 1
        return(viol_pos, viol_boundindex)
    
    def _compute_seq_decerror(self, y_ref, y_imposter, viol_pos):
        """compute the decoding error of a sequence
        
           Args:
               y_ref: reference sequence list of labels 
               y_imposter: imposter/decoded sequence list of labels
               viol_pos: index where violation occurred, 
                         it is identified using :func:`_identify_violation_indx` function
        """
#         print("yref ", y_ref)
#         print("y_imposter ", y_imposter)
#         print("viol_pos ", viol_pos)
        T = len(y_ref[:viol_pos])
        #^print("T ", T)
        #^print("viol_pos ", viol_pos)
        missmatch = [i for i in range(T) if y_ref[i] != y_imposter[i]]
        len_diff = len(missmatch)
        # range of error is [0-1]
        seq_err_count = float(len_diff/T)
        return(seq_err_count)

    def _unpack_windxfval(self, y_windxfval):
        """unpack the weight indices and corresponding feature values 
        
           Args:
               y_windxfval: tuple having two numpy array entries; the first representing
                            the weight indices of the features while the second representing
                            the values that are feature sum/count
        """
        windx, fval = y_windxfval
        return(windx, fval)

    def _find_update_violation(self, w, seq_id):
        """determine the *best* imposter sequence for weight updates
        
           Args:
               w: weight vector (numpy vector)
               seq_id: integer representing unique id assigned to the sequence
        """
        method = self.training_description['method']
        beam_size = self.training_description['beam_size']
        update_type = self.training_description['update_type']
        topK = self.training_description.get('topK')
        crf_model = self.crf_model
        seqs_info = crf_model.seqs_info
        l = {'Y':(seq_id, )}
        crf_model.check_cached_info(seq_id, l)
        y_ref = seqs_info[seq_id]['Y']['flat_y']
        y_ref_boundaries = seqs_info[seq_id]['Y']['boundaries']
        
        if(update_type in {'max-fast', 'max-exhaustive', 'latest'}):
            early_stop = False
        else:
            early_stop = True
        
        if(not topK):
            y_imposter, viol_indx = crf_model.viterbi(w, seq_id, beam_size, early_stop, y_ref)
            y_imposters = [y_imposter]        
        else:
            y_imposters, viol_indx = crf_model.viterbi(w, seq_id, beam_size, early_stop, y_ref, topK)

        seq_err_count = None
        ref_unp_windxfval = None
        imps_unp_windxfval = None
        
        #^print("y_ref ", y_ref)
        #^print("y_imposter ", y_imposter)
        # top decoded sequence
        y_imposter = y_imposters[0]
        if(not viol_indx):
            # we can perform full update
            print("in full update routine ...")
            T = seqs_info[seq_id]['T']
            seq_err_count = self._compute_seq_decerror(y_ref, y_imposter, T)
            if(seq_err_count or method == "SAPO"):       
                ref_unp_windxfval, imps_unp_windxfval = self._load_gfeatures(seq_id, "globalfeatures", y_imposters, T, len(y_ref_boundaries))             

        else:
            if(update_type == "early"):
                print("in early update routine ...")
                # viol_index is 1-based indexing
                earlyviol_indx = viol_indx[0]
                viol_pos, viol_boundindex = self._identify_violation_indx(earlyviol_indx, y_ref_boundaries)
                seq_err_count = self._compute_seq_decerror(y_ref, y_imposter, viol_pos)
                ref_unp_windxfval, imps_unp_windxfval = self._load_gfeatures(seq_id, "globalfeatures_per_boundary", y_imposters, viol_pos, viol_boundindex)

            elif(update_type == "max-exhaustive"):
                # max update is only supported for one imposter sequence
                max_diff = numpy.inf
                L = crf_model.model.L
                print("in max-exhaustive update routine ...")
                test = []
                # viol_index is 1-based indexing
                for i in range(len(viol_indx)):
                    indx = viol_indx[i]
                    if(i == 0):
                        # case of early update index
                        if(L > 1):
                            viol_pos, viol_boundindex = self._identify_violation_indx(indx, y_ref_boundaries)
                        else:
                            viol_pos = indx
                            viol_boundindex = viol_pos
                        seq_err_count = self._compute_seq_decerror(y_ref, y_imposter, viol_pos)
                    else:
                        if(L>1):
                            __, v = y_ref_boundaries[viol_boundindex]
                            viol_pos = v
                            viol_boundindex += 1
                        else:
                            viol_pos = indx
                            viol_boundindex = viol_pos
#                     seq_err_count = self._compute_seq_decerror(y_ref, y_imposter, viol_pos)
                    ref_unp_windxfval, imps_unp_windxfval = self._load_gfeatures(seq_id, "globalfeatures_per_boundary", y_imposters, viol_pos, viol_boundindex)
                    ref_windx, ref_fval = ref_unp_windxfval
                    imp_windx, imp_fval = imps_unp_windxfval[0]
                    
                    diff = numpy.dot(w[ref_windx], ref_fval) - numpy.dot(w[imp_windx], imp_fval)
                    test.append(diff)
#                     print("diff = {}, max_diff = {} ".format(diff, max_diff))
                    if(diff <= max_diff):
                        # using less than or equal would allow for getting the longest sequence having max difference
                        max_diff = diff
                        ref_unp_windxfval = (ref_windx, ref_fval)
                        imp_unp_windxfval = (imp_windx, imp_fval)
                imps_unp_windxfval = [imp_unp_windxfval]
#                 print("test ", test)
            elif(update_type == "max-fast"):
                # based on empirical observation, the last violation index (i.e. where the beam falls off) is almost always yielding the max violation
                # this is a heuristic, for an exhaustive procedure, choose `max-exhaustive`
                # max update is only supported for one imposter sequence
                max_diff = numpy.inf
                L = crf_model.model.L
                print("in max-fast update routine ...")
                # viol_index is 1-based indexing
                lastviol_indx = viol_indx[-1]
                viol_pos, viol_boundindex = self._identify_violation_indx(lastviol_indx, y_ref_boundaries)
                seq_err_count = self._compute_seq_decerror(y_ref, y_imposter, viol_pos)
                ref_unp_windxfval, imps_unp_windxfval = self._load_gfeatures(seq_id, "globalfeatures_per_boundary", y_imposters, viol_pos, viol_boundindex)
            elif(update_type == 'latest'):
                # to implement lastest update at some point..
                pass
                        
        return(ref_unp_windxfval, imps_unp_windxfval, seq_err_count)   
    

        
    def _load_gfeatures(self, seq_id, gfeatures_type, y_imposters, ypos_indx, boundpos_indx):
        """load the global features of the reference and imposter/decoded sequence
        
           Args:
               seq_id: id of the sequence
               gfeatures_type: determine the representation either aggregated or by boundary
               y_imposters: list of imposter sequences
               ypos_indx: index of the considered end of the label sequence
               boundpos_indx: index of the boundary corresponding to the identified `ypos_indx`
        """
        seg_other_symbol = self.training_description['seg_other_symbol']
        crf_model = self.crf_model
        seqs_info = crf_model.seqs_info
        y_ref_boundaries = seqs_info[seq_id]['Y']['boundaries']
        if(gfeatures_type == "globalfeatures"):
            per_boundary = False
            y_ref_boundaries = None
        else:
            per_boundary = True
            # to assign y_ref_boundries here -> y_ref_boundaries = y_ref_boundaries[:boundpos_indx]
        l = {gfeatures_type:(seq_id, per_boundary)}
        crf_model.check_cached_info(seq_id, l)
        ref_gfeatures = seqs_info[seq_id][gfeatures_type]
        if(y_ref_boundaries):
            y_ref_windxfval = crf_model.represent_globalfeature(ref_gfeatures, y_ref_boundaries[:boundpos_indx])
        else:
            y_ref_windxfval = seqs_info[seq_id][gfeatures_type]

        #ref_unp_windxfval = self._unpack_windxfval(y_ref_windxfval)
        # generate global features for the imposters
        imposters_windxfval = []
        for y_imposter in y_imposters:
            # generate global features for the current imposter 
            imposter_gfeatures_perboundary, y_imposter_boundaries = crf_model.load_imposter_globalfeatures(seq_id, y_imposter[:ypos_indx], seg_other_symbol)                     
            #^print("imposter_gfeatures_perboundary ", imposter_gfeatures_perboundary)
            #^print("imposter y_boundaries ", y_imposter_boundaries)
            y_imposter_windxfval = crf_model.represent_globalfeature(imposter_gfeatures_perboundary, y_imposter_boundaries)
            imposters_windxfval.append(y_imposter_windxfval)
        
        return(y_ref_windxfval, imposters_windxfval)   

    
    def _update_weights_sapo(self, w, ref_unp_windxfval, imps_unp_windxfval, prob_vec):
        """update weight vector for the SAPO method
        
           Args:
               w: weight vector (numpy vector)
               ref_unp_windxfval: tuple of two numpy array elements representing the weight indices
                                  and corresponding feature sum/count of the reference sequence
               imps_unp_windxfval: list of tuples each comprising two numpy array elements representing 
                                   the weight indices and corresponding feature sum/count of the imposter sequences
               prob_vec: numpy vector representing the probability of each imposter sequence
        """
        gamma = self.training_description['gamma']
        # update weights using the decoded sequences
        for i in range(len(imps_unp_windxfval)):
            windx, fval = imps_unp_windxfval[i]
            w[windx] -= (gamma*prob_vec[i]) * fval
        # update weights using the reference sequence   
        windx, fval = ref_unp_windxfval
        w[windx] += gamma * fval

    def _compute_probvec_sapo(self, w, imps_unp_windxfval):
        """compute the probabilty of each imposter sequence in the SAPO algorithm
        
           Args:
               w: weight vector (numpy vector)
               imps_unp_windxfval: list of dictionaries (unpacked) representing the weight indices and corresponding feature sum/count
                                   of the imposter sequences
        """
        # normalize
        num_imposters = len(imps_unp_windxfval)
        ll_vec = numpy.zeros(num_imposters)
        for i in range(num_imposters):
            windx, fval = imps_unp_windxfval[i]
            ll_vec[i] = numpy.dot(w[windx], fval)
        Z = vectorized_logsumexp(ll_vec)
        prob_vec = numpy.exp(ll_vec - Z)
#         print("prob_vec ", prob_vec)
        return(prob_vec)
    
    def _sapo(self, w, train_seqs_id):
        """implements Search-based Probabilistic Online Learning Algorithm (SAPO)

           this implementation adapts it to 'violation-fixing' framework (i.e. inexact search is supported)
          
           .. see:: 
           
              original paper at https://arxiv.org/pdf/1503.08381v1.pdf
                       
           .. note:: 
               
              the regularization is based on averaging rather than l2 as it seems to be consistent during training
              while using exact or inexact search
        """
        self._report_training()
        num_epochs = self.training_description["num_epochs"]
#         regularization_type = self.training_description["regularization_type"]
        # regularization parameter lambda
#         C = self.training_description['regularization_value']
#         gamma = self.training_description['gamma']
        shuffle_seq = self.training_description['shuffle_seq']
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")

        N = len(train_seqs_id)
        crf_model = self.crf_model
        # instance variable to keep track of elapsed time between optimization iterations
        self._elapsed_time = datetime.now()
        self._exitloop = False
        
        avg_error_list = [0]
        w_avg = numpy.zeros(len(w), dtype='longdouble')

        for k in range(num_epochs):
            seq_left = N
            error_count = 0
            if(shuffle_seq):
                numpy.random.shuffle(train_seqs_id)
            for seq_id in train_seqs_id:
                ref_unp_windxfval, imps_unp_windxfval, seq_err_count = self._find_update_violation(w, seq_id)
                prob_vec = self._compute_probvec_sapo(w, imps_unp_windxfval)
                self._update_weights_sapo(w, ref_unp_windxfval, imps_unp_windxfval, prob_vec)
                # regularize the weights 
#                 reg = -(C/N)* w
#                 w += gamma*reg
                w_avg += w
                crf_model.clear_cached_info([seq_id])
                seq_left -= 1
                #print('seq_err_count ', seq_err_count)
                if(seq_err_count):
                    error_count += seq_err_count
#                 print("error count {}".format(error_count))
                print("sequences left {}".format(seq_left))
            avg_error_list.append(float(error_count/N))
            self._track_perceptron_optimizer(w, k, avg_error_list)
            ReaderWriter.dump_data(w_avg/((k+1)*N), os.path.join(model_dir, "model_avgweights_epoch_{}".format(k+1)))
            print("average error : {}".format(avg_error_list[1:]))
#             print("self._exitloop {}".format(self._exitloop))
            if(self._exitloop):
                break
            self._elapsed_time = datetime.now()
            
        line = "---Model training--- end time {} \n".format(datetime.now())
        ReaderWriter.log_progress(line, log_file)
        w = w_avg/(num_epochs*N) 
        ReaderWriter.dump_data(avg_error_list, os.path.join(model_dir, 'avg_decodingerror_training'))

        return(w)      
    
    def _update_weights_perceptron(self, w, ref_unp_windxfval, imp_unp_windxfval):
        """update weight vector for the COLLINS-PERCEPTRON method
        
           Args:
               w: weight vector (numpy vector)
               ref_unp_windxfval: dictionary (unpacked) representing the weight indices and corresponding feature sum/count
                                  of the reference sequence
               imps_unp_windxfval: list of dictionaries (unpacked) representing the weight indices and corresponding feature sum/count
                                   of the imposter sequences
        """
        ref_windx, ref_fval = ref_unp_windxfval
        imp_windx, imp_fval = imp_unp_windxfval
        w[ref_windx] += ref_fval
        w[imp_windx] -= imp_fval
        
    def _structured_perceptron(self, w, train_seqs_id):
        """implements structured perceptron algorithm in particular the average perceptron 
            
           it was introduced by Michael Collins in 2002 (see his paper http://www.aclweb.org/anthology/W02-1001)
           this implementation supports different averaging schemes for the weight learning

           Args:
               w: weight vector (numpy vector)
               seqs_id: list of integers representing ids assigned to the sequence
        """
        self._report_training()
        num_epochs = self.training_description["num_epochs"]
        avg_scheme = self.training_description["avg_scheme"]
        shuffle_seq = self.training_description["shuffle_seq"]
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")

        N = len(train_seqs_id)
        crf_model = self.crf_model
        # instance variable to keep track of elapsed time between optimization iterations
        self._elapsed_time = datetime.now()
        self._exitloop = False
        
        if(avg_scheme in {"avg_error", "avg_uniform"}):
            # accumulated sum of estimated weights
            w_avg = numpy.zeros(len(w), dtype = "longdouble")
            avg_error_list = [0]
            num_upd = 0
            for k in range(num_epochs):
                seq_left = N
                error_count = 0
                if(shuffle_seq):
                    numpy.random.shuffle(train_seqs_id)
                for seq_id in train_seqs_id:
                    print("sequences left {}".format(seq_left))
                    ref_unp_windxfval, imps_unp_windxfval, seq_err_count = self._find_update_violation(w, seq_id)
                    # if decoding errors with the current weight occurs
                    #^print("seq_err_count ", seq_err_count)
                    #^print("y_ref_windxfval ", y_ref_windxfval)
                    if(seq_err_count):
                        error_count += seq_err_count
                        if(avg_scheme == "avg_error"):
                            # consider/emphasize more on previous weights that have small average error decoding per sequence
                            w_avg += (1-seq_err_count) * w
                            num_upd += (1-seq_err_count)
                        else:
                            w_avg += w
                            num_upd += 1
                        # update current weight
                        self._update_weights_perceptron(w, ref_unp_windxfval, imps_unp_windxfval[0])
                    crf_model.clear_cached_info([seq_id])
                    seq_left -= 1
#                 print("error count {}".format(error_count))
                avg_error_list.append(float(error_count/N))
                self._track_perceptron_optimizer(w, k, avg_error_list)
                if(num_upd):
                    w_dump = w_avg/num_upd
                else:
                    w_dump = w_avg
                ReaderWriter.dump_data(w_dump, os.path.join(model_dir, "model_avgweights_epoch_{}".format(k+1)))
                print("average error : {}".format(avg_error_list[1:]))
#                 print("self._exitloop {}".format(self._exitloop))
                if(self._exitloop):
                    break
                self._elapsed_time = datetime.now()
            if(num_upd):
                w = w_avg/num_upd
            
        line = "---Model training--- end time {} \n".format(datetime.now())
        ReaderWriter.log_progress(line, log_file)
        ReaderWriter.dump_data(avg_error_list, os.path.join(model_dir, 'avg_decodingerror_training'))

        return(w)      

    def _track_perceptron_optimizer(self, w, k, avg_error_list):
        """track search based optimized (such as SAPO and COLLINS-PERCEPTRON) by logging each iteration 
        
           Args:
               w: weight vector (numpy vector)
               k: current epoch
               avg_error_list: list of the decoding errors in each previous epochs

        """
        delta_time = datetime.now() - self._elapsed_time 
        self._check_reldiff(avg_error_list[-2], avg_error_list[-1])
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")
        line = "--- Iteration {} --- \n".format(k+1)
        line += "Average percentage of decoding error: {} \n".format(avg_error_list[-1]*100)
        line += "Number of seconds spent: {} \n".format(delta_time.total_seconds())
        ReaderWriter.log_progress(line, log_file)
        # dump the learned weights for every pass
        ReaderWriter.dump_data(w, os.path.join(model_dir, "model_weights_epoch_{}".format(k+1)))

        
    def _sga_adadelta(self, w, train_seqs_id):
        """implements stochastic gradient ascent using adaptive approach of ADADELTA 
            
           the original paper is found in https://arxiv.org/abs/1212.5701

           Args:
               w: weight vector (numpy vector)
               train_seqs_id: list of integers representing ids assigned to the sequence
        """
        self._report_training()
        crf_model = self.crf_model
        num_epochs = self.training_description["num_epochs"]
        regularization_type = self.training_description["regularization_type"]
        # regularization parameter lambda
        C = self.training_description['regularization_value']
        # number of training sequences
        N = len(train_seqs_id)
         
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")

        # keeps track of the log-likelihood of a sequence before weight updating
        seqs_loglikelihood_vec = numpy.zeros(N)
        seqs_id_mapper = {seq_id:unique_id for unique_id, seq_id in enumerate(train_seqs_id)}
        # step size decides the number of data points to average in the seqs_loglikelihood_vec
        # using 10% of data points
        step_size = round(N * 0.1)
        if step_size == 0:
            step_size = 1
        mean_cost_vec = [0]
        
        p_rho = self.training_description["p_rho"]
        epsilon = self.training_description["epsilon"]
        E_g2 = numpy.zeros(len(w), dtype="longdouble")
        E_deltaw2 = numpy.zeros(len(w), dtype="longdouble")
        if(regularization_type == "l1"):
            u = 0
            q = numpy.zeros(len(w), dtype = "longdouble")
        # gradient
        grad = numpy.zeros(len(w), dtype = "longdouble")      
        # instance variable to keep track of elapsed time between optimization iterations
        self._elapsed_time = datetime.now()
        self._exitloop = False
        for k in range(num_epochs):
            # shuffle sequences at the beginning of each epoch 
            numpy.random.shuffle(train_seqs_id)
            numseqs_left = N
            print("k ",k)
            for seq_id in train_seqs_id:
#                     print(seq_id)
                    
#                 print("first seqs_info[{}]={}".format(seq_id, crf_model.seqs_info[seq_id]))
                seq_loglikelihood = crf_model.compute_seq_loglikelihood(w, seq_id)
                seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood
                target_indx = crf_model.compute_seq_gradient(w, seq_id, grad)                
                if(C):
                    if(regularization_type == 'l2'):
                        seq_loglikelihood += - ((C/N) * (1/2) * numpy.dot(w, w))
                        grad -= ((C/N)* w)
                        
                    elif(regularization_type == 'l1'):
                        seq_loglikelihood += - (C/N) * numpy.sum(numpy.abs(w))
                        
                    # update the computed sequence loglikelihood by adding the regularization term contribution   
                    seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood

                    # accumulate gradient
                    E_g2 = p_rho * E_g2 + (1-p_rho) * numpy.square(grad) 
                    RMS_g = numpy.sqrt(E_g2 + epsilon)
                    RMS_deltaw = numpy.sqrt(E_deltaw2 + epsilon)
                    ratio = (RMS_deltaw/RMS_g)
                    deltaw =  ratio * grad
                    E_deltaw2 = p_rho * E_deltaw2 + (1-p_rho) * numpy.square(deltaw)                    
                    w += deltaw
                    if(regularization_type == "l1"):
                        u += ratio * (C/N)
                        w_upd, q_upd = self._apply_l1_penalty(w, q, u, target_indx)
                        w = w_upd
                        q = q_upd
                else:

                    # accumulate gradient
                    fval = grad[target_indx]
                    E_g2 = p_rho * E_g2
                    E_g2[target_indx] += (1-p_rho) * numpy.square(fval)
                    RMS_g = numpy.sqrt(E_g2 + epsilon)
                    RMS_deltaw = numpy.sqrt(E_deltaw2 + epsilon)
                    ratio = (RMS_deltaw/RMS_g)
                    deltaw = ratio[target_indx] * fval
                    E_deltaw2 = p_rho * E_deltaw2 
                    E_deltaw2[target_indx] += (1-p_rho) * numpy.square(deltaw)                    
                    w[target_indx] += deltaw
                
#                 print("second seqs_info[{}]={}".format(seq_id, crf_model.seqs_info[seq_id]))
                # clean cached info
                crf_model.clear_cached_info([seq_id])
                numseqs_left -= 1
#                 print("third seqs_info[{}]={}".format(seq_id, crf_model.seqs_info[seq_id]))
                # reset the gradient
                grad.fill(0)
                print("num seqs left: {}".format(numseqs_left))
            
            seqs_cost_vec = [numpy.mean(seqs_loglikelihood_vec[i:i+step_size]) for i in range(0, N, step_size)]
            # to consider plotting this vector
            mean_cost_vec.append(numpy.mean(seqs_loglikelihood_vec))
            self._track_sga_optimizer(w, seqs_cost_vec, mean_cost_vec, k)
            if(self._exitloop):
                break
            self._elapsed_time = datetime.now()

            
        line = "---Model training--- end time {} \n".format(datetime.now())
        ReaderWriter.log_progress(line, log_file)
        ReaderWriter.dump_data(mean_cost_vec, os.path.join(model_dir, 'avg_loglikelihood_training'))

        return(w)  
    
    def _sga_classic(self, w, train_seqs_id):
        """implements stochastic gradient ascent
            
           Args:
               w: weight vector (numpy vector)
               train_seqs_id: list of integers representing ids assigned to the sequence
        """
        self._report_training()
        crf_model = self.crf_model
        num_epochs = self.training_description["num_epochs"]
        regularization_type = self.training_description["regularization_type"]
        # regularization parameter lambda
        C = self.training_description['regularization_value']
        # number of training sequences
        N = len(train_seqs_id)
         
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")

        # keeps track of the log-likelihood of a sequence before weight updating
        seqs_loglikelihood_vec = numpy.zeros(N)
        seqs_id_mapper = {seq_id:unique_id for unique_id, seq_id in enumerate(train_seqs_id)}
        # step size decides the number of data points to average in the seqs_loglikelihood_vec
        # using 10% of data points
        step_size = round(N * 0.1)
        if step_size == 0:
            step_size = 1
        mean_cost_vec = [0]
        
        # instance variable to keep track of elapsed time between optimization iterations
        self._elapsed_time = datetime.now()
        self._exitloop = False
        
        if(regularization_type == "l1"):
            u = 0
            q = numpy.zeros(len(w), dtype = "longdouble")
               
        learning_rate_schedule = self.training_description["learning_rate_schedule"]
        t0 = self.training_description["t0"]
        # 0<a<1 -- a parameter should be between 0 and 1 exclusively
        a = self.training_description["a"]
        t = 0
        # gradient
        grad = numpy.zeros(len(w), dtype = "longdouble")
        for k in range(num_epochs):
            # shuffle sequences at the beginning of each epoch 
            numpy.random.shuffle(train_seqs_id)
            numseqs_left = N
            
            for seq_id in train_seqs_id:
                # compute/update learning rate
                if(learning_rate_schedule == "bottu"):
                    eta = C/(t0 + t)
                elif(learning_rate_schedule == "exponential_decay"):
                    eta = t0*a**(t/N)
                elif(learning_rate_schedule == "t_inverse"):
                    eta = t0/(1 + a*(t/N))
                elif(learning_rate_schedule == "constant"):
                    eta = t0
                    
#                 print("eta {}".format(eta))
#                 print(seq_id)
                
                seq_loglikelihood = crf_model.compute_seq_loglikelihood(w, seq_id)
                seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood
                target_index = crf_model.compute_seq_gradient(w, seq_id, grad)
#                 print("seq_grad {}".format(seq_grad))
                if(C):
                    if(regularization_type == 'l2'):
                        seq_loglikelihood += - ((C/N) * (1/2) * numpy.dot(w, w))
                        grad -= ((C/N)* w)
                        w += eta * grad
                        
                    elif(regularization_type == 'l1'):
                        seq_loglikelihood += - (C/N) * numpy.sum(numpy.abs(w))
                        u += eta * (C/N)
                        w_upd, q_upd = self._apply_l1_penalty(w, q, u, target_index)
                        w = w_upd
                        q = q_upd
                        
                    # update the computed sequence loglikelihood by adding the regularization term contribution   
                    seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood

                else:                   
#                     print("fval {}".format(fval)) 
                    w[target_index] += eta * grad[target_index]
                
                t += 1
                # clean cached info
                crf_model.clear_cached_info([seq_id])
                # reset the gradient
                grad.fill(0)
                numseqs_left -= 1
                print("num seqs left: {}".format(numseqs_left))
                
            seqs_cost_vec = [numpy.mean(seqs_loglikelihood_vec[i:i+step_size]) for i in range(0, N, step_size)]
            # to consider plotting this vector
            mean_cost_vec.append(numpy.mean(seqs_loglikelihood_vec))
            self._track_sga_optimizer(w, seqs_cost_vec, mean_cost_vec, k)
            if(self._exitloop):
                break
            self._elapsed_time = datetime.now()

            
        line = "---Model training--- end time {} \n".format(datetime.now())
        ReaderWriter.log_progress(line, log_file)
        ReaderWriter.dump_data(mean_cost_vec, os.path.join(model_dir, 'avg_loglikelihood_training'))
        return(w)

    def _sga_svrg(self, w, train_seqs_id):
        """implements the stochastic variance reduced gradient
        
           The algorithm is reported in  `Johnson R, Zhang T. Accelerating Stochastic Gradient Descent using  Predictive Variance Reduction.
           <https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf>`__
           
           Args:
               w: weight vector (numpy vector)
               train_seqs_id: list of integers representing sequences IDs
           
        """
        # keep the original number of epochs requested
        num_epochs = self.training_description["num_epochs"]
        # run stochastic gradient ascent to initialize the weights
        self.training_description["num_epochs"] = 1
        # current snapshot of w (i.e. w tilda)
        w_tilda_c = self._sga_classic(w, train_seqs_id)
        self.cleanup()

        self.training_description["num_epochs"] = num_epochs
        crf_model = self.crf_model
        regularization_type = self.training_description["regularization_type"]
        # regularization parameter lambda
        C = self.training_description['regularization_value']
        # number of training sequences
        N = len(train_seqs_id)
         
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")

        # keeps track of the log-likelihood of a sequence before weight updating
        seqs_loglikelihood_vec = numpy.zeros(N)
        seqs_id_mapper = {seq_id:unique_id for unique_id, seq_id in enumerate(train_seqs_id)}
        # step size decides the number of data points to average in the seqs_loglikelihood_vec
        # using 10% of data points
        step_size = round(N * 0.1)
        if step_size == 0:
            step_size = 1
        mean_cost_vec = [0]
        
        if(regularization_type == "l1"):
            u = 0
            q = numpy.zeros(len(w), dtype = "longdouble")
               
        eta = self.training_description["t0"]

        m = 2*N
        saved_grad = {}
        # gradient
        grad = numpy.zeros(len(w), dtype = "longdouble")    
        # instance variable to keep track of elapsed time between optimization iterations
        self._elapsed_time = datetime.now()
        self._exitloop = False
        
        for s in range(num_epochs):
            print("stage {}".format(s))
            
            # ###################################
            # compute the average gradient using the snapshot of w (i.e. w tilda)
            mu_grad = numpy.zeros(len(w_tilda_c), dtype = "longdouble")
            # compute average gradient
            seqs_left = N
            for seq_id in train_seqs_id:
                target_indx = crf_model.compute_seq_gradient(w_tilda_c, seq_id, grad)
                fval = grad[target_indx]
                mu_grad[target_indx] += fval
                crf_model.clear_cached_info([seq_id])
                saved_grad[seq_id] = (target_indx, fval)
                # reset grad
                grad.fill(0)
                seqs_left -= 1
                print("average gradient phase: {} seqs left".format(seqs_left))
            mu_grad /= N
            #######################################
                
            w = numpy.copy(w_tilda_c) 
                
            for t in range(m):
                seq_id = numpy.random.choice(train_seqs_id, 1)[0]
                print("round {} out of {}".format(t+1, m))
                
                seq_loglikelihood = crf_model.compute_seq_loglikelihood(w, seq_id)
                seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood
                target_indx = crf_model.compute_seq_gradient(w, seq_id, grad)
                fval = grad[target_indx]
                if(C):
                    if(regularization_type == 'l2'):
                        seq_loglikelihood += - ((C/N) * (1/2) * numpy.dot(w, w))
                        grad -= ((C/N)* w)
                        
                        grad[saved_grad[seq_id][0]] -= saved_grad[seq_id][1] 
                        grad += mu_grad
                        
                        w += eta * grad
                        
                    elif(regularization_type == 'l1'):
                        seq_loglikelihood += - (C/N) * numpy.sum(numpy.abs(w))
                        u += eta * (C/N)
                        grad[saved_grad[seq_id][0]] -= saved_grad[seq_id][1] 
                        grad +=  mu_grad
                        w_upd, q_upd = self._apply_l1_penalty(w, q, u, target_indx)
                        w = w_upd
                        q = q_upd
                        
                    # update the computed sequence loglikelihood by adding the regularization term contribution   
                    seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood

                else:                    
                    w[target_indx] += eta * (fval - saved_grad[seq_id][1])
                    w += eta * mu_grad
                t += 1
                # clean cached info
                crf_model.clear_cached_info([seq_id])
                grad.fill(0)
            w_tilda_c = w
                
            seqs_cost_vec = [numpy.mean(seqs_loglikelihood_vec[i:i+step_size]) for i in range(0, N, step_size)]
            # to consider plotting this vector
            mean_cost_vec.append(numpy.mean(seqs_loglikelihood_vec))
            self._track_sga_optimizer(w, seqs_cost_vec, mean_cost_vec, s)
            if(self._exitloop):
                break
            self._elapsed_time = datetime.now()

            
        line = "---Model training--- end time {} \n".format(datetime.now())
        ReaderWriter.log_progress(line, log_file)
        ReaderWriter.dump_data(mean_cost_vec, os.path.join(model_dir, 'avg_loglikelihood_training'))
        return(w)     
        
    def _apply_l1_penalty(self, w, q, u, w_indx):
        """apply l1 regularization to the weights 
        
           it uses the approach of Tsuruoka et al. Stochastic gradient descent training for L1-regularized log-linear models with cumulative penalty
           
           Args:
               w: weight vector (numpy vector)
               q: total L1 penalty that current weights (corresponding to the features) did receive up to the current time
               u: absolute value of total L1 penalty that each weight could receive up to the current time
               w_indx: weight indices corresponding to the current features under update
               
           TODO: vectorize this function
        """
        for indx in w_indx:
            z = w[indx]
#             print("z is {}".format(z))
#             print("q[indx] is {}".format(q[indx]))
            if(w[indx] > 0):
#                 print("we want the max between 0 and {}".format(w[indx] - (u + q[indx])))
                w[indx] = numpy.max([0, w[indx] - (u + q[indx])])
            elif(w[indx] < 0):
#                 print("we want the min between 0 and {}".format(w[indx] + (u - q[indx])))
                w[indx] = numpy.min([0, w[indx] + (u - q[indx])])
#             print("z is {}".format(z))
#             print("w[indx] is {}".format(w[indx]))
            q[indx] = q[indx] + (w[indx] - z)
        return((w, q))
#             print("q[indx] becomes {}".format(q[indx]))

    def _track_sga_optimizer(self, w, seqs_loglikelihood, mean_loglikelihood, k):
        """track stochastic gradient ascent optimizers by logging each iteration 
        
           Args:
               w: weight vector (numpy vector)
               seqs_loglikelihood: numpy vector representing the average loglikelihood of batches of sequences
               mean_loglikelihood: mean of the seqs_loglikelihood vector
               k: current epoch

        """
        delta_time = datetime.now() - self._elapsed_time 
        self._check_reldiff(mean_loglikelihood[-2], mean_loglikelihood[-1])
        epoch_num = k
        # log file 
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")
        line = "--- Epoch/pass {} --- \n".format(epoch_num+1)
        line += "Estimated training cost (average loglikelihood) is {} \n".format(mean_loglikelihood[-1])
        line += "Number of seconds spent: {} \n".format(delta_time.total_seconds())
        ReaderWriter.log_progress(line, log_file)
    
    def cleanup(self):
        """End of training -- cleanup"""
        # reset iteration counter
        self._iter_count = None
        # reset elapsed time between iterations
        self._elapsed_time = None
        self._exitloop = None


class SeqDecodingEvaluator(object):
    """Evaluator class to evaluate performance of the models
    
       Args:
           model_repr: the CRF model representation that has a suffix of `ModelRepresentation`
                       such as :class:`HOCRFADModelRepresentation`
       
       Attributes:
           model_repr: the CRF model representation that has a suffix of `ModelRepresentation`
                       such as :class:`HOCRFADModelRepresentation`
                       
       .. note::
       
          this class does not support evaluation of segment learning (i.e. notations that include IOB2/BIO notation)
    """
    def __init__(self, model_repr):
        self.model_repr = model_repr
                    
    def compute_states_confmatrix(self, Y_seqs_dict):
        """compute/generate the confusion matrix for each state
           
           Args:
               Y_seqs_dict: dictionary where each sequence has the reference label sequence
                            and its corresponding predicted sequence. It has the following form
                            ``{seq_id:{'Y_ref':[reference_ylabels], 'Y_pred':[predicted_ylabels]}}``
        """
        Y_codebook = self.model_repr.Y_codebook
        M = len(Y_codebook)
        # add another state in case unseen states occur in the test data
        self.model_confusion_matrix = numpy.zeros((M+1, M+1), dtype="float")
        for seq_id in Y_seqs_dict:
            Y_pred = Y_seqs_dict[seq_id]['Y_pred']
            Y_ref = Y_seqs_dict[seq_id]['Y_ref']
            self._compute_model_confusionmatrix(self.map_states_to_num(Y_ref, Y_codebook, M),
                                                self.map_states_to_num(Y_pred, Y_codebook, M)
                                                )
        statelevel_confmatrix  = self._generate_statelevel_confusion_matrix()
        
        return(statelevel_confmatrix)
    
    def _generate_statelevel_confusion_matrix(self):
        model_confusion_matrix = self.model_confusion_matrix
        num_states = model_confusion_matrix.shape[0]
        total = model_confusion_matrix.sum()
        statelevel_confmatrix = numpy.zeros((num_states, 2, 2), dtype='float')
        for i in range(num_states):
            tp = model_confusion_matrix[i, i]
            fp = model_confusion_matrix[i, :].sum() - tp
            fn = model_confusion_matrix[:, i].sum() - tp
            tn = total - (tp+fp+fn)
            statelevel_confmatrix[i] = numpy.array([[tp, fn], [fp, tn]])
        return(statelevel_confmatrix)
    
    def get_performance_metric(self, taglevel_performance, metric, exclude_states=[]):
        """compute the performance of the model using a requested metric
           
           Args:
               taglevel_performance: `numpy` array with Mx2x2 dimension. For every state code a 2x2 confusion matrix
                                     is included. It is computed using :func:`compute_model_performance`
               metric: evaluation metric that could take one of ``{'f1', 'precision', 'recall', 'accuracy'}``
           
           Keyword Arguments:
               exclude_states: list (default empty list) of states to exclude from the computation. Usually, in NER applications the non-entity symbol
                               such as 'O' is excluded from the computation. Example: If ``exclude_states = ['O']``, this will replicate the behavior of `conlleval script <http://www.cnts.ua.ac.be/conll2000/chunking/output.html>`__
        """
        Y_codebook = self.model_repr.Y_codebook
        # do not include 'exclude states' in the computation
        exclude_indices = [Y_codebook[state] for state in exclude_states]
        # total number of states plus 1
        M = len(Y_codebook) + 1
        include_indices = list(set(range(M)) - set(exclude_indices))
        # perform sum across all layers to get micro-average
        collapsed_performance = taglevel_performance[include_indices].sum(axis = 0)
#         print("collapsed performance \n {}".format(collapsed_performance))
        tp = collapsed_performance[0,0]
        fp = collapsed_performance[1,0]
        fn = collapsed_performance[0,1]
        tn = collapsed_performance[1,1]
                    
        perf_measure = 0

        try:
            if(metric == "f1"):
                precision = tp/(tp + fp)
                recall = tp/(tp + fn)
                f1 = (2 * precision * recall)/(precision +  recall)
                print("f1 {}".format(f1))
                perf_measure = f1
            elif(metric == "precision"):
                precision = tp/(tp + fp)
                print("precision {}".format(precision))
                perf_measure = precision
            elif(metric == "recall"):
                recall = tp/(tp + fn)
                print("recall {}".format(recall))
                perf_measure = recall
            elif(metric == "accuracy"):
                accuracy = (tp + tn)/(tp + fp + fn + tn)
                print("accuracy {}".format(accuracy))
                perf_measure = accuracy
        except ZeroDivisionError as e:
            print("dividing by Zero: check/investigate the confusion matrix")
        finally:
            return(perf_measure)
    
    def map_states_to_num(self, Y, Y_codebook, M):
        """map states to their code/number using the `Y_codebook`
           
           Args:
               Y: list representing label sequence 
               Y_codebook: dictionary containing the states as keys and the assigned unique code as values
               M: number of states
               
           .. note:: we give one unique index for tags that did not occur in the training data such as len(Y_codebook)

        """
        Y_coded = [Y_codebook[state] if state in Y_codebook else M for state in Y]
#         print("Y_coded {}".format(Y_coded))
        return(Y_coded)
        


    def _compute_model_confusionmatrix(self, Y_ref, Y_pred):
        """compute confusion matrix on the level of the tag/state
        
           Args:
               Y_ref: list of reference label sequence (represented by the states code)
               Y_pred: list of predicted label sequence (represented by the states code) 
        """
        Y_ref = numpy.asarray(Y_ref)
        Y_pred = numpy.asarray(Y_pred)
        model_confusion_matrix = self.model_confusion_matrix
        for i in range(len(Y_ref)):
            ref_state = Y_ref[i]
            pred_state = Y_pred[i]
            model_confusion_matrix[ref_state, pred_state] += 1
    
class Evaluator(object):
    """Evaluator class to evaluate performance of the models 
    
       Args:
           model_repr: the CRF model representation that has a suffix of `ModelRepresentation`
                       such as :class:`HOCRFADModelRepresentation`
       
       Attributes:
           model_repr: the CRF model representation that has a suffix of `ModelRepresentation`
                       such as :class:`HOCRFADModelRepresentation`
                       
       .. note::
       
          this class is **EXPERIMENTAL/work in progress*** and does not support evaluation of segment learning.
          Use instead :class:`SeqDecodingEvaluator` for evaluating models learned using **sequence** learning.
    """
    def __init__(self, model_repr):
        self.model_repr = model_repr
        
    def transform_codebook(self, Y_codebook, prefixes):
        """map states coded in BIO notation to their original states value
        
           Args:
               Y_codebook: dictionary of states each assigned a unique integer
               prefixes: tuple of prefix notation used such as ("B-","I-") for BIO 
        """
        state_mapper = {}
        for state in Y_codebook:
            if(state != "O"):
                for prefix in prefixes:
                    elems = state.split(prefix)
                    if(len(elems)>1):
                        new_state = elems[-1]
                        state_mapper[state] = new_state
                        break
            else:
                state_mapper[state] = state
        return(state_mapper)
                    
    def compute_model_performance(self, Y_seqs_dict, metric, output_file, states_notation):
        r"""compute the performance of the model
           
           Args:
               Y_seqs_dict: dictionary where each sequence has the reference label sequence
                            and its corresponding predicted sequence. It has the following form
                            ``{seq_id:{'Y_ref':[reference_ylabels], 'Y_pred':[predicted_ylabels]}}``
               metric: evaluation metric that could take one of {'f1', 'precision', 'recall', 'accuracy'}
               output_file: file where to output the evaluation result
               states_notation: notation used to code the state (i.e. BIO)
          
        """
        Y_codebook = self.model_repr.Y_codebook

        if(states_notation == "BIO"):
            prefixes = ("B-", "I-")
            state_mapper = self.transform_codebook(Y_codebook, prefixes)
            transformed_codebook = {}
            counter = 0
            for new_state in state_mapper.values():
                if(new_state not in transformed_codebook):
                    transformed_codebook[new_state] = counter
                    counter += 1
        else:
            state_mapper = {state:state for state in Y_codebook}
            transformed_codebook = Y_codebook
        
        transformed_codebook_rev = {code:state for state, code in transformed_codebook.items()}
        #^print("original Y_codebook ", Y_codebook)
        #^print("state_mapper ", state_mapper)
        #^print("transformed_codebook ", transformed_codebook)
        M = len(transformed_codebook)
        # add another state in case unseen states occur in the test data
        model_taglevel_performance = numpy.zeros((M + 1, 2, 2))

        for seq_id in Y_seqs_dict:
            Y_pred = Y_seqs_dict[seq_id]['Y_pred']
            Y_ref = Y_seqs_dict[seq_id]['Y_ref']
            #^print("Y_pred ", Y_pred)
            #^print("Y_ref ", Y_ref)
            taglevel_performance = self.compute_tags_confusionmatrix(self.map_states_to_num(Y_ref, state_mapper, transformed_codebook, M),
                                                                     self.map_states_to_num(Y_pred, state_mapper,transformed_codebook, M),
                                                                     transformed_codebook_rev,
                                                                     M)
#             print("taglevel_performance {}".format(taglevel_performance))
#             print("tagging performance \n {}".format(taglevel_performance))
            model_taglevel_performance += taglevel_performance
            #^print("model_taglevel_performance ", model_taglevel_performance)

        # perform sum across all layers to get micro-average
        collapsed_performance = model_taglevel_performance.sum(axis = 0)
#         print("collapsed performance \n {}".format(collapsed_performance))
        tp = collapsed_performance[0,0]
        fp = collapsed_performance[0,1]
        fn = collapsed_performance[1,0]
        tn = collapsed_performance[1,1]
        
        perf_measure = 0
        if(metric == "f1"):
            precision = tp/(tp + fp)
            recall = tp/(tp + fn)
            f1 = 2 * ((precision * recall)/(precision +  recall))
            print("f1 {}".format(f1))
            perf_measure = f1
        elif(metric == "precision"):
            precision = tp/(tp + fp)
            print("precision {}".format(precision))
            perf_measure = precision
        elif(metric == "recall"):
            recall = tp/(tp + fn)
            print("recall {}".format(recall))
            perf_measure = recall
        elif(metric == "accuracy"):
            accuracy = (tp + tn)/(tp + fp + fn + tn)
            print("accuracy {}".format(accuracy))
            perf_measure = accuracy
        
        with open(output_file, mode = 'w') as f:
            f.write("The performance of the model based on the {} measure is {}\n".format(metric, perf_measure))
            f.write('Confusion matrix: tp:{} fp:{} fn:{} tn:{}\n'.format(tp, fp, fn, tn))
        return(perf_measure)
        
    def map_states_to_num(self, Y, state_mapper, transformed_codebook, M):
        """map states to their code/number using the `Y_codebook`
           
           Args:
               Y: list representing label sequence 
               state_mapper: mapper between the old and new generated states generated from :func:`tranform_codebook` method
               trasformed_codebook: the transformed codebook of the new identified states
               M: number of states
               
           .. note:: we give one unique index for tags that did not occur in the training data such as len(Y_codebook)

        """
#         Y_coded = []
#         for state in Y:
#             mapped_state = state_mapper[state]
#             if(mapped_state in transformed_codebook):
#                 Y_coded.append(transformed_codebook[mapped_state])
#             else:
#                 Y_coded.append(M)
        Y_coded = [transformed_codebook[state_mapper[state]] if state_mapper.get(state) in transformed_codebook else M for state in Y]
#         print("Y_coded {}".format(Y_coded))
        return(Y_coded)
        
    def compute_tags_confusionmatrix(self, Y_ref, Y_pred, transformed_codebook_rev, M):
        """compute confusion matrix on the level of the tag/state
        
           Args:
               Y_ref: list of reference label sequence (represented by the states code)
               Y_pred: list of predicted label sequence (represented by the states code) 
               transformed_codebook: the transformed codebook of the new identified states
               M: number of states
        """
        #^print("Y_ref coded ", Y_ref)
        #^print("Y_pred coded ", Y_pred)
        detected_statescode = set(Y_ref)
        Y_ref = numpy.asarray(Y_ref)
        Y_pred = numpy.asarray(Y_pred)
#         print("Y_ref as numpy array {}".format(Y_ref))
        tagslevel_performance = numpy.zeros((M + 1, 2, 2))
        
        for statecode in detected_statescode:
            # get all indices of the target tag (gold-standard)
            tag_indx_origin = numpy.where(Y_ref == statecode)[0]
            # get all indices of the target tag (predicted)
            tag_indx_pred = numpy.where(Y_pred == statecode)[0]
            tag_tp = len(numpy.where(numpy.in1d(tag_indx_origin, tag_indx_pred))[0])
            tag_fn = len(tag_indx_origin) - tag_tp
            other_indx_origin = numpy.where(Y_ref != statecode)[0]
            tag_fp = len(numpy.where(numpy.in1d(other_indx_origin, tag_indx_pred))[0])
            tag_tn = len(other_indx_origin) - tag_fp
            tagslevel_performance[statecode] = numpy.array([[tag_tp, tag_fp], [tag_fn, tag_tn]])
            
        return(tagslevel_performance)
    
if __name__ == "__main__":
    pass
    