'''
HEVC intra-prediction
--DC mode(Mode 0)
--Planar mode(Mode 1) 
--33 angular modes(Mode 2-34) 
--Compute the nearest 5 neighbors of the hevc prediction for data augmentation
'''
import numpy as np
import cv2
from skimage.measure import compare_ssim
from copy import deepcopy
import random

class Predictor(object):
    def __init__(self,odp = None,block_size = 32,bit_depth=8, filepath = None,diskpath = None,patch=None,multiplier = 5):
        '''
        -----The takes an optional filepath or the ODP as a numpy array and diskpath(for output images)
        -IF ODP segments are 64x64, the block size is set to 32 by default. however YOU must set the block_size when initializing the 
        augmenter class for other ODP sizes.
        -bit_depth is set to 8 by default
        '''
        self.block_size = block_size
        self.bit_depth = bit_depth
        self.odp = odp
        self.filepath = filepath #input path to ODP
        self.diskpath = diskpath #storage path for outputs
        self.input_block = np.array([])
        self.output_patch = np.array([])
        self.left_ref = np.zeros(2*block_size)#left reference array
        self.top_ref = np.zeros(2*block_size +1)#top reference array
        self.context = np.zeros((2*self.block_size,2*self.block_size)) #prediction context
        self.original_pu = np.zeros((self.block_size,self.block_size))# original prediction unit
        self.hevc_pu = np.zeros((self.block_size,self.block_size))
        self.pred_out = np.zeros((self.block_size,self.block_size)) #prediction output
        self.residual = np.zeros((self.block_size,self.block_size))
        self.multiplier = multiplier #number of augmentation units per context(multiplication factor) 
        self.select_candidates = False
        self.compare_original = False #compares with HM encoder result by default- set to True to compare with original PU
        self.mode = 0 
        self.mode_mse = 0
        self.mode_psnr = 0 
        self.candidate_modes = []
        self.ssim_score = 0
        self.list_mse_scores = []
        self.angular_mode_candidates = []
        self.best_predictions = [] #hevc predictions with closest ssim to HM encoder
        self.patch = patch #added just for analytics
    '''
    Data handling methods
    '''
    def read_context(self):
        '''acquire the incoming context and prepare for interpolation'''
        if(type(self.filepath) == str):
            try:#Read from disk using the specified filepath
                self.input_block = cv2.imread(self.filepath, cv2.IMREAD_GRAYSCALE)
            except IOError:
                print('File could not be read: Check file path and storage directory')
            except Exception:
                print('Initialization error/Invalid inputs')
            else:
                _,width = np.shape(self.input_block)
                if(width == self.block_size*6):
                    decoded = self.input_block[0:self.block_size*2,self.block_size*2:self.block_size*4] #extract the middle block of the ODP
                    self.original_pu = self.input_block[self.block_size:self.block_size*2,self.block_size:self.block_size*2]
                    self.hevc_pu  = self.input_block[self.block_size:self.block_size*2,self.block_size*5:self.block_size*6]
                    self.context = decoded
                    self.residual = np.subtract(self.original_pu.astype('float32'), self.hevc_pu.astype('float32'))
                    #reference pixels extraction from the decoded neighborhood blocks{PUs}
                    self.left_ref = decoded[self.block_size:2*self.block_size,self.block_size-1]
                    self.top_ref = decoded[self.block_size-1, self.block_size:2*self.block_size]
                    #add the top left reference- THIS MEANS THE REAL TOP REFERENCE STARTS AT INDEX(1) of the top_ref array
                    self.top_ref = np.insert(self.top_ref,0,decoded[self.block_size-1,self.block_size-1]) 
                else:#if only the decoded block is available in file
                    self.context = self.input_block
                    self.left_ref = self.input_block[self.block_size:2*self.block_size,self.block_size-1]
                    self.top_ref = self.input_block[self.block_size-1, self.block_size:2*self.block_size]
                    self.top_ref = np.insert(self.top_ref,0,input_block[self.block_size-1,self.block_size-1])# top left reference pixel is added to the top reference
                return self.left_ref,self.top_ref,self.context,self.original_pu        
        else:
            try: #Pass the ODP or 64x64 CTU directly instead of a filepath
                self.input_block = deepcopy(self.odp)
                _,width = self.input_block.shape
                if(width == self.block_size*6):
                    decoded = self.input_block[0:2*self.block_size,2*self.block_size:4*self.block_size] #extract the middle block of the ODP
                    self.original_pu = self.input_block[self.block_size:self.block_size*2,self.block_size:self.block_size*2]
                    self.hevc_pu  = self.input_block[self.block_size:self.block_size*2,self.block_size*5:self.block_size*6]
                    self.context = decoded
                    self.residual = np.subtract(self.original_pu.astype('float32'), self.hevc_pu.astype('float32'))
                    self.left_ref = decoded[self.block_size:2*self.block_size,self.block_size-1]
                    self.top_ref = decoded[self.block_size-1, self.block_size:2*self.block_size]
                    self.top_ref = np.insert(self.top_ref,0,decoded[self.block_size-1,self.block_size-1]) 
                else:
                    self.context = self.input_block #the input is considered as a 2Nx2N block
                    self.left_ref = self.input_block[self.block_size:2*self.block_size,self.block_size-1]
                    self.top_ref = self.input_block[self.block_size-1, self.block_size:2*self.block_size]
                    self.top_ref = self.np.insert(self.top_ref,0,input_block[self.block_size-1,self.block_size-1])
                return self.left_ref,self.top_ref,self.context,self.original_pu
            except ValueError as error:
                print("{}! reference set to zeros".format(error))
                return self.left_ref,self.top_ref,self.context,self.original_pu
            except Exception as error:
                print('Invalid inputs: {}'.format(error))
    #fill in the missing samples through interpolation
    def interpolation(self):
        '''extend the length of the reference arrays'''
        self.top_ref = np.insert(self.top_ref,self.block_size+1,[0 for i in range(self.block_size)])
        self.left_ref = np.insert(self.left_ref,self.block_size,[0 for i in range(self.block_size)])
        try:
            if(sum(self.top_ref)==0 and sum(self.left_ref)==0):#fill all the reference values with a nominal average given the bit depth
                self.top_ref[:] = (2**self.bit_depth-1)/2
                self.left_ref[:] = (2**self.bit_depth-1)/2
            if(self.top_ref[0]==0):# when the top left context is missing
               self.top_ref[0] =  self.left_ref[0]         
            if(np.all(self.top_ref[self.block_size+1:2*self.block_size])==0):#when the top right context is missing
                self.top_ref[self.block_size+1:2*self.block_size+1] = self.top_ref[self.block_size]
            if(np.all(self.left_ref[self.block_size:2*self.block_size-1])==0):#when the bottom left context is missing
                self.left_ref[self.block_size:2*self.block_size]= self.left_ref[self.block_size-1]
            return self.left_ref,self.top_ref
        except TypeError as error:
            print('{}: Interpolation not possible'.format(error))
            return self.left_ref,self.top_ref
        except Exception as error:
            print('{}. Check your input variables'.format(error))

    def filter_reference_array(self): 
            '''
                Reference array is filtered if the target PB is >= 32
                filter kernel = [1, 2, 1]/4 # a simple smoothing filter
            '''
            try:
                self.left_ref = np.flip(self.left_ref,0) #flip left reference
                ref_pixels = np.insert(self.top_ref,0,self.left_ref)
                #filter everything else
                for i in range(1,len(ref_pixels)-1):
                    ref_pixels[i] = int((ref_pixels[i-1]+2*ref_pixels[i]+ref_pixels[i+1])/4)
            except Exception as e:
                print('Error during the sample filtering: {}'.format(e))
            else:
                self.left_ref = np.flip(ref_pixels[0:2*self.block_size])
                self.top_ref = ref_pixels[2*self.block_size:]
                return self.left_ref,self.top_ref

    def filter_prediction_output(self,pred_out):
        '''filter the edge pixels for DC, vertical and horizontal predictions'''
        return filtered_pu_output
    '''
    Performance metrics
    '''
    def mse(self):
        if self.compare_original:
            self.mode_mse = np.sum((self.original_pu.astype("float") - self.pred_out.astype("float")) ** 2)/ self.block_size**2
        else:
            self.mode_mse = np.sum((self.original_pu.astype("float") - self.pred_out.astype("float")) ** 2)/ self.block_size**2
        return self.mode_mse
    def psnr(self):
        # NOTE: the two images must have the same dimensions
        self.mse()
        self.mode_psnr = 10*np.log10(255**2/self.mode_mse)
    def compute_ssim(self): 
        '''structural similarity between prediction and original image'''
        try:
            if self.compare_original:
                self.ssim_score = compare_ssim(self.original_pu, self.pred_out.astype('uint8'))
            else:
                self.ssim_score = compare_ssim(self.hevc_pu, self.pred_out.astype('uint8'))
        except TypeError as e:
            print("SSIM Error: ", e)
        except Exception as err:
            print("SSIM Error: ", err)
        
    def decode_pu(self,pred_out):
        '''reconstruct the pu using hevc residuals'''
        decoded = pred_out + self.residual
        self.output_patch = deepcopy(self.input_block)
        if (np.amax(decoded)>255) or (np.amin(decoded)<0):
            max_value = np.amax(decoded)
            min_value = np.amin(decoded)
            decoded = ((decoded-min_value)/(max_value-min_value))*255
        self.output_patch[self.block_size:2*self.block_size,3*self.block_size:4*self.block_size] = decoded.astype('uint8')
        self.output_patch[self.block_size:2*self.block_size,5*self.block_size:6*self.block_size] = pred_out.astype('uint8')
        return self.output_patch

    def selection(self):
        '''select the augmentation modes: DC, Planar and best angular modes'''
        self.mse()
        self.psnr()
        self.compute_ssim()
        try:
            if self.mode== 0 or self.mode==1: #add DC and Planar modes
                self.best_predictions.append(self.output_patch)
            else:
                if len(self.list_mse_scores) < self.multiplier-2:
                    self.list_mse_scores.append(self.mode_mse)
                    self.candidate_modes.append(self.mode)
                    self.angular_mode_candidates.append(self.output_patch)
                else:
                    if self.mode_mse< max(self.list_mse_scores):
                        update_index = self.list_mse_scores.index(max(self.list_mse_scores))
                        self.list_mse_scores[update_index] = self.mode_mse
                        self.candidate_modes[update_index] = self.mode
                        self.angular_mode_candidates[update_index] = self.output_patch
            for pred in self.angular_mode_candidates:
                self.best_predictions.append(pred)
        except TypeError as e:
            print("Selection Error: this", e)
        except Exception as err:
            print("Selection Error: ", err)

        

    def de_blocking(self,decoded):
        '''remove boundary artifacts in dc, vertical and horizontal modes'''
        #filter the edges to remove blocking
        decoded[0:,0] = (decoded[0:,0] + (self.left_ref[0:self.block_size] - self.top_ref[0]))/2
        decoded[0,0:]  = (decoded[0,0:] + (self.top_ref[1:self.block_size+1] - self.top_ref[0]))/2
        return decoded
    '''
    Wrapper methods for prediction
    '''
    #Generate a specific mode
    def predict_one(self,mode = np.random.randint(0,35)):
        '''
        Generates HEVC intra-angular prediction for a specified mode direction
        '''
        self.read_context()
        self.interpolation() 
        try:
            if(self.diskpath):
                prediction,context,mode,ssim = self.prediction(mode) #A mode is chosen at random if you don't specify one
                print('Prediction:: Mode: {} : MSE: {}: PSNR: {} SSIM: {}'.format(self.mode,self.mode_mse,self.mode_psnr,self.ssim_score))
                cv2.imwrite(self.diskpath + '{}_mode{}_pred.jpg'.format(self.patch,mode), context)
            else:
                prediction,context,mode,ssim = self.prediction(mode) #A mode is chosen at random if you don't specify one
        except TypeError:
            print('Invalid inputs: filepath must be a valid path to ODP on disk or np array containing ODP data')
            return None
        except Exception as err:
            print('Error: > {}'.format(err))
        else:
            return self.output_patch
        
        #Generate all modes and save to disk
    def predict_all(self,select = True, compare_original_pu = True):
        '''
        Generates HEVC intra-angular prediction for all the 33 prediction directions, Planar and DC predictions
        in one beautiful loop.
        '''
        self.read_context()
        self.interpolation() 
        all_predictions = []
        self.select_candidates = select
        self.compare_original = compare_original_pu
        try:
            if self.select_candidates:
                for i in range(35):
                    prediction,mode,ssim  =  self.prediction(mode=i)

                choice_pred = random.choices(self.best_predictions, k=1)[0]
                return choice_pred
            else:
                for i in range(35):
                    prediction,output_patch,mode,psnr = self.prediction(mode=i) #A mode is chosen at random if you don't specify one
                    cv2.imwrite(self.diskpath + 'mode{}_pred.jpg'.format(self.mode), output_patch)
                    print('Prediction:: Mode: {} : MSE: {}: PSNR: {} SSIM: {}'.format(self.mode,self.mode_mse,psnr,self.ssim_score))
                    print('...................................')
                    all_predictions.append(prediction)
                return all_predictions
        except TypeError as e:
            print('Error predict_all() : ',e)
        except Exception as err:
            print('Error predict_all() : {}'.format(err))

    def prediction(self,mode=None):
        '''
            This is the actual predictor
        '''
        if(mode !=0 and mode !=1 and self.block_size>4): # do not filter DC and planar and blocks larger than 4x4
            if(mode != 10 and mode!=26): #exclude pure vertical and pure horizontal modes
                self.filter_reference_array()
            '''Add more rules to exclude near-horizontal and near-vertical angular modes when blocks are smaller than 32'''
        elif(mode==1 and self.block_size>32): #filter planar mode if PU_size is > 32x32
            self.filter_reference_array()  
        else:
            pass

        if(mode==0):
            pred_out= self.intra_prediction_dc()
        elif(mode==1):
            pred_out= self.intra_prediction_planar()
        else:
            pred_out = self.intra_prediction_angular(mode)

        self.decode_pu(pred_out)
        
        self.mode = mode
        self.mse()

        if self.select_candidates:
            self.selection()
            return pred_out,mode,self.ssim_score
        else:
            self.psnr()
            self.compute_ssim()
            return pred_out,self.output_patch,mode,self.mode_psnr
    '''
    The actual mode predictors
    '''
    #DC prediction(mode 0)
    def intra_prediction_dc(self): #dc predictions(Mode 0)
        '''
        Generates an DC prediction from the reference context
        '''
        try:
            #mean of the reference samples
            dc_val  = (1/(2*self.block_size))*(sum(self.top_ref[1:self.block_size+1])+sum(self.left_ref[0:self.block_size]))
            self.pred_out[:] = int(dc_val) #cast to integer
        except ValueError:
            print('Invalid values for DC prediction')
        except Exception as err:
            print('Error in the DC prediction: {}'.format(err))
        else:
            return self.pred_out
    
    def intra_prediction_planar(self):#planar prediction(Mode 1)
        '''
        Generates an planar prediction from the reference context
        '''
        #initialize the interpolators
        try:
            h_values = np.zeros((self.block_size,self.block_size))
            v_values = np.zeros((self.block_size,self.block_size))
            # planar_pred = np.zeros((self.block_size,self.block_size))
            for x in range(self.block_size):   #create the vertical projection
                for y in range(self.block_size):
                    v_values[x,y] = (self.block_size-y)*self.left_ref[x] + x*self.left_ref[self.block_size]
            for x in range(self.block_size): #create the horizontal projection
                for y in range(self.block_size):
                    h_values[x,y] = (self.block_size-x)*self.top_ref[y+1] + y*self.top_ref[self.block_size+1]
            #Finally create the planar prediction
            for y in range(self.block_size):
                for x in range(self.block_size):
                    self.pred_out[x,y] = int((v_values[x,y]+h_values[x,y]+self.block_size)/(2**(np.log2(self.block_size)+1))) #cast to integer     
            
        except ValueError:
            print('Invalid values in the reference samples or block size')
        except Exception as err:
            print('Error in the Planar prediction: {}'.format(err))
        else:
            return self.pred_out

    #all 35 angular predictions
    def intra_prediction_angular(self,mode):
        '''
        Generates an angular prediction from a given context
        '''
        #select the main reference depending on the mode
        mode_displacement = [32,26,21,17,13,9,5,2]
        mode_displacement_inv = [2,5,9,13,17,21,26,32] 
        main_reference = []
        # pred_angular = np.zeros((self.block_size,self.block_size)) #prediction output
        if(mode >= 2 and mode < 18):#left context is the main reference for these modes
            main_reference = self.left_ref
            main_reference_ext = [] #extension used for modes with negative displacement
            positive_modes = [mode for mode in range(2,10)]
            negative_modes = [mode for mode in range(11,18)]#handle with inverse angles when extedinding the reference samples
            #set the mode displacement
            displacement = 0
            if(mode in positive_modes and mode != 10):
                displacement = mode_displacement[positive_modes.index(mode)]
                #predictions for modes with positive displacement
                for x in range(self.block_size):
                    for y in range(self.block_size):
                        #calculate the pixel projection on to the reference array
                        c = (y*displacement)>>5  #offset
                        w = (y*displacement) and 31 #weighting factor
                        i = x + c #index of the reference pixel
                        #estimate the pixel value as the weighted sum of pixel at position i and i+1
                        self.pred_out[x,y] = int(((32-w)*main_reference[i] + w*main_reference[i+1])/32)

            elif(mode==10 ):
                for i in range(self.block_size):
                    for j in range(self.block_size):
                        self.pred_out[i,j] = main_reference[i] #pure horizontal prediction
            else:
                displacement = -(mode_displacement_inv[negative_modes.index(mode)])
                inv_angle = (256*32)/displacement #compute an equivalent of the negative angle
                #extend the main reference according to the negative prediction directions
                for i in range(1,self.block_size):
                    index = -1+(int(-i*inv_angle+128)>>8)
                    if(index<=self.block_size-1):
                        main_reference_ext.append(self.top_ref[index])
                #create a new reference array with extension for negative angles
                extension_len = len(main_reference_ext)
                #insert the top left context to the left ref array
                main_reference = np.insert(main_reference,0,self.top_ref[0])
                for val in main_reference_ext:
                    main_reference = np.insert(main_reference,0,val)
                #prediction for modes with negative displacement
                for x in range(self.block_size):
                    for y in range(self.block_size):
                        c = (y*displacement)>>5
                        w = (y*displacement) and 31
                        i = x + c
                        #if i is negative use the extended reference array 
                        self.pred_out[x,y] = int(((32-w)*main_reference[i+1+extension_len] + w*main_reference[i+2+extension_len]+16)/32)
        else:#top reference is used otherwise
            main_reference = self.top_ref
            main_reference_ext = []
            positive_modes = [mode for mode in range(26,35)]
            negative_modes = [mode for mode in range(18,26)]
            if(mode in positive_modes and mode != 26):
                displacement = mode_displacement_inv[positive_modes.index(mode)-1]
                for x in range(self.block_size):
                    for y in range(self.block_size):
                        c = (y*displacement)>>5
                        w = (y*displacement) and 31
                        i = x + c
                        self.pred_out[x,y] = int(((32-w)*main_reference[i+1] + w*main_reference[i+2]+16)/32)
            elif(mode==26): #pure vertical prediction
                for i in range(self.block_size):
                    for j in range(self.block_size):
                        self.pred_out[i,j] = main_reference[j+1]
            else:
                displacement = -(mode_displacement[negative_modes.index(mode)])
                inv_angle = (256*32)/displacement
                #extend the main reference according to the negative prediction directions
                for i in range(1,self.block_size):
                    index = -1+(int(-i*inv_angle+128)>>8)
                    if(index<=self.block_size-1):
                        main_reference_ext.append(self.left_ref[index])
                #create a new reference array with extension for negative angles
                extension_len = len(main_reference_ext)
                for val in main_reference_ext:
                    main_reference = np.insert(main_reference,0,val)
                for x in range(self.block_size):
                    for y in range(self.block_size):
                        c = (y*displacement)>>5
                        w = (y*displacement) and 31
                        i = x + c 
                        self.pred_out[x,y] = int(((32-w)*main_reference[i+1+extension_len] + w*main_reference[i+2+extension_len]+16)/32)
        return self.pred_out

 
        '''
        Generates an angular prediction from a given context
        '''
        #select the main reference depending on the mode
        mode_displacement = [32,26,21,17,13,9,5,2]
        mode_displacement_inv = [2,5,9,13,17,21,26,32] 
        main_reference = []
        # pred_angular = np.zeros((self.block_size,self.block_size)) #prediction output
        if(mode >= 2 and mode < 18):#left context is the main reference for these modes
            main_reference = self.left_ref
            main_reference_ext = [] #extension used for modes with negative displacement
            positive_modes = [mode for mode in range(2,10)]
            negative_modes = [mode for mode in range(11,18)]#handle with inverse angles when extedinding the reference samples
            #set the mode displacement
            displacement = 0
            if(mode in positive_modes and mode != 10):
                displacement = mode_displacement[positive_modes.index(mode)]
                #predictions for modes with positive displacement
                for x in range(self.block_size):
                    for y in range(self.block_size):
                        #calculate the pixel projection on to the reference array
                        c = (y*displacement)>>5  #offset
                        w = (y*displacement) and 31 #weighting factor
                        i = x + c #index of the reference pixel
                        #estimate the pixel value as the weighted sum of pixel at position i and i+1
                        self.pred_out[x,y] = int(((32-w)*main_reference[i] + w*main_reference[i+1])/32)

            elif(mode==10 ):
                for i in range(self.block_size):
                    for j in range(self.block_size):
                        self.pred_out[i,j] = int(main_reference[i]) #pure horizontal prediction
            else:
                displacement = -(mode_displacement_inv[negative_modes.index(mode)])
                inv_angle = (256*32)/displacement #compute an equivalent of the negative angle
                #extend the main reference according to the negative prediction directions
                for i in range(1,self.block_size):
                    index = -1+(int(-i*inv_angle+128)>>8)
                    if(index<=self.block_size-1):
                        main_reference_ext.append(self.top_ref[index])
                #create a new reference array with extension for negative angles
                extension_len = len(main_reference_ext)
                #insert the top left context to the left ref array
                main_reference = np.insert(main_reference,0,self.top_ref[0])
                for val in main_reference_ext:
                    main_reference = np.insert(main_reference,0,val)
                #prediction for modes with negative displacement
                for x in range(self.block_size):
                    for y in range(self.block_size):
                        c = (y*displacement)>>5
                        w = (y*displacement) and 31
                        i = x + c
                        #if i is negative use the extended reference array 
                        self.pred_out[x,y] = int(((32-w)*main_reference[i+1+extension_len] + w*main_reference[i+2+extension_len]+16)/32)
        else:#top reference is used otherwise
            main_reference = self.top_ref
            main_reference_ext = []
            positive_modes = [mode for mode in range(26,35)]
            negative_modes = [mode for mode in range(18,26)]
            if(mode in positive_modes and mode != 26):
                displacement = mode_displacement_inv[positive_modes.index(mode)-1]
                for x in range(self.block_size):
                    for y in range(self.block_size):
                        c = (y*displacement)>>5
                        w = (y*displacement) and 31
                        i = x + c
                        self.pred_out[x,y] = int(((32-w)*main_reference[i+1] + w*main_reference[i+2]+16)/32)
            elif(mode==26): #pure vertical prediction
                for i in range(self.block_size):
                    for j in range(self.block_size):
                        self.pred_out[i,j] = main_reference[j+1].astype('uint8')
            else:
                displacement = -(mode_displacement[negative_modes.index(mode)])
                inv_angle = (256*32)/displacement
                #extend the main reference according to the negative prediction directions
                for i in range(1,self.block_size):
                    index = -1+(int(-i*inv_angle+128)>>8)
                    if(index<=self.block_size-1):
                        main_reference_ext.append(self.left_ref[index])
                #create a new reference array with extension for negative angles
                extension_len = len(main_reference_ext)
                for val in main_reference_ext:
                    main_reference = np.insert(main_reference,0,val)
                for x in range(self.block_size):
                    for y in range(self.block_size):
                        c = (y*displacement)>>5
                        w = (y*displacement) and 31
                        i = x + c 
                        self.pred_out[x,y] = int(((32-w)*main_reference[i+1+extension_len] + w*main_reference[i+2+extension_len]+16)/32)
        return self.pred_out