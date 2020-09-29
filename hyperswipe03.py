"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flagreader
if __name__ == '__main__':
    #linear_unit_list = [ 50, 100, 200, 500]
    #linear_unit_list = [1000, 500]
    #conv_kernel_size_first_list = [2, 4, 8, 16, 32]
    #conv_kernel_size_second_list = [3, 5, 9, 15, 33]
    #linear_unit_list = [30, 50, 100]
    # reg_scale_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    #reg_scale_list = [5e-4, 1e-3, 2e-4, 5e-3, 1e-2]
    #reg_scale_list = [5e-4, 1e-3]
    #for linear_unit in linear_unit_list:
    #    for layer_num in range(3,5):
    #for kernel_first in conv_kernel_size_first_list:
    #    for kernel_second in conv_kernel_size_second_list:
    
    #weight_list = [1, 2, 4, 8, 10, 16, 20, 25, 50] #Full
    #weight_list = [14, 18, 22, 26, 30]  #Subset
    #for ratio in np.arange(0.1, 0.6, 0.1):
    GA_list = [0.02, 0.01]
    for gradient_ascend_strength in GA_list:
    #for ratio in np.arange(0.01, 0.1, 0.01):
        #for weight in weight_list:
            # Setting the loop for setting the parameter
            #for i in range(3, 10):
        flags = flagreader.read_flag()  	#setting the base case
        flags.gradient_ascend_strength = gradient_ascend_strength
            ############
            # FC layer #
            ############
            #linear = [linear_unit for j in range(layer_num)]        #Set the linear units
            #linear[0] = 8                   # The start of linear
            #linear[-1] = 1                # The end of linear
            #flags.linear = linear

            #####################
            # Convolution layer #
            #####################
            #flags.conv_kernel_size[0] = kernel_first
            #flags.conv_kernel_size[1] = kernel_second
            #flags.conv_kernel_size[2] = kernel_second

            #######
            # Reg #
            #######
            #for reg_scale in reg_scale_list:
            #    flags.reg_scale = reg_scale
            
            ############################
            # Lorentz ratio and weight #
            ############################
            #flags.lor_ratio = ratio
            #flags.lor_weight = weight
        for j in range(2):
                #flags.model_name ="reg"+ str(flags.reg_scale) + "trail_"+str(j) + "linear_num" + str(layer_num) + "_unit_layer" + str(linear_unit)
            flags.model_name = 'trail_' + str(j) + '_gradient_ascend_strength_' + str(gradient_ascend_strength)
                        
                        
                #flags.model_name ="reg"+ str(flags.reg_scale) + "trail_"+str(j) + "_conv_kernel_swipe[" + str(kernel_first)+ "," + str(kernel_second) + "," + str(kernel_second) + "]"
            train.training_from_flag(flags)

