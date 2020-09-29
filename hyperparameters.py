"""
Parameter file for specifying the running parameters for forward model
"""
# Model Architectural Parameters
USE_LORENTZ = True
NUM_LORENTZ_OSC = 2
USE_CONV = False                         # Whether use upconv layer when not using lorentz @Omar
LINEAR = [2*NUM_LORENTZ_OSC, 50, 50]
# If the Lorentzian is False
CONV_OUT_CHANNEL = [4, 4, 4]
CONV_KERNEL_SIZE = [8, 5, 5]
CONV_STRIDE = [2, 1, 1]

# Optimization parameters
OPTIM = "Adam"
REG_SCALE = 1e-3
BATCH_SIZE = 2048
EVAL_STEP = 20
RECORD_STEP = 20
TRAIN_STEP = 10000
LEARN_RATE = 5e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 0.0005
USE_CLIP = False
GRAD_CLIP = 1
USE_WARM_RESTART = False
LR_WARM_RESTART = 600
ERR_EXP = 2
DELTA = 0
GRADIENT_ASCEND_STRENGTH = 0.1
OPTIMIZE_W0_RATIO = 0

# Data Specific parameters
X_RANGE = [i for i in range(0, 2*NUM_LORENTZ_OSC )]
Y_RANGE = [i for i in range(2*NUM_LORENTZ_OSC , 300+2*NUM_LORENTZ_OSC )]
FREQ_LOW = 0.5
FREQ_HIGH = 5
NUM_SPEC_POINTS = 300
FORCE_RUN = True
DATA_DIR = ''                # For local usage
# DATA_DIR = '/work/sr365/Omar_data'
# DATA_DIR = 'C:/Users/labuser/mlmOK_Pytorch/'                # For Omar office desktop usage
# DATA_DIR = '/home/omar/PycharmProjects/mlmOK_Pytorch/'  # For Omar laptop usage
GEOBOUNDARY =[20, 200, 20, 100]
NORMALIZE_INPUT = True
TEST_RATIO = 0.2
LOR_RATIO = 0.1
LOR_WEIGHT = 10
GT_MATCH_STYLE = 'gt'
TRAIN_LOR_STEP = 50

# Running specific
USE_CPU_ONLY = False
MODEL_NAME = "Gradient_Ascend"
EVAL_MODEL =  "trail_1_gradient_ascend_strength_0.005"
PRE_TRAIN_MODEL = "Gradient_Ascend"
NUM_PLOT_COMPARE = 5
