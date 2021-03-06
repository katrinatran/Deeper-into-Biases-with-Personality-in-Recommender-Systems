SEEDS = [
    6547893,
    2034976,
    2345303,
]

UN_SEEDS = [
    6547893,
    2034976,
    2345303,
    7887871,
    1023468,
    8812394,
    2132395,
    4444637,
    7192837,
    6574836,
]

TRAITS = [
    'ope',
    'con',
    'agr',
    'ext',
    'neu'
]

LEVELS = [
    1,
    3,
    5,
    10,
    20,
    50,
]

VAE_SEED = 101315
VAE_MAX_EPOCHS = 100

# Controlled experiments
# Structure is res/{ algorithm name }/{ date and time }/{ trait }/{ seed }
LOG_VAL_STR = r'C:\Users\ktran\cs274_project\prototype\pers_bias\res\con\{}\{}\val\{}\{}'
LOG_TE_STR = r'C:\Users\ktran\cs274_project\prototype\pers_bias\res\con\{}\{}\test\{}\{}'
# Uncontrolled experiments
# Structure is res/un/{ algorithm name }/{ date and time }/{ seed }
UN_LOG_VAL_STR = r'C:\Users\ktran\cs274_project\prototype\pers_bias\res\un\{}\{}\val\{}'
UN_LOG_TE_STR = r'C:\Users\ktran\cs274_project\prototype\pers_bias\res\un\{}\{}\test\{}'

DATA_PATH = '../../data/inter.csv'
PERS_PATH = '../../data/pers.csv'
OUT_DIR = '../../data/seed_trait/'
UN_OUT_DIR = '../../data/seed/'

