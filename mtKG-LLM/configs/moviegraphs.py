from yacs.config import CfgNode as CN

data_cfg = CN()

data_cfg.DATA = CN()

data_cfg.DATA.SPLIT_FILE_PATH = ''
data_cfg.DATA.NO_MOVIE_OVERLAP_SPLIT_FILE_PATH = ''
data_cfg.DATA.LABEL_DIR = ''
data_cfg.DATA.FACE_DIR = ''
data_cfg.DATA.SUBTITLE_DIR = ''
data_cfg.DATA.FRAMES_DIR = ''
data_cfg.DATA.PRE_PROCESSED_PATH = ''
data_cfg.DATA.PRE_PROCESSED_BACKUP_PATH = ''
data_cfg.DATA.PRE_LOADED_PATH = ''
data_cfg.DATA.PREDICTED_PATH = ''
data_cfg.DATA.PREDICTED_BACKUP_PATH = ''

data_cfg.DATA.TEMP_EXCLUDED = ['']


