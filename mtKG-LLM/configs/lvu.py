from yacs.config import CfgNode as CN

data_cfg = CN()

data_cfg.DATA = CN()

data_cfg.DATA.SPLIT_PATH = ''
data_cfg.DATA.LABEL_DIR = ''

data_cfg.DATA.PRE_PROCESSED_PATH = ''
data_cfg.DATA.PRE_PROCESSED_BACKUP_PATH = ''