from Model.model import MNB_Model
from Model.preprocessing import *
from Model.fv_main import fv_run
from Model.ov_main import ov_run

if __name__ == '__main__':
    ov_run()
    print('======================================================================')
    fv_run()

