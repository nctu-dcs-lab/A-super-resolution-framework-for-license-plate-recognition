import os.path as osp
import swinfir.archs
import swinfir.data
import swinfir.models
from basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)

# python swinfir/test.py -opt options/test/SwinFIR/SwinFIR_SRx2.yml
# python swinfir/test.py -opt options/test/SwinFIR/testOnPKU_LRnew.yml
