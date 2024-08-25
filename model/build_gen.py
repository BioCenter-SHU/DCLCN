from model import attention3d
from model import attention3d_devide
def Generator(pixelda=False):
    return attention3d.Feature()

def FC():
    return attention3d.FC_1()

def Generator1():
    return attention3d_devide.devide_1(block_num=1)

def Generator2():
    return attention3d_devide.devide_2(block_num=1)

def Generator3():
    return attention3d_devide.devide_3(block_num=1)

def Classifier():
    return attention3d.Predictor()