import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
from fedml_api.model.cv.vgg import vgg11_bn, VGG_SNIP
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.cv.wrn.wrn import WRN
import numpy as np

# init_model = resnet56(10)
# init_model = WRN(28, 4)
init_model = vgg11_bn()
init_model.load_state_dict(torch.load('./debug/0_model.pt'))

# comp_model = resnet56(10)
# comp_model = WRN(28, 4)
comp_model = vgg11_bn()
comp_model.load_state_dict(torch.load('./debug/2_model.pt'))

# print(comp_model)


count = 0
for param_tensor in init_model.state_dict():
    # print(comp_model.state_dict()[param_tensor])p
    # if init_model.state_dict()[param_tensor].type() == 'torch.LongTensor':
    #     print(param_tensor)
    # print(comp_model.state_dict()[param_tensor].type())

    diff_params = init_model.state_dict()[param_tensor] - comp_model.state_dict()[param_tensor]
    diff_params = diff_params.view(-1).numpy()

    # diff_params = init_model.state_dict()[param_tensor].view(-1).numpy()

    # print(diff_params)
    count += sum(np.where(diff_params, 0, 1))
    # print(diff_params)
    # break
print(count)