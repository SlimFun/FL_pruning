# from fedml_api.model.cv.vgg import VGG_SNIP
# from fedml_api.model.SNIP.snip import SNIP


# model = VGG_SNIP('D').to('cuda:0')
# keep_masks = SNIP(model, 0.95, train_data, device)

def factorization(num):
    factor = []
    while num > 1:
        for i in range(num - 1):
            k = i + 2
            if num % k == 0:
                factor.append(k)
                num = int(num/k)
                break
    return factor