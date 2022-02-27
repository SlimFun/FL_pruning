import json
import numpy as np
from collections import Counter
import matplotlib as plt

ms = []

with open(f'{0}_keep_masks.txt', 'r') as f:
    for j in range(20):
        # if j < 18:
        #     f.readline()
        # else:
        masks_str = f.readline()
        masks_list = json.loads(masks_str)
        # print(masks_list)
        # print(len(masks_list))
        ms.append(np.array(masks_list))

for i in range(1, len(ms)):
    ms[0] += ms[i]

# print(ms[0])
print(Counter(ms[0]))
c = Counter(ms[0])
all_count = sum(c.values())
for k,v in c.items():
    c[k] = float(v) / all_count
print(c)

# a1 = np.array([0, 1, 0, 1])
# a2 = np.array([1, 0, 0, 0])
# a1

# a = [0.6958440920491927, 0.09045508342142759, 0.05723445327940433, 0.04209893039350226, 0.032480310113658005, 0.025331851325353417, 0.019803355570292648, 0.01503313383541586, 0.010976660221012614, 0.0070884411585105615, 0.0036536886322300214]
# plt.bar(range(len(a)), a)
# plt.save('count_result.jpg')