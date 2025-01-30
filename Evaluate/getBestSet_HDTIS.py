import numpy as np
from public import glo

# glo.position = [-39, 34, 37] # dlpfc
# glo.position = [47, -13, 52] # motor
glo.position = [14,-99,-3] # v1
# glo.position = [-31, -20, -14] # hippo
# glo.position = [10, -19, 6] # thalamus

glo.NUM_ELE = 75
glo.head_model = 'hcpgroup'
# glo.six_pos = 'yes'
glo.six_pos = 'no'
import getRIM
# 电极个数
num = 2
# 解集合

import pickle

list_name = "path"

ls = []
with open(list_name, 'rb') as f:
    ls = pickle.load(f)
print(ls)
# exit()

maxbody_target = -1
res_all = []
tmpN_target = []
for _ in ls:

    num = 37
    endnum = int(_[-1])
    print(endnum)
    R, I, M, N2, N, MR = getRIM.get_mti4_lfm(_[0:endnum], _[num:num + endnum],_[2 * num :2 * num + endnum], endnum)
    if N > maxbody_target:
        maxbody_target = N
    res_all.append([1/R, I, M])
    tmpN_target.append(N)
    # print(_)
    # print([R, I, M])
i = 0
res = []
res_ls = []
for _1, _2 in zip(res_all, ls):
    if tmpN_target[i] >= maxbody_target/2:
        res.append(_1)
        res_ls.append(_2)
    i+=1
res1 = np.array(res)

res1_normed = res1 / res1.max(axis=0)
res1 = res1_normed.sum(axis=1).tolist()


def layer_norm_columns(x, epsilon=1e-6):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    normalized = (x - mean) / (std + epsilon)

    return normalized


res = np.array(res)

res_normed = layer_norm_columns(res)

res = np.sum(res_normed, axis=0).tolist()

# 获得降序排序的idx
sorted_id = sorted(range(len(res)), key=lambda k: res[k], reverse=True)
print("Best Set: ", res_ls[sorted_id[0]])
_ = res_ls[sorted_id[0]]
num = 37
endnum = int(_[-1])
R, I, M, N2, N, MR = getRIM.get_mti4_lfm(_[0:endnum], _[num:num + endnum], _[2 * num:2 * num + endnum], endnum)
print("intensity: ", I)
print("ratio: ", R)
print("max intensity: ", M)
print("> 0.2 V/m : ", N2)
print("> 0.2 V/m target : ", N)
print("max avoid / max target : ", MR)
_2 = _[:-num].copy()
for idx in range(num):
    _2.append(_[idx-num])
    _2.append(-_[idx-num])
s = ''
for i, j in zip(_[0:endnum], _[num:num + endnum]):
    s += str(int(i)) + ', '
    s += str(int(j)) + ', '
for i in _[2 * num:2 * num + endnum]:
    s += f'{i:.3f}' + ', '
    s += f'{-i:.3f}' + ', '
print("First value：", s[:-2])
