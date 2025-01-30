import numpy as np
from public import glo


# glo.position = [-39, 34, 37] # dlpfc
# glo.position = [47, -13, 52] # motor
# glo.position = [14,-99,-3] # v1
# glo.position = [-31, -20, -14] # hippo
glo.position = [10, -19, 6] # thalamus

glo.NUM_ELE = 75
glo.head_model = 'hcpgroup'
glo.six_pos = 'yes'
# glo.six_pos = 'no'
import getRIM
import pickle
num = 2

list_name = "path"

ls = np.load(list_name)
print(ls)
# exit()

maxbody_target = -1
res_all = []
tmpN_target = []
for _ in ls:

    R, I, M, N2, N, MR = getRIM.get_tdcs_lfm(_[0:2 * num], _[2 * num:], num)
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
res = np.array(res)


res_normed = res
res = np.sum(res_normed, axis=0).tolist()

sorted_id = sorted(range(len(res)), key=lambda k: res[k], reverse=True)
print("Best Set: ", res_ls[sorted_id[0]].tolist())
_ = res_ls[sorted_id[0]]
R, I, M, N2, N, MR = getRIM.get_tdcs_lfm(_[0:2 * num], _[2 * num:], num)
print("intensity: ", I)
print("ratio: ", R)
print("max intensity: ", M)
print("> 0.2 V/m : ", N2)
print("> 0.2 V/m target : ", N)
print("max avoid / max target : ", MR)

s = ''
for i in range(num * 2):
    s += str(int(_[i])) + ', '
for i in _[2 * num:]:
    s += f'{i:.3f}' + ', '
for i in _[2 * num:]:
    s += f'{-i:.3f}' + ', '
print(s[:-2])
