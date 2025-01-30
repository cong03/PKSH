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
num = 2


list_name = "path"
ls = np.load(list_name)
print(ls)
# exit()

maxbody_target = -1
res_all = []
tmpN_target = []
for _ in ls:

    num = 37
    endnum = int(_[-1])
    x1 = _[0:endnum]
    x2 = _[num:num + endnum]
    x = np.concatenate((x1, x2))
    i = _[2 * num :2 * num + endnum]
    R, I, M, N2, N, MR = getRIM.get_tdcs_lfm(x, i, endnum)
    if N > maxbody_target:
        maxbody_target = N
    res_all.append([1/R, I, M])
    tmpN_target.append(N)

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
num = 37
endnum = int(_[-1])
x1 = _[0:endnum]
x2 = _[num:num + endnum]
x = np.concatenate((x1, x2))
i = _[2 * num:2 * num + endnum]
R, I, M, N2, N, MR = getRIM.get_tdcs_lfm(x, i, endnum)
print("intensity: ", I)
print("ratio: ", R)
print("max intensity: ", M)
print("> 0.2 V/m : ", N2)
print("> 0.2 V/m target : ", N)
print("max avoid / max target : ", MR)

s = ''
for k in range(len(x)):
    s += str(int(x[k])) + ', '
for t in _[2 * num:2 * num + endnum]:
    s += f'{t:.3f}' + ', '
for t in _[2 * num:2 * num + endnum]:
    s += f'{-t:.3f}' + ', '
print(s[:-2])

