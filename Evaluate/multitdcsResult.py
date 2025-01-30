import numpy as np
from public import glo

# glo.position = [-39, 34, 37] # dlpfc
glo.position = [47, -13, 52] # motor
# glo.position = [14,-99,-3] # v1
# glo.position = [-31, -20, -14] # hippo
# glo.position = [10, -19, 6] # thalamus

glo.NUM_ELE = 75
glo.head_model = 'hcpgroup'
glo.six_pos = 'yes'
import getRIM

_ = [] #

num = 3

# _ = [25, 46, 14, 47, 1.0, 1.0] # movea
# _ = [36, 37, 35, 23, 1.881, 0.100] # pksha
# num = 2

R, I, M, N2, N, MR = getRIM.get_tdcs_lfm(_[0:2 * num], _[2 * num:], num)

print(3)
s = ''
for i in range(num * 2):
    s += str(int(_[i])) + ', '
for i in _[2 * num:]:
    s += f'{i:.3f}' + ', '
for i in _[2 * num:]:
    s += f'{-i:.3f}' + ', '
print(s[:-2])
print("intensity: ", I)
print("ratio: ", R)
print("max intensity: ", M)
print("> 0.2 V/m : ", N2)
print("> 0.2 V/m target : ", N)
print("max avoid / max target : ", MR)
getRIM.gettdcs_cenAndtgt(_[0:2 * num], _[2 * num:], num)
