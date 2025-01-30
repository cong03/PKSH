import numpy as np
from public import glo

# glo.position = [-39, 34, 37] # dlpfc
# glo.position = [47, -13, 52] # motor
glo.position = [14,-99,-3] # v1
# glo.position = [-31, -20, -14] # hippo
# glo.position = [10, -19, 6] # thalamus

glo.NUM_ELE = 75
glo.head_model = 'hcpgroup'
glo.six_pos = 'no'
import getRIM

# _ = [70, 15, 71, 11, 1.022, 0.965] # PKSHA
# _ = [26, 46, 35, 27, 0.988, 1.012] # GA
_ = [51, 67, 71, 72, 1.126, 0.873] # enum
R, I, M, MM = getRIM.get_ratio_intensity1(_[0], _[1], _[2], _[3], 0, _[4], _[5])
s = ''
for i in range(4):
    s += str(int(_[i])) + ', '

for i in range(5, len(_), 1):
    s += f'{_[i]:.3f}' + ', '
    s += f'{-_[i]:.3f}' + ', '
print(1)
print(s[:-2])
print("intensity: ", I)
print("ratio: ", R)
print("max intensity: ", M)
print("max intensity / max: ", MM)
getRIM.getti_cenAndtgt(_[0], _[1], _[2], _[3], 0, _[4], _[5])
print('over!')