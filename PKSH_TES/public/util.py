import numpy as np
import math
from public import glo

# wyh

inf = 9999999999
NUM_ELE = glo.NUM_ELE

POINT_NUM = 1000



print("glo: ",glo,"head_model: ",glo.head_model)
if glo.head_model == 'hcp4':
    print("loading hcp4")
    # grey and white matter
    lfm = np.load(r'./data/lfm_hcp4_20.npy')
    print("load grey and white matter: ",lfm.shape)
    pos = np.load(r'./data/pos_hcp4_20.npy')
    print("load position: ", pos.shape)

if glo.head_model == 'hcpgroup':
    print("loading hcpgroup")
    # grey and white matter
    lfm = np.load(r'./data/lfm_hcpgroup.npy')
    print("load grey and white matter: ",lfm.shape)
    pos = np.load(r'./data/pos_hcpgroup.npy')
    print("load position: ", pos.shape)

if glo.head_model == 'ernie':
    print("loading ernie")
    # grey and white matter
    lfm = np.load(r'./data/lfm_ernie.npy')
    print("load grey and white matter: ",lfm.shape)
    pos = np.load(r'./data/pos_ernie.npy')
    print("load position: ", pos.shape)

#for avoidance 
# position =[-39, 34, 37] # dlpfc
# distance = np.zeros(len(pos))
# print(position)

# for i in range(len(pos)):
#     distance[i] = (pos[i, 0] - position[0])**2 + (pos[i, 1] - position[1])**2 + (pos[i, 2] - position[2])**2
# AVOID_POSITION = np.where(distance < 10**2)
# AVOID_POSITION = AVOID_POSITION[0]
# print(len(AVOID_POSITION))

position =  glo.position 
distance = np.zeros(len(pos))
print("roi_positon: ", position)

for i in range(len(pos)):
    distance[i] = (pos[i, 0] - position[0])**2 + (pos[i, 1] - position[1])**2 + (pos[i, 2] - position[2])**2

print('min_distance:' + str(min(distance)))
TARGET_POSITION = np.where(distance <= 10**2) #roi size (CM^2)
TARGET_POSITION = TARGET_POSITION[0]
AVOID_POSITION = np.where(distance > 10**2)
AVOID_POSITION = AVOID_POSITION[0]
if glo.six_pos == 'yes':
    positions = [
        np.array([-39, 34, 37]),
        np.array([47, -13, 52]),
        np.array([14,-99,-3]),
        np.array([-31, -20, -14]),
        np.array([10, -19, 6])
    ]

    for position in positions:
        distance = np.zeros(len(pos))
        for i in range(len(pos)):
            distance[i] = (pos[i, 0] - position[0]) ** 2 + (pos[i, 1] - position[1]) ** 2 + (
                        pos[i, 2] - position[2]) ** 2

        print("Position:", position)
        print('min_distance:', min(distance))

        tem_TARGET_POSITION = np.where(distance <= 10 ** 2)[0]
        tem_AVOID_POSITION = np.where(distance > 10 ** 2)[0]

        print('tem TARGET_POSITION:', len(tem_TARGET_POSITION))
        print('tem AVOID_POSITION:', len(tem_AVOID_POSITION))

        combined_targets = np.union1d(tem_TARGET_POSITION, TARGET_POSITION)
        combined_avoids = np.union1d(tem_AVOID_POSITION, AVOID_POSITION)

        print('Combined Targets:', len(combined_targets))
        print('Combined Avoids:', len(combined_avoids))

        TARGET_POSITION = combined_targets
        AVOID_POSITION = combined_avoids
        AVOID_POSITION = np.setdiff1d(AVOID_POSITION, TARGET_POSITION)

print('volume in all:' + str(len(pos)))
print('volume in avoid:' + str(len(AVOID_POSITION)))
print('volume in roi:' + str(len(TARGET_POSITION)))


def tdcs_I(x, i, num):
    stimulation1 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num)):
        stimulation1[int(x[k])] = i[k]
    for k in range(int(num)):
        stimulation1[int(x[num + k])] = -i[k]
    # print(stimulation1)
    e = ((np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)) ** 2) ** 0.5 / 1000

    e_avg = np.mean(abs(e))
    return 1 / e_avg


def tdcs_R(x, i, num):
    stimulation1 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num)):
        stimulation1[int(x[k])] = i[k]
    for k in range(int(num)):
        stimulation1[int(x[num + k])] = -i[k]

    e1 = ((np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)) ** 2) ** 0.5 / 1000

    e2 = ((np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)) ** 2) ** 0.5 / 1000

    e_ratio = np.mean(abs(e2)) / np.mean(abs(e1))
    return e_ratio

def tdcs_M(x, i, num):
    stimulation1 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num)):
        stimulation1[int(x[k])] = i[k]
    for k in range(int(num)):
        stimulation1[int(x[num + k])] = -i[k]

    e = ((np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)) ** 2) ** 0.5 / 1000

    e_max = np.max(abs(e))
    return 1 / e_max

def tdcs2_two(x, i, num):
    stimulation1 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num)):
        stimulation1[int(x[k])] = i[k]
    for k in range(int(num)):
        stimulation1[int(x[num + k])] = -i[k]

    e1 = ((np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1)) ** 2 +
          (np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1)) ** 2 +
          (np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)) ** 2) ** 0.5 / 1000

    e2 = ((np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1)) ** 2 +
          (np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1)) ** 2 +
          (np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)) ** 2) ** 0.5 / 1000

    e1_avg = np.mean(abs(e1))
    e2_avg = np.mean(abs(e2))
    e_ratio = e2_avg / e1_avg
    evg_target = np.mean(abs(e1))

    return 1 / evg_target, e_ratio

def tdcs2_three(x, i, num):
    stimulation1 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num)):
        stimulation1[int(x[k])] = i[k]
    for k in range(int(num)):
        stimulation1[int(x[num + k])] = -i[k]

    e1 = ((np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1)) ** 2 +
          (np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1)) ** 2 +
          (np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)) ** 2) ** 0.5 / 1000

    e2 = ((np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1)) ** 2 +
          (np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1)) ** 2 +
          (np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)) ** 2) ** 0.5 / 1000

    e1_avg = np.mean(abs(e1))
    e2_avg = np.mean(abs(e2))
    e_ratio = e2_avg / e1_avg
    evg_target = np.mean(abs(e1))

    return 1 / evg_target, e_ratio, 1 / np.max(abs(e1))

def tdcs2_four(x, i, num):
    stimulation1 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num)):
        stimulation1[int(x[k])] = i[k]
    for k in range(int(num)):
        stimulation1[int(x[num + k])] = -i[k]

    e1 = ((np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)) ** 2) ** 0.5 / 1000

    e2 = ((np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1)) ** 2 +
         (np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)) ** 2) ** 0.5 / 1000

    e1_avg = np.mean(abs(e1))
    e1_max = np.max(abs(e1))
    e2_avg = np.mean(abs(e2))
    e2_max = np.max(abs(e2))
    e_ratio = e2_avg / e1_avg
    em_ratio = e2_max / e1_max
    evg_target = np.mean(abs(e1))

    return 1 / evg_target, e_ratio, 1 / e1_max, em_ratio

def twotdcs_enum_I(x1, x2, x3, x4,x5):
    maxeam = 0
    i_ = 0
    for i in np.arange(0.1, 2, 0.05):
        if float(i) < x5/NUM_ELE:
            continue
        electrode1 = x1
        electrode2 = x2
        electrode3 = x3
        electrode4 = x4
        stimulation1 = np.zeros(NUM_ELE)
        stimulation1[electrode1] = float(i) + x5/NUM_ELE
        stimulation1[electrode2] = -float(i) - x5/NUM_ELE
        stimulation1[electrode3] = float(2.0 - i) - x5/NUM_ELE
        stimulation1[electrode4] = -float(2.0 - i) + x5/NUM_ELE
        e = ((np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1)) ** 2 +
              (np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1)) ** 2 +
              (np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)) ** 2) ** 0.5 / 1000

        if maxeam < np.mean(abs(e)):
            maxeam = np.mean(abs(e))
            i_ = i

    if 1/maxeam < glo.GA_enum_Best_ti_intensity:
        glo.GA_enum_Best_ti_intensity = 1/maxeam
        glo.GA_enum_Best_ti_I = np.array([i_, -i_, 2.0-i_, i_-2.0])
        # print(glo.GA_enum_Best_ti_I)
    return 1/maxeam
