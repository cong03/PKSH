from public import glo
import numpy as np

NUM_ELE = glo.NUM_ELE

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
    # 储存六个位置的坐标
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

# 获得比例和电场
def get_tdcsenum_lfm(x, i, num):
    stimulation1 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num)):
        stimulation1[x[k]] = i[k]

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
    print("no abs e1 target > 0.2 V/m: ", np.sum(e1 > 0.2))
    print("no abs e2 target > 0.2 V/m: ", np.sum(e2 > 0.2))
    print("e1 < 0: ", np.sum(e1 < 0))
    print("e2 < 0: ", np.sum(e2 < 0))
    return e_ratio, evg_target, e1_max, np.sum(abs(e1) > 0.2) + np.sum(abs(e2) > 0.2), np.sum(abs(e1) > 0.2), em_ratio

def get_tdcs_lfm(x, i, num):
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
    print("no abs e1 target > 0.2 V/m: ", np.sum(e1 > 0.2))
    print("no abs e2 target > 0.2 V/m: ", np.sum(e2 > 0.2))
    print("e1 < 0: ", np.sum(e1 < 0))
    print("e2 < 0: ", np.sum(e2 < 0))
    return e_ratio, evg_target, e1_max, np.sum(abs(e1) > 0.2) + np.sum(abs(e2) > 0.2), np.sum(abs(e1) > 0.2), em_ratio

def get_autotdcs_lfm(x, i, num):
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
    print("no abs e1 target > 0.2 V/m: ", np.sum(e1 > 0.2))
    print("no abs e2 target > 0.2 V/m: ", np.sum(e2 > 0.2))
    print("e1 < 0: ", np.sum(e1 < 0))
    print("e2 < 0: ", np.sum(e2 < 0))
    return e_ratio, evg_target, e1_max, np.sum(abs(e1) > 0.2) + np.sum(abs(e2) > 0.2), np.sum(abs(e1) > 0.2), em_ratio