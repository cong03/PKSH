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

def envelop(e1, e2):
    eam = np.zeros(len(e1))
    l_x = np.sqrt(np.sum(e1 * e1, axis=1))
    l_y = np.sqrt(np.sum(e2 * e2, axis=1))
    l = l_x * l_y

    # wyh :Add logic to handle division by zero or invalid values
    d_zero_indices = np.where(l == 0)  # Find indices where d is zero
    d_invalid_indices = np.where(np.isnan(l))  # Find indices where d is NaN
    l[d_zero_indices] = 1e-16  # Replace zeros with 1e-9 to avoid division by zero
    l[d_invalid_indices] = 10000000.0  # Replace NaN with 10000000

    point = np.sum(e1 * e2, axis=1)
    cos_ = point / l

    mask = cos_ <= 0
    e1[mask] = -e1[mask]
    cos_[mask] = -cos_[mask]

    equal_vectors = np.all(e1 == e2, axis=1)

    eam[equal_vectors] = 2 * l_x[equal_vectors]
    not_equal_vectors = ~equal_vectors
    mask2 = not_equal_vectors & (l_y < l_x)
    # mask3 = not_equal_vectors & (l_x < l_y * cos_)
    mask3 = not_equal_vectors & (l_y < l_x * cos_)

    eam[mask2 & mask3] = 2 * l_y[mask2 & mask3]
    eam[mask2 & ~mask3] = 2 * np.linalg.norm(np.cross(e2[mask2 & ~mask3], (e1[mask2 & ~mask3] - e2[mask2 & ~mask3])),
                                             axis=1) / np.linalg.norm(e1[mask2 & ~mask3] - e2[mask2 & ~mask3], axis=1)

    # mask4 = not_equal_vectors & (l_y < l_x * cos_)
    mask4 = not_equal_vectors & (l_x < l_y * cos_)
    mask5 = not_equal_vectors & (l_x < l_y)
    eam[mask5 & mask4] = 2 * l_x[~mask2 & mask4]
    eam[mask5 & ~mask4] = 2 * np.linalg.norm(np.cross(e1[mask5 & ~mask4], (e2[mask5 & ~mask4] - e1[mask5 & ~mask4])),
                                             axis=1) / np.linalg.norm(e2[mask5 & ~mask4] - e1[mask5 & ~mask4], axis=1)

    return eam

def get_ratio_intensity(x1, x2, x3, x4,x5, i1, i2):
    electrode1 = int(x1)
    electrode2 = int(x2)
    stimulation1 = np.zeros(NUM_ELE)
    # print(electrode1)
    # print(electrode2)
    # print(type(electrode2))
    # print(electrode2.shape)
    stimulation1[electrode1] = i1 + x5 / NUM_ELE
    stimulation1[electrode2] = -i1 - x5 / NUM_ELE
    e1 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T / 1000

    electrode3 = int(x3)
    electrode4 = int(x4)
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = i2 - x5 / NUM_ELE
    stimulation2[electrode4] = -i2 + x5 / NUM_ELE
    e2 = np.array(
        [np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam_target = envelop(e1, e2) # shape(x,)
    e1_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T / 1000
    e2_ = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    eam_avoid = envelop(e1_, e2_) # shape(x,)
    et_avg = np.mean(abs(eam_target))
    return np.mean(abs(eam_avoid))/et_avg, et_avg, np.max(abs(eam_target))

def get_lfm(x1, x2, x3, x4,x5, i1, i2):
    electrode1 = int(x1)
    electrode2 = int(x2)
    stimulation1 = np.zeros(NUM_ELE)
    # print(electrode1)
    # print(electrode2)
    # print(type(electrode2))
    # print(electrode2.shape)
    stimulation1[electrode1] = i1 + x5 / NUM_ELE
    stimulation1[electrode2] = -i1 - x5 / NUM_ELE
    e1 = np.array(
        [np.matmul(lfm[:, :, 0].T, stimulation1), np.matmul(lfm[:, :, 1].T, stimulation1),
         np.matmul(lfm[:, :, 2].T, stimulation1)]).T / 1000

    electrode3 = int(x3)
    electrode4 = int(x4)
    stimulation2 = np.zeros(NUM_ELE)

    stimulation2[electrode3] = i2 - x5 / NUM_ELE
    stimulation2[electrode4] = -i2 + x5 / NUM_ELE
    e2 = np.array(
        [np.matmul(lfm[:, :, 0].T, stimulation2), np.matmul(lfm[:, :, 1].T, stimulation2),
         np.matmul(lfm[:, :, 2].T, stimulation2)]).T / 1000

    eam = envelop(e1, e2) # shape(x,)
    return eam

def get_lfm1(X, I, num):
    stimulation1 = np.zeros(NUM_ELE)
    stimulation2 = np.zeros(NUM_ELE)
    for k in range(int(num)):
        stimulation1[int(X[k])] = I[k]
    for k in range(int(num)):
        # print(k)
        stimulation2[int(X[num + k])] = I[num + k]


    e1 = np.array(
        [np.matmul(lfm[:, :, 0].T, stimulation1), np.matmul(lfm[:, :, 1].T, stimulation1),
         np.matmul(lfm[:, :, 2].T, stimulation1)]).T / 1000
    e2 = np.array(
        [np.matmul(lfm[:, :, 0].T, stimulation2), np.matmul(lfm[:, :, 1].T, stimulation2),
         np.matmul(lfm[:, :, 2].T, stimulation2)]).T / 1000

    eam = envelop(e1, e2) # shape(x,)
    return eam
def get_mti4_lfm(x1, x2, i, num):
    stimulation1 = np.zeros(NUM_ELE)
    stimulation2 = np.zeros(NUM_ELE)
    # print(int(num/2))
    for k in range(int(num/2)):
        # print(k)
        stimulation1[int(x1[k])] = i[k]
        stimulation1[int(x2[k])] = -i[k]
    for k in range(int(num/2)):
        stimulation2[int(x1[int(num/2) + k])] = i[int(num/2) + k]
        stimulation2[int(x2[int(num/2) + k])] = -i[int(num/2) + k]
    # print(stimulation1)
    # print(stimulation2)
    # print(i)
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000

    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000

    eam_target = envelop(e1,e2)

    e1_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T /1000

    e2_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000

    eam_avoid = envelop(e1_,e2_)
    evg_target = np.mean(abs(eam_target))
    print("< 0 V/m: ", np.sum(eam_target < 0))
    return np.mean(abs(eam_avoid))/evg_target, evg_target, np.max(abs(eam_target)), np.sum(eam_target > 0.2) + np.sum(eam_avoid > 0.2), np.sum(eam_target > 0.2), np.max(abs(eam_avoid)) / np.max(abs(eam_target))

def get_mti_lfm(Vars):
    var_set = np.arange(0, NUM_ELE, 1)
    ii = 0
    X = var_set[np.int32(Vars[0:37])]
    # print(type(x))
    I = Vars[37:]
    l_X = len(X)
    x_nonull = [False] * 75
    stimulation1 = np.zeros(75)
    n_s1 = 0
    stimulation2 = np.zeros(75)
    n_s2 = 0
    used_i = [False] * 37
    sum_I = 0
    I_list = []
    for i in range(l_X):
        if i < l_X / 2:
            if x_nonull[i] or sum_I + I[i] > 2.0:
                continue
            stimulation1[i] = I[i]
            I_list.append(I[i])
            sum_I += I[i]
            x_nonull[i] = True
            used_i[i] = True
            n_s1 += 1
        else:
            if x_nonull[i] or sum_I + I[i] > 2.0:
                continue
            stimulation2[i] = I[i]
            I_list.append(I[i])
            sum_I += I[i]
            x_nonull[i] = True
            used_i[i] = True
            n_s2 += 1

    i = 0
    # print(sum_I)
    sum_I = 0
    # print(ii,I_list)
    for k in range(75):
        if len(I_list) == 0:
            break
        # print(I_list[i])
        # print(i)
        if x_nonull[k] or sum_I + I_list[i] > 2.0:
            continue

        if i < n_s1:
            stimulation1[k] = -I_list[i]
            sum_I += I_list[i]
            x_nonull[k] = True
            i += 1
        if i >= n_s1 and i < n_s1 + n_s2:
            stimulation2[k] = -I_list[i]
            sum_I += I_list[i]
            x_nonull[k] = True
            i += 1
        if i >= n_s1 + n_s2:
            break
    print(ii,"TIS array1: ", np.nonzero(stimulation1)[0])
    print(ii,"TIS array1 intensity: ", stimulation1[stimulation1 != 0])
    print(ii,"TIS array2: ", np.nonzero(stimulation2)[0])
    print(ii,"TIS array2 intensity: ", stimulation2[stimulation2 != 0])
    if np.sum(abs(stimulation2)) < 0.001 or np.sum(abs(stimulation1)) < 0.001:
        return 1000
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000

    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam_target = envelop(e1,e2)

    e1_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T /1000

    e2_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    eam_avoid = envelop(e1_,e2_)

    evg_target = np.mean(abs(eam_target))
    return np.mean(abs(eam_avoid))/evg_target, evg_target, np.max(abs(eam_target))

def get_mti_lfm2(Vars):
    var_set = np.arange(0, NUM_ELE, 1)
    # i = args[0]
    ii = 0
    # Vars = args[1]
    x1 = var_set[np.int32(Vars[0:37])]
    x2 = var_set[np.int32(Vars[37:74])]
    i1 = Vars[74:-1]
    r = Vars[-1]
    x_idx = [False] * 75
    used_i = [False] * 37
    stimulation1 = np.zeros(75)
    stimulation2 = np.zeros(75)
    k1 = 0
    sum_I = 0
    I_list = []
    for i in range(37):
        if sum_I + i1[i] > 2.0:
            break
        if x_idx[x1[i]] or x_idx[x2[i]]:
            continue
        stimulation1[x1[i]] = i1[i] * r
        stimulation1[x2[i]] = - i1[i] * r
        I_list.append(i1[i])
        k1 += 1
        x_idx[x1[i]] = True
        x_idx[x2[i]] = True
        used_i[i] = True
        sum_I += i1[i]
        if k1 > 37 / 2:
            break
    # print(sum_I)
    # print(np.nonzero(stimulation1)[0])
    sum_I = 0
    k = 0
    for i in range(37):
        if sum_I + i1[i] > 2.0:
            break
        if x_idx[x1[i]] or x_idx[x2[i]]:
            continue
        stimulation2[x1[i]] = I_list[k] * (1 - r)
        stimulation2[x2[i]] = - I_list[k] * (1 - r)
        k += 1
        x_idx[x1[i]] = True
        x_idx[x2[i]] = True
        used_i[i] = True
        sum_I += i1[i]
        if k >= k1:
            break
    print(r)
    used_i = np.array(used_i)
    print(abs(i1[used_i == True]))
    print(ii,'sum_i',np.sum(abs(i1[used_i == True])) - 2.0)
    print(ii, "TIS array1: ", np.nonzero(stimulation1)[0])
    print(ii, "TIS array1 intensity: ", stimulation1[stimulation1 != 0])
    print(ii, "TIS array2: ", np.nonzero(stimulation2)[0])
    print(ii, "TIS array2 intensity: ", stimulation2[stimulation2 != 0])
    if np.sum(abs(stimulation2)) < 0.001 or np.sum(abs(stimulation1)) < 0.001:
        return 1000
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000

    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    eam_target = envelop(e1,e2)

    e1_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T /1000

    e2_ = np.array([np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
                   np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    eam_avoid = envelop(e1_,e2_)

    evg_target = np.mean(abs(eam_target))
    return np.mean(abs(eam_avoid))/evg_target, evg_target, np.max(abs(eam_target))