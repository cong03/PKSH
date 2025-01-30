import numpy as np

def calculate_weighted_centroid(weights, coordinates, target_point, name=''):

    centroid = np.zeros(3)  #
    num_points = len(weights)  #
    total_weight = 0
    for i in range(num_points):
        #
        centroid += weights[i] * coordinates[i]
        total_weight += weights[i]

    centroid /= total_weight
    print(f"{name}质心坐标: {centroid.tolist()}")
    dist = np.linalg.norm(centroid - target_point)
    print(f"{name}质心到目标点的距离: {dist}")


def main():
    weights = np.array([2, 3, 5])  #
    coordinates = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])  #
    target_point = np.array([1, 1, 1])  #

    #
    calculate_weighted_centroid(weights, coordinates, target_point)



# if __name__ == "__main__":
#     main()