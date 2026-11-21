import numpy as np
from cluster import createClusters
from point import makePointList


def kmeans(point_data, cluster_data):
    points = makePointList(point_data)
    clusters = createClusters(cluster_data)

    changed = True

    while changed:
        changed = False

        for p in points:
            closest_cluster = p.closest(clusters)

            if p.moveToCluster(closest_cluster):
                changed = True

        for c in clusters:
            c.updateCenter()

    return clusters



if __name__ == "__main__":
    data = np.array(
        [
            [12.1, -7.1], [0.5, 2.8], [1.2, 5.3], [10.3, -4.8], [-1.1, 3.9],
            [8.9, -3.6], [11.5, -6.2], [7.4, -2.5], [10.8, -5.5], [9.4, -4.3]
        ],
        dtype=float,
    )
    centers = np.array([[0, 0], [1, 1]], dtype=float)

    clusters = kmeans(data, centers)
    for c in clusters:
        c.printAllPoints()
