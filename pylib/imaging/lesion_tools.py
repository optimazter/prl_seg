
import SimpleITK as sitk
import monai.transforms
import torch
from collections import deque
import torch.nn.functional as F
from torchvision.transforms import functional as VF
from typing import Type, Union, List, Tuple
import numpy as np
import monai
from rich.progress import Progress
from pylib.imaging.transforms import crop_and_pad






def grow_all_regions(lesions_ten: torch.Tensor, prls_ten: torch.Tensor):
    for i in range(prls_ten.shape[0]):
        seed_points = torch.nonzero(prls_ten[i])
        for seed in seed_points:
            grown_region = region_grow(lesions_ten[i], seed)
            prls_ten[i][grown_region] = 1
    return lesions_ten, prls_ten



def region_grow(tensor: torch.Tensor, seed: torch.Tensor):
    H, W = tensor.shape
    grown_region = torch.zeros_like(tensor, dtype=torch.bool)
    queue = deque()
    queue.append((seed[0].item(), seed[1].item()))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        if 0 <= x < H and 0 <= y < W:
            if tensor[x, y].item() and not grown_region[x, y].item():
                grown_region[x, y] = True
                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if tensor[nx, ny].item() and not grown_region[nx, ny].item():
                            queue.append((nx, ny))
    return grown_region



def expand_lesion(mask: torch.Tensor, expand: int):
    H, W = mask.shape
    mask = torch.rand_like(mask.to(torch.float32)) * mask
    mask = np.kron(mask, np.ones((expand, expand)))
    mask = torch.tensor(mask > 0, dtype=torch.uint8)
    mask = VF.center_crop(mask, [H, W])
    return mask

def expand_lesion_border(mask: torch.Tensor, expand: int):
    #(x - center_x)² + (y - center_y)² < radius²
    non_zero = mask.nonzero().tolist()
    neighbors = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    while len(non_zero) > 0:
        cx, cy = non_zero.pop(0)
        queue = deque()
        queue.append([cx, cy])
        while queue:
            x, y = queue.popleft()
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]:
                    if mask[nx, ny] == 0 and (nx - cx)**2 + (ny - cy)**2 <= expand**2:
                        mask[nx, ny] = 1
                        queue.append([nx, ny])
    return mask


def get_clusters(tensor: torch.Tensor):
    non_zero = set(map(tuple, tensor.nonzero().tolist()))
    H, W = tensor.shape
    neighbors = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]

    clusters = []

    while non_zero:
        queue = deque()
        queue.append(non_zero.pop())
        cluster = []
        while queue:
            x, y = queue.popleft()
            if (x, y) not in cluster:
                cluster.append((x, y))
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if (nx, ny) in non_zero:
                        non_zero.remove((nx, ny))
                        queue.append((nx, ny))

        clusters.append(cluster)

    return clusters


def get_clusters_3d(tensor: torch.Tensor):
    assert(len(tensor.shape) == 3)

    D, H, W = tensor.shape
    non_zero = set(map(tuple, tensor.nonzero().tolist()))
    neighbors = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], 
                 [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0],
                 [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1],
                 [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 1],
                 [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                 [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]

    clusters = []
    while non_zero:
        queue = deque()
        queue.append(non_zero.pop())
        cluster = []
        while queue:
            x, y, z = queue.popleft()
            if (x, y, z) not in cluster:
                cluster.append((x, y, z))
            for dx, dy, dz in neighbors:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < D and 0 <= ny < H and 0 <= nz < W:
                    if (nx, ny, nz) in non_zero:
                        non_zero.remove((nx, ny, nz))
                        queue.append((nx, ny, nz))

        clusters.append(cluster)
    
    return clusters


def get_overlapping_clusters_from_central_lines(clusters, central_lines):
    overlapping_clusters = []
    for line in central_lines:
        for cluster in clusters:
            if line in cluster:
                overlapping_clusters.append(cluster)

    return overlapping_clusters


def get_mask_from_clusters(clusters, shape):
    mask = torch.zeros(shape, dtype=torch.uint8)
    for cluster in clusters:
        for coord in cluster:
            mask[coord] = 1
    return mask


def combine_clusters_by_distance(clusters, min_euclidean_distance):
    combined_clusters = []
    for cluster in clusters:
        if len(combined_clusters) == 0:
            combined_clusters.append(cluster)
            continue
        for combined_cluster in combined_clusters:
            combined = False
            for x, y in cluster:
                for cx, cy in combined_cluster:
                    if np.sqrt((cx - x)**2 + (cy - y)**2) < min_euclidean_distance:
                        combined_cluster.extend(cluster)
                        combined = True
                        break
                if combined:
                    break
            if combined:
                break
        if not combined:
            combined_clusters.append(cluster)
    return combined_clusters



def find_center_of_mass(clusters):
    centers = []
    for cluster in clusters:
        x_sum = 0
        y_sum = 0
        for x, y in cluster:
            x_sum += x
            y_sum += y
        centers.append([int(x_sum / len(cluster)), int(y_sum / len(cluster))])
    return centers

def find_center_of_mass_3d(clusters):
    centers = []
    for cluster in clusters:
        z_sum = 0
        x_sum = 0
        y_sum = 0
        for z, x, y in cluster:
            z_sum += z
            x_sum += x
            y_sum += y
        centers.append([int(z_sum / len(cluster)), int(x_sum / len(cluster)), int(y_sum / len(cluster))])
    return centers


def split_img_by_lesion(img: torch.Tensor, lesions: torch.Tensor, size: int, min_inter_lesion_distance: int, mask: bool = True, expand: int = None):
    assert(len(img.shape) == 2 and len(lesions.shape) == 2)

    clusters = get_clusters(lesions)
    clusters = combine_clusters_by_distance(clusters, min_inter_lesion_distance)
    centres = find_center_of_mass(clusters)


    imgs = []
    labels = []
    for centre, cluster in zip(centres, clusters):
        if mask:
            cluster_mask = torch.zeros_like(lesions, dtype=torch.uint8)
            cluster_mask[torch.tensor(cluster)[:, 0], torch.tensor(cluster)[:, 1]] = 1

        x, y = centre
        label = lesions[x, y]
        if label != 0:
            cluster_img = img.detach().clone()
            if expand is not None and mask:
                cluster_mask = expand_lesion_border(cluster_mask, expand)

            if mask:
                cluster_img[cluster_mask == 0] = 0

            shift = [cluster_img.shape[0] // 2 - x , cluster_img.shape[1] // 2 - y ]
            cluster_img = cluster_img.roll(shift, dims=(0, 1))
            cluster_img = VF.center_crop(cluster_img, size)
            cluster_img = cluster_img.squeeze()

            imgs.append(cluster_img)
            labels.append(label)

    return imgs, labels








def split_img_by_lesions_3d(img: torch.Tensor, lesions: torch.Tensor, size: tuple, min_inter_lesion_distance: int, mask: bool = True, expand: int = None, progress: Progress = None):
    assert(len(img.shape) == 3 and len(lesions.shape) == 3)
    assert(len(size) == 3)

    D, H, W = size
    assert(D % 2 == 0 and H % 2 == 0 and W % 2 == 0)

    clusters = [get_clusters(lesion) for lesion in lesions]
    clusters = [combine_clusters_by_distance(cluster, min_inter_lesion_distance) for cluster in clusters]
    clusters = combine_clusters_by_overlap(clusters)
    centres = find_center_of_mass_3d(clusters)


    imgs, labels = [], []
    if progress:
        task = progress.add_task("Splitting 3D image by lesions...", total=len(centres))

    for centre, cluster in zip(centres, clusters):
        #transform = monai.transforms.SpatialCrop(roi_center = centre, roi_size = size, lazy=None)
        if mask:
            cluster_mask = torch.zeros_like(lesions, dtype=torch.uint8)
            for z, x, y in cluster:
                cluster_mask[z, x, y] = 1

        z, x, y = centre
        label = lesions[z, x, y]
        if label != 0:
            cluster_img = crop_and_pad(img, centre, size)
            if mask:
                cluster_mask = crop_and_pad(cluster_mask, centre, size)
                if expand:
                    for i in range(D):
                        cluster_mask[i] = expand_lesion_border(cluster_mask[i], expand)
                cluster_img[cluster_mask == 0] = 0

            cluster_img.unsqueeze_(0)
            # cluster_img  = transform(cluster_img)

            imgs.append(cluster_img)
            labels.append(label)

        if progress:
            progress.update(task, advance=1)
            progress.refresh()
    
    if progress:
        progress.remove_task(task)
    
    return imgs, labels





def combine_clusters_by_overlap(clusters_4d):
    """
    Merges 2D lesion clusters in consecutive depth slices if they overlap in (x, y).

    Args:
        clusters_4d: A nested list/array of shape [D, N, M, 2], where:
            D = depth (number of 2D slices)
            N = number of clusters in that slice
            M = number of coordinates in each cluster
            The final dimension of size 2 holds the [x, y] coordinates.

    Returns:
        A list of merged 3D clusters, each cluster is a list of [z, x, y].
    """
    D = len(clusters_4d)  # Number of slices (depth)

    # Keep track of whether we've visited a cluster at [slice d, index n]
    visited = [[False] * len(clusters_4d[d]) for d in range(D)]

    # Helper function to check if two clusters overlap in (x, y)
    def overlap(coords_a, coords_b):
        set_a = set(tuple(coord) for coord in coords_a)
        set_b = set(tuple(coord) for coord in coords_b)
        return len(set_a.intersection(set_b)) > 0

    # Depth-first search to gather all connected clusters across slices
    def dfs(d, n, merged_coords):
        stack = [(d, n)]
        while stack:
            dd, nn = stack.pop()
            if visited[dd][nn]:
                continue
            visited[dd][nn] = True

            # Add all (x, y) from this cluster with the current depth dd
            for (x, y) in clusters_4d[dd][nn]:
                merged_coords.append([dd, x, y])

            # Look for overlap in the next slice to merge clusters
            if dd + 1 < D:
                for next_n in range(len(clusters_4d[dd + 1])):
                    if not visited[dd + 1][next_n]:
                        if overlap(clusters_4d[dd][nn], clusters_4d[dd + 1][next_n]):
                            stack.append((dd + 1, next_n))

    merged_3d_clusters = []
    # Go through each slice and each cluster, and if it's not visited, we merge it
    for d in range(D):
        for n in range(len(clusters_4d[d])):
            if not visited[d][n]:
                merged_coords = []
                dfs(d, n, merged_coords)
                merged_3d_clusters.append(merged_coords)

    return merged_3d_clusters



def get_max_diameter_of_cluster(cluster: list):
    """
    Calculate the maximum diameter of a cluster of points.

    Args:
        cluster (list): A list of tuples representing the coordinates of the points in the cluster.

    Returns:
        float: The maximum diameter of the cluster.
    """
    max_distance = 0
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            distance = np.sqrt((cluster[i][0] - cluster[j][0]) ** 2 + (cluster[i][1] - cluster[j][1]) ** 2)
            if distance > max_distance:
                max_distance = distance
    return max_distance

    