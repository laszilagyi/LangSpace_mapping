import open3d as o3d
import numpy as np

from langspace2_utils import *
from langspace2_eval import *

import gzip
import pickle
import open3d as o3d
#from your_module import MapObjectList  # adjust if it's defined in a notebook or script

from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList


def load_conceptgraphs_objects(path):
    """
    Load conceptgraphs objects from a serialized file.
    
    Args:
        path (str): Path to the serialized objects file.
        
    Returns:
        MapObjectList: A list of map objects loaded from the file.
    """
    with gzip.open(path, "rb") as f:
        serialized_data = pickle.load(f)

    objects = MapObjectList()
    objects.load_serializable(serialized_data['objects'])
    
    # extract all open3d point clouds, like this: object_cg = o3d.geometry.PointCloud(objects[51]['pcd'])

    cg_objects = []
    cg_clip_fts = []

    for obj in objects:
        pcd = o3d.geometry.PointCloud(obj['pcd'])
        cg_objects.append(pcd)

        clip_ft = obj['clip_ft']
        cg_clip_fts.append(clip_ft)

    return cg_objects, cg_clip_fts


def load_gt_objects(pcd_path, instance_labels_path):
    """
    Load ground truth objects from a point cloud and instance labels.
    
    Args:
        pcd_path (str): Path to the point cloud file.
        instance_labels_path (str): Path to the instance labels file.
        
    Returns:
        dict: A dictionary mapping instance IDs to Open3D PointCloud objects.
    """
    
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)

    # Load instance segmentation labels
    instance_ids = np.loadtxt(instance_labels_path, dtype=int)

    # Convert to numpy array (optional, for further processing)
    points = np.asarray(pcd.points)

    colors = np.asarray(pcd.colors)

    # Split the GT point cloud into separate instance point clouds based on instance IDs
    assert len(instance_ids) == len(pcd.points), "labels and points must align"

    unique_ids = np.unique(instance_ids)

    instance_pcds = {}  # id  →  open3d.geometry.PointCloud
    gt_objects = []
    gt_instance_ids = []

    for inst_id in unique_ids:
        mask = instance_ids == inst_id  # boolean mask, shape (N,)
        if not np.any(mask):  # safety-check (shouldn’t happen)
            continue

        pts = points[mask]  # (K, 3)
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(pts)

        # If you have per-point colour (or normals, etc.) copy them too
        if pcd.has_colors():
            new_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        if pcd.has_normals():
            new_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])

        instance_pcds[inst_id] = new_pcd
        gt_objects.append(new_pcd)
        gt_instance_ids.append(inst_id)

    return gt_objects, gt_instance_ids


def filter_overlapping_objects(gt_objects, cg_objects, distance_threshold=0.02, overlap_threshold=0.5):
    """
    Filters cg_objects based on whether they overlap with any gt_object above the overlap_threshold.
    
    Args:
        gt_objects (list of o3d.geometry.PointCloud): Ground truth objects.
        cg_objects (list of o3d.geometry.PointCloud): Candidate objects to filter.
        distance_threshold (float): Distance to consider two points overlapping.
        overlap_threshold (float): Minimum fraction of gt_object that must be overlapped.

    Returns:
        cg_subset (list of o3d.geometry.PointCloud): Subset of cg_objects passing the overlap check.
    """
    cg_subset = []

    for cg_obj in cg_objects:
        for gt_obj in gt_objects:
            overlap_ratio = compute_overlap_ratio(gt_obj, cg_obj, distance_threshold)
            if overlap_ratio >= overlap_threshold:
                cg_subset.append(cg_obj)
                break  # no need to compare with more gt_objects

    return cg_subset


def filter_overlapping_objects_clip(gt_objects, cg_objects, cg_clip_fts, distance_threshold=0.02, overlap_threshold=0.5):
    """
    Filters cg_objects based on whether they overlap with any gt_object above the overlap_threshold.
    
    Args:
        gt_objects (list of o3d.geometry.PointCloud): Ground truth objects.
        cg_objects (list of o3d.geometry.PointCloud): Candidate objects to filter.
        distance_threshold (float): Distance to consider two points overlapping.
        overlap_threshold (float): Minimum fraction of gt_object that must be overlapped.

    Returns:
        cg_subset (list of o3d.geometry.PointCloud): Subset of cg_objects passing the overlap check.
    """
    cg_subset = []
    cg_clip_subset = []

    for cg_obj, cg_clip in zip(cg_objects, cg_clip_fts):
        for gt_obj in gt_objects:
            overlap_ratio = compute_overlap_ratio(gt_obj, cg_obj, distance_threshold)
            if overlap_ratio >= overlap_threshold:
                cg_subset.append(cg_obj)
                cg_clip_subset.append(cg_clip)
                break  # no need to compare with more gt_objects

    return cg_subset, cg_clip_subset



###################
# room0
###

def load_replica_room0_data(postfix = '_1'):

    cg_mapping_path = f"/home/laszlo/Stanford/concept-graphs_data/Replica/room0/exps/backup/r_mapping_stride11{postfix}/pcd_r_mapping_stride11.pkl.gz"
    pcd_path = "/home/laszlo/Stanford/concept-graphs_data/Replica/room0_mesh.ply"
    instance_labels_path = "/home/laszlo/Stanford/opennerf/datasets/replica_gt_instances/instance_labels_room0.txt"

    subset_indices = [3, 10, 14, 15, 24, 25, 29, 30, 31, 32, 33, 34, 
                    35, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 53, 57, 58, 59,
                    60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71]
    ref_gt_indices = [33, 34, 60, 61, 62, 63, 64, 68]
    ref_object_indices = [33, 60, 61, 64, 68]
    ref_labels = ['armchair', 'bowl', 'sofa', 'stool', 'coffee table']

    cg_objects, cg_clip_fts = load_conceptgraphs_objects(cg_mapping_path)
    gt_objects, gt_instance_ids = load_gt_objects(pcd_path, instance_labels_path)
    ref_gt_subset = [gt_objects[i] for i in ref_gt_indices]
    gt_subset = [gt_objects[i] for i in subset_indices]
    ref_objects = [gt_objects[i] for i in ref_object_indices]

    cg_subset, cg_clip_subset = filter_overlapping_objects_clip(gt_subset, cg_objects, cg_clip_fts, distance_threshold=0.1, overlap_threshold=0.2)
    ref_cg_subset, ref_cg_clip_subset = filter_overlapping_objects_clip(ref_gt_subset, cg_objects, cg_clip_fts, distance_threshold=0.1, overlap_threshold=0.2)

    return gt_subset, cg_subset, cg_clip_subset, ref_gt_subset, ref_cg_subset, ref_cg_clip_subset, ref_objects, ref_labels





###################
# room2
###

def load_replica_room2(postfix = '_1'):

    cg_mapping_path = f"/home/laszlo/Stanford/concept-graphs_data/Replica/room2/exps/backup/r_mapping_stride11{postfix}/pcd_r_mapping_stride11.pkl.gz"
    pcd_path = "/home/laszlo/Stanford/concept-graphs_data/Replica/room2_mesh.ply"
    instance_labels_path = "/home/laszlo/Stanford/opennerf/datasets/replica_gt_instances/instance_labels_room2.txt"

    subset_indices = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 27, 28, 33, 34, 35, 36, 37, 39, 40, 
                    41, 42, 43, 44]
    ref_gt_indices = [16, 17, 18, 19, 20, 21, 22, 23, 37, 39, 40, 41]
    ref_object_indices = [ 18, 37, 39, 40, 41]
    ref_labels = ['chair', 'cabinet', 'table', 'bowl', 'vase']

    cg_objects, cg_clip_fts = load_conceptgraphs_objects(cg_mapping_path)
    gt_objects, gt_instance_ids = load_gt_objects(pcd_path, instance_labels_path)
    ref_gt_subset = [gt_objects[i] for i in ref_gt_indices]
    gt_subset = [gt_objects[i] for i in subset_indices]
    ref_objects = [gt_objects[i] for i in ref_object_indices]

    cg_subset, cg_clip_subset = filter_overlapping_objects_clip(gt_subset, cg_objects, cg_clip_fts, distance_threshold=0.1, overlap_threshold=0.2)
    ref_cg_subset, ref_cg_clip_subset = filter_overlapping_objects_clip(ref_gt_subset, cg_objects, cg_clip_fts, distance_threshold=0.1, overlap_threshold=0.2)

    return gt_subset, cg_subset, cg_clip_subset, ref_gt_subset, ref_cg_subset, ref_cg_clip_subset, ref_objects, ref_labels

















