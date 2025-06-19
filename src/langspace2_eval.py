
import open3d as o3d
import numpy as np
import gzip
import pickle
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList


def evaluate_instance_segmentation(pcd_path, instance_labels_path, objects):

    pcd = o3d.io.read_point_cloud(pcd_path)
    instance_ids = np.loadtxt(instance_labels_path, dtype=int)

    sum_iou = 0
    sum_iou_count = 0

    for ii in range(len(objects)):

        # match gt replica instance segments with conceptgraph objects (to be added to a loop)
        # first, find the closest point in the full scene for each point in the object point cloud
        
        full_points = np.asarray(pcd.points)
        part_points = np.asarray(objects[ii].points)

        # Convert to Open3D point clouds
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(full_points)

        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(part_points)

        # Build KDTree for fast NN search
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_full)

        # Find closest point in full cloud for each point in part cloud
        indices = []
        for i in range(len(part_points)):
            [_, idx, _] = pcd_tree.search_knn_vector_3d(part_points[i], 1)
            indices.append(idx[0])

        matched_indices = np.unique(indices)

        # Get matched subset from full cloud
        matched_subset = full_points[matched_indices]

        matched_ids = instance_ids[matched_indices]

        # Get id with highest count
        unique, counts = np.unique(matched_ids, return_counts=True)
        id_counts = dict(zip(unique, counts))
        max_id = max(id_counts, key=id_counts.get)

        instance_id_indices = np.where(instance_ids == max_id)[0]

        # IoU calculation
        intersection = np.intersect1d(instance_id_indices, matched_indices)
        union = np.union1d(instance_id_indices, matched_indices)
        iou = len(intersection) / len(union) if len(union) > 0 else 0
        #print(f"IoU for ID {max_id}: {iou:.4f}")

        if iou > 0.1:

            #print(f"IoU for ID {max_id}: {iou:.4f}")

            sum_iou += iou
            sum_iou_count += 1
        
            # Visualize intersection with green, remaining points of object object 1 with yellow and object 2 with red, the rest of the scene with gray
            pcd_intersection = o3d.geometry.PointCloud()
            pcd_intersection.points = o3d.utility.Vector3dVector(full_points[intersection])
            max_id_points = full_points[instance_ids == max_id]
            pcd_max_id = o3d.geometry.PointCloud()
            pcd_max_id.points = o3d.utility.Vector3dVector(max_id_points)

            pcd_full.paint_uniform_color([0.7, 0.7, 0.7])  # gray
            pcd_intersection.paint_uniform_color([0, 0, 1])  # blue
            pcd_max_id.paint_uniform_color([1, 0, 0])  # red
            #o3d.visualization.draw_geometries([pcd_full, pcd_max_id, pcd_intersection])

    avg_iou = sum_iou / sum_iou_count if sum_iou_count > 0 else 0
    print(f"Average IoU: {avg_iou:.4f}")



def compute_iou_with_intersection_cloud(reference_object, mapped_object, distance_threshold):
    
    # if mapped_object is empty, return 0.0 IoU
    if len(mapped_object.points) == 0:
        return 0.0, 0, 0, o3d.geometry.PointCloud()
    
    ref_points = np.asarray(reference_object.points)
    map_tree = o3d.geometry.KDTreeFlann(mapped_object)

    intersection_indices = []
    for i, pt in enumerate(ref_points):
        [_, _, sq_dists] = map_tree.search_knn_vector_3d(pt, 1)
        if sq_dists[0] < distance_threshold ** 2:
            intersection_indices.append(i)

    # Create a point cloud for the intersection subset of the reference
    intersection_cloud = reference_object.select_by_index(intersection_indices)

    # Compute IoU
    intersection = len(intersection_indices)
    union = len(ref_points) + len(mapped_object.points) - intersection
    iou = intersection / union if union > 0 else 0.0

    return iou, intersection, union, intersection_cloud


def compute_overlap_ratio(reference_object, mapped_object, distance_threshold):
    """
    Returns the overlap ratio: intersection / len(reference_object)
    """
    ref_points = np.asarray(reference_object.points)
    map_tree = o3d.geometry.KDTreeFlann(mapped_object)

    intersection_count = 0
    for pt in ref_points:
        [_, _, sq_dists] = map_tree.search_knn_vector_3d(pt, 1)
        if sq_dists[0] < distance_threshold ** 2:
            intersection_count += 1

    if len(ref_points) == 0:
        return 0.0

    return intersection_count / len(ref_points)

def eval_reference_object_ious(gt_objects, cg_objects, gt_indices, cg_indices):

    miou = 0.0
    iou_list = []

    gt_list = []
    cg_list = []
    intersection_list = []

    for i in range(len(gt_indices)):

        gt_index = gt_indices[i]
        cg_index = cg_indices[i]

        gt_pcd = o3d.geometry.PointCloud(gt_objects[gt_index])
        cg_pcd = o3d.geometry.PointCloud(cg_objects[cg_index])

        distance_threshold = 0.01

        iou, intersection_cloud = compute_iou_with_intersection_cloud(gt_pcd, cg_pcd, distance_threshold)

        iou_list.append(iou)
        miou += iou
        gt_list.append(gt_pcd)
        cg_list.append(cg_pcd)
        intersection_list.append(intersection_cloud)

    miou /= len(gt_indices)

    print(f"Mean IoU: {miou:.4f}")
    print(f"IoU List: {iou_list}")

    return miou, iou_list, gt_list, cg_list, intersection_list


def overwrite_gt_colors_with_intersection(gt_pcd, intersection_pcd, intersection_color=[0, 1, 0], threshold=1e-5):
    """
    Overwrite colors of gt_pcd with the given color for points that are in intersection_pcd.
    
    Args:
        gt_pcd (o3d.geometry.PointCloud): Ground truth point cloud (will be modified in-place).
        intersection_pcd (o3d.geometry.PointCloud): Subset of gt_pcd (must be close in space).
        intersection_color (list): RGB color to assign to matching points.
        threshold (float): Distance threshold for point matching.
    """
    gt_points = np.asarray(gt_pcd.points)
    gt_colors = np.asarray(gt_pcd.colors)
    intersection_points = np.asarray(intersection_pcd.points)

    # Build KDTree for fast matching
    gt_tree = o3d.geometry.KDTreeFlann(gt_pcd)

    for ipt in intersection_points:
        [_, idx, dist] = gt_tree.search_knn_vector_3d(ipt, 1)
        if dist[0] < threshold ** 2:
            gt_colors[idx[0]] = intersection_color

    gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors)

    return gt_pcd


def compute_intersection_and_union(gt_obj, cg_obj, distance_threshold):
    gt_points = np.asarray(gt_obj.points)
    cg_points = np.asarray(cg_obj.points)

    # Build KDTree for CG
    cg_tree = o3d.geometry.KDTreeFlann(cg_obj)

    # Compute intersection (points in GT close to CG)
    intersection = 0
    for pt in gt_points:
        [_, _, sq_dists] = cg_tree.search_knn_vector_3d(pt, 1)
        if sq_dists[0] < distance_threshold ** 2:
            intersection += 1

    union = len(gt_points) + len(cg_points) - intersection
    iou = intersection / union if union > 0 else 0.0
    return iou

def compute_intersection_size(ref_obj, cand_obj, distance_threshold):
    ref_points = np.asarray(ref_obj.points)
    cand_tree = o3d.geometry.KDTreeFlann(cand_obj)

    count = 0
    for pt in ref_points:
        [_, _, sq_dists] = cand_tree.search_knn_vector_3d(pt, 1)
        if sq_dists[0] < distance_threshold ** 2:
            count += 1
    return count

def eval_segmentation(gt_objects, cg_objects, distance_threshold=0.05):
    num_cg = len(cg_objects)
    num_gt = len(gt_objects)

    # Step 1: Compute cg → gt mapping
    cg_gt_association = []
    cg_gt_overlap = []
    cg_orphan_list = []

    for cg_idx, cg_obj in enumerate(cg_objects):
        best_gt = -1
        best_overlap = 0

        for gt_idx, gt_obj in enumerate(gt_objects):
            overlap = compute_intersection_size(gt_obj, cg_obj, distance_threshold)
            if overlap > best_overlap:
                best_overlap = overlap
                best_gt = gt_idx

        cg_gt_association.append(best_gt)
        cg_gt_overlap.append(best_overlap)

    # Step 2: Compute gt → best cg mapping
    gt_cg_association = [-1] * num_gt
    gt_best_overlap = [0] * num_gt
    gt_orphan_cg_lists = [[] for _ in range(num_gt)]

    for cg_idx, (gt_idx, overlap) in enumerate(zip(cg_gt_association, cg_gt_overlap)):
        if gt_idx == -1:
            continue
        if overlap > gt_best_overlap[gt_idx]:
            # Previously matched cg becomes orphan
            if gt_cg_association[gt_idx] != -1:
                gt_orphan_cg_lists[gt_idx].append((gt_cg_association[gt_idx], gt_best_overlap[gt_idx]))
            gt_cg_association[gt_idx] = cg_idx
            gt_best_overlap[gt_idx] = overlap
        else:
            gt_orphan_cg_lists[gt_idx].append((cg_idx, overlap))

    for gt_idx in range(num_gt):
        gt_orphan_cg_lists[gt_idx].sort(key=lambda x: -x[1])

    # put all orphan indices in the cg_orphan_list list:
    for gt_idx, orphans in enumerate(gt_orphan_cg_lists):
        if len(orphans) > 0:
            cg_orphan_list.append([cg_idx for cg_idx, _ in orphans])
        #else:
        #    cg_orphan_list.append([])

    cg_orphan_list = [item for sublist in cg_orphan_list for item in sublist]


    gt_orphan_counts = [len(lst) for lst in gt_orphan_cg_lists]

    # Step 3: Compute IoUs for GT objects
    gt_ious = []
    for gt_idx, cg_idx in enumerate(gt_cg_association):
        if cg_idx == -1:
            gt_ious.append(0.0)
        else:
            iou = compute_intersection_and_union(gt_objects[gt_idx], cg_objects[cg_idx], distance_threshold)
            gt_ious.append(iou)

    mIoU = sum(gt_ious) / len(gt_ious) if len(gt_ious) > 0 else 0.0

    return {
        'cg_gt_association': cg_gt_association,
        'cg_gt_overlap': cg_gt_overlap,
        'gt_cg_association': gt_cg_association,
        'gt_orphan_cg_lists': gt_orphan_cg_lists,
        'gt_orphan_counts': gt_orphan_counts,
        'cg_orphan_list': cg_orphan_list,
        'gt_ious': gt_ious,
        'mIoU': mIoU
    }


def merge_point_clouds(pcd_list):
    """
    Merges a list of Open3D point clouds into one combined point cloud.
    
    Args:
        pcd_list (list of o3d.geometry.PointCloud): List of input point clouds.

    Returns:
        o3d.geometry.PointCloud: A single merged point cloud.
    """
    merged_pcd = o3d.geometry.PointCloud()
    
    all_points = []
    all_colors = []

    for pcd in pcd_list:
        all_points.append(np.asarray(pcd.points))
        if pcd.has_colors():
            all_colors.append(np.asarray(pcd.colors))
    
    merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    
    if all_colors:
        merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    return merged_pcd


def eval_and_visualize(gt_subset, cg_subset, distance_threshold=0.05):
    
    # 1
    eval = eval_segmentation(gt_subset, cg_subset, distance_threshold)
    per_object_miou = eval['mIoU']
    orphan_cg_objects = sum(eval['gt_orphan_counts'])

    skipped_gt_objects_number = eval['gt_cg_association'].count(-1)
    skipped_gt_indices = [i for i, val in enumerate(eval['gt_cg_association']) if val == -1]

    skipped_gt_objects = [gt_subset[i] for i in skipped_gt_indices]

    print("Per-object mIoU:", per_object_miou)
    print("Orphan CG segments:", orphan_cg_objects)
    print("Skipped GT objects:", skipped_gt_objects_number)

    # 2

    intersection_pcd_list = []
    object_intersection_sum = 0
    object_union_sum = 0

    for gt_i, cg_i in enumerate(eval['gt_cg_association']):
        if cg_i != -1:
            object_iou, object_intersection, object_union, intersection_pcd = compute_iou_with_intersection_cloud(gt_subset[gt_i], cg_subset[cg_i], distance_threshold)
            intersection_pcd_list.append(intersection_pcd)
            object_intersection_sum += object_intersection
            object_union_sum += object_union

    intersection_pcd_union = merge_point_clouds(intersection_pcd_list)

    global_iou = object_intersection_sum / object_union_sum if object_union_sum > 0 else 0.0

    gt_union = merge_point_clouds(gt_subset)

#    segmented_gt_indices = [i for i, val in enumerate(eval['gt_cg_association']) if val != -1]
#    segmented_cg_indices = [eval['gt_cg_association'][i] for i in segmented_gt_indices]
#    segmented_cg_objects = [cg_subset[i] for i in segmented_cg_indices]
#    segmented_cg_union = merge_point_clouds(segmented_cg_objects)

    # number of eval['gt_cg_association'] that are -1:
    



    unsegmented_cg_objects = [cg_subset[i] for i in eval['cg_orphan_list']]
    
    if len(unsegmented_cg_objects) == 0:
        unsegmented_cg_union = o3d.geometry.PointCloud()
    else:
        unsegmented_cg_union = merge_point_clouds(unsegmented_cg_objects)

    # 3
    #global_iou, _, _, gt_intersection_pcd = compute_iou_with_intersection_cloud(gt_union, segmented_cg_union, distance_threshold)
    _ , _, _, orphan_intersection_pcd = compute_iou_with_intersection_cloud(gt_union, unsegmented_cg_union, distance_threshold)

    print("Global IoU:", global_iou)

    # 4
    gt_union.paint_uniform_color([1, 0, 0])  # red
    intersection_pcd_union.paint_uniform_color([0, 1, 0])  # green
    orphan_intersection_pcd.paint_uniform_color([1, 1, 0])  # blue

    #o3d.visualization.draw_geometries([gt_union, intersection_pcd_union, orphan_intersection_pcd])

    return gt_union, intersection_pcd_union, orphan_intersection_pcd, skipped_gt_objects



    