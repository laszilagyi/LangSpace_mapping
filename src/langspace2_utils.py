import open3d as o3d
import numpy as np
import gzip
import pickle
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList

import torch.nn.functional as F

from pathlib import Path
import numpy as np
import open3d as o3d
import torch, clip
from tqdm import tqdm
from PIL import Image
import gc


import torch, numpy as np
from geotransformer.utils.data  import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import (
    make_open3d_point_cloud, get_color, draw_geometries
)
from geotransformer.utils.registration import compute_registration_error

import sys, pathlib
sys.path.append(str(pathlib.Path('/home/laszlo/Stanford/GeoTransformer/experiments/match').resolve()))
from config import make_cfg
from model import create_model



def center_random_rotate_object(object):
    """
    Center and randomly rotate an Open3D PointCloud object.
    
    Args:
        object (o3d.geometry.PointCloud): The point cloud object to center and rotate.
        
    Returns:
        o3d.geometry.PointCloud: The centered and rotated point cloud object.
    """
    # Center the object
    center = object.get_center()
    object.translate(-center, relative=True)

    # seed depending on object point cloud sie
    seed = int(np.sum(np.asarray(object.points)) * 1000) % 2**32

    # Random rotation around z axis
    rng = np.random.default_rng(seed)
    angle_z = rng.uniform(0, 2 * np.pi)  # random angle in [0, 2π)

    # Build a rotation matrix that has rotation only in z
    R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, angle_z))
    object.rotate(R, center=(0, 0, 0))  # apply around origin

    #Note: random across all axes
    #rng = np.random.default_rng(0)
    #rng = np.random.default_rng()
    #angles = rng.uniform(0, 2*np.pi, size=3)
    #R = o3d.geometry.get_rotation_matrix_from_xyz(angles)
    #object.rotate(R, center=(0, 0, 0))

    return object


def register_with_geotransformer(object_gt, object_cg, weights_path):
                                 
    # ------------------------------------------------------------------
    # 2. Load point clouds  (same helper logic as demo.py → load_data)
    # ------------------------------------------------------------------
    src_pts = torch.tensor(np.asarray(object_gt.points), dtype=torch.float32)  # (Nsrc, 3)
    ref_pts = torch.tensor(np.asarray(object_cg.points), dtype=torch.float32)  # (Nref, 3)
    src_feats = np.ones_like(src_pts[:, :1])            # dummy 1‑D feature
    ref_feats = np.ones_like(ref_pts[:, :1])

    pair_dict = {
        "ref_points": ref_pts,
        "src_points": src_pts,
        "ref_feats":  ref_feats,
        "src_feats":  src_feats,
        # transform key is optional unless you want GT error later
        "transform":  np.eye(4, dtype=np.float32)
    }

    # ------------------------------------------------------------------
    # 3. Collate to the stack‑mode batch tensor
    # ------------------------------------------------------------------
    cfg = make_cfg()
    neighbor_limits = [38, 36, 36, 38]                  # 3DMatch default
    batch = registration_collate_fn_stack_mode(
        [pair_dict],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
    )

    # ------------------------------------------------------------------
    # 4. Build model and load checkpoint
    # ------------------------------------------------------------------
    model = create_model(cfg).cuda().eval()             # same as demo.py
    state_dict = torch.load(weights_path, map_location="cuda")
    model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------
    batch_cuda = to_cuda(batch)
    with torch.no_grad():
        out = model(batch_cuda)
    out = release_cuda(out)                             # back to numpy / CPU

    # ------------------------------------------------------------------
    # 6. Results: R, t  (+ optional visual check)
    # ------------------------------------------------------------------
    T = out["estimated_transform"]                      # 4×4 SE(3)
    R, t = T[:3, :3], T[:3, 3]
    print("Rotation R:\n", np.round(R, 4))
    print("\nTranslation t:\n", np.round(t, 4))

    # visualise raw vs. aligned
    ref_pcd = make_open3d_point_cloud(out["ref_points"])
    ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    src_pcd = make_open3d_point_cloud(out["src_points"])
    src_pcd.paint_uniform_color(get_color("custom_blue"))
    #draw_geometries(ref_pcd, src_pcd)                   # before alignment
    #draw_geometries(ref_pcd, src_pcd.transform(T.copy()))  # after alignment

    # optional GT error if you provided a real transform
    rre, rte = compute_registration_error(pair_dict["transform"], T)
    print(f"RRE (deg): {rre:.3f}   RTE (m): {rte:.3f}")

    transformation = T
    
    return transformation, rre, rte

def eval_registraion(gt_object, cg_object, distance_threshold = 0.01):

    # assume both clouds are already in the same frame
    #voxel_size   = 0.02          # 1 cm tolerance (change to suit your scale)
    #max_corr_dist = voxel_size * 1.5
    #max_corr_dist = voxel_size * 0.4


    eval = o3d.pipelines.registration.evaluate_registration(
            cg_object,                           # source (partial / noisy)
            gt_object,                           # target (reference)
            distance_threshold,
            np.eye(4))                           # identity, because already aligned

    fitness = eval.fitness          # ∈ [0,1]  – fraction of source points that have
    rmse    = eval.inlier_rmse      #           a match closer than `max_corr_dist`
    print(f"Overlap (fitness): {fitness:.3f}, inlier‑RMSE: {rmse*1000:.1f} mm")

    return fitness, rmse


def clip_embed_object(gt_object):
    """
    Render 6 canonical views with white background and true colours
    (no directional light) → CLIP embedding.

    Open3D ≥ 0.16; if you have 0.18+ the renderer.release() call will run.
    """

    # ─── 0.  Load point cloud ──────────────────────────────────────────────
    #PCD_FILE = "object_colored.ply"       # change me
    pcd = gt_object
    assert len(pcd.points) > 0, "Empty cloud"
    if not pcd.has_colors():
        raise ValueError("The point cloud has no vertex colours!")

    pcd = o3d.geometry.PointCloud(pcd)



    # seed depending on object point cloud sie
    seed = int(np.sum(np.asarray(pcd.points)) * 1000) % 2**32

    # Random rotation around z axis
    rng = np.random.default_rng(seed)
    angle_z = rng.uniform(0, 2 * np.pi)  # random angle in [0, 2π)

    # Build a rotation matrix that has rotation only in z
    R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, angle_z))
    pcd.rotate(R, center=(0, 0, 0))  # apply around origin



    # ---------- optional: in‑paint missing (nearly black) colours ----------
#    cols = np.asarray(pcd.colors)         # (N,3) in [0..1]
#    near_black = (cols < 0.05).all(axis=1)   # tweak threshold if needed
#    if near_black.any():
#        tree = o3d.geometry.KDTreeFlann(pcd)
#        for idx in tqdm(np.where(near_black)[0], desc="In‑painting colours"):
#            _, nn_idx, _ = tree.search_knn_vector_3d(pcd.points[idx], 8)
#            cols[idx] = cols[nn_idx].mean(axis=0)
#        pcd.colors = o3d.utility.Vector3dVector(cols)

    # ---------- recenter & isotropic scale (optional but handy) -----------
    bbox  = pcd.get_axis_aligned_bounding_box()
    pcd.translate(-bbox.get_center())
    scale = 1.0 / np.linalg.norm(bbox.get_extent())
    pcd.scale(scale, center=(0, 0, 0))
    diag_len = np.linalg.norm(bbox.get_extent())

    # ─── 1.  Off‑screen renderer without lighting ─────────────────────────
    IMG_SIZE   = 256
    POINT_SIZE = 3.0
    FOV_DEG    = 60
    CAM_DIST   = diag_len * 0.6

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader     = "defaultUnlit"          # << keeps raw RGB
    mat.point_size = POINT_SIZE

    renderer = o3d.visualization.rendering.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])       # opaque white
    scene.set_background([0.5, 0.5, 0.5, 0.5])       # opaque white
    scene.scene.enable_sun_light(True)      # turn off default sun light
    scene.add_geometry("obj", pcd, mat)

#    view_dirs = [
#        np.array([ 1,  0,  0]), np.array([-1,  0,  0]),
#        np.array([ 0,  1,  0]), np.array([ 0, -1,  0]),
#        np.array([ 0,  0,  1]), np.array([ 0,  0, -1]),
#    ]

#    rng = np.random.default_rng()
    # draw 6 random 3‑D vectors from a normal distribution
#    vecs = rng.normal(size=(12, 3))
    # normalise each row to length 1  → points on the unit sphere
#    view_dirs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    view_dirs = [
        np.array([ 1,  0,  0]),   # +X  (right side)
        np.array([-1,  0,  0]),   # –X  (left side)
        np.array([ 0,  1,  0]),   # +Y  (front)
        np.array([ 0, -1,  0]),   # –Y  (back)
        np.array([ 0,  0,  1]),     # +Z (top‑down)
    ]

    sqrt2 = np.sqrt(2)          # normalisation factor for [±1, 0, 1] and [0, ±1, 1]

    view_dirs.extend([
    #    np.array([ 1,  0,  1]) / sqrt2,   # +X, up 45°
    #    np.array([-1,  0,  1]) / sqrt2,   # –X, up 45°

    #    np.array([-1,  0, -1]) / sqrt2,   # +X, up 45°
    #    np.array([ 1,  0, -1]) / sqrt2,   # –X, up 45°

    #    np.array([ 0,  1,  1]) / sqrt2,   # +Y, up 45°
    #    np.array([ 0, -1,  1]) / sqrt2,   # –Y, up 45°


        np.array([ 0.        , -0.70710678,  0.70710678]),
        np.array([ 0.70710678,  0.        ,  0.70710678]),
        np.array([ 0.        ,  0.70710678,  0.70710678]),
        np.array([-0.70710678,  0.        ,  0.70710678])
    ])


    renderer.setup_camera(FOV_DEG, center=[0,0,0],
                        eye=[0,0,CAM_DIST], up=[0,1,0])

    img_dir = Path("renders_unlit")
    img_dir.mkdir(exist_ok=True)

    pil_imgs = []
    for i, v in enumerate(view_dirs):
        eye = v * CAM_DIST
        up  = [0,1,0] if abs(v[1]) < 0.99 else [0,0,1]
        renderer.setup_camera(FOV_DEG, center=[0,0,0], eye=eye, up=up)
        img = renderer.render_to_image()
        pil = Image.fromarray(np.asarray(img))
        pil.save(img_dir / f"view_{i}.png")
        pil_imgs.append(pil)

    if hasattr(renderer, "release"):
        renderer.release()
    del renderer

    # ─── 2.  CLIP embedding  ───────────────────────────────────────────────
    device, model_name = ("cuda", "ViT-L/14") if torch.cuda.is_available() \
                        else ("cpu",  "ViT-B/32")
    model, preprocess = clip.load(model_name, device=device)

    with torch.no_grad():
        feats = []
        for img in pil_imgs:
            inp = preprocess(img).unsqueeze(0).to(device)
            f   = model.encode_image(inp)
            feats.append(f / f.norm(dim=-1, keepdim=True))
        feats = torch.cat(feats, 0)
        emb = feats.mean(0)
        emb /= emb.norm()

    print("Final embedding dim:", emb.shape[0])
    # np.save("object_clip_embedding.npy", emb.cpu().numpy())

    clip_ft = emb.cpu().numpy()

    del scene, mat, pcd, model, preprocess

    gc.collect()

    return clip_ft, pil_imgs


def cosine_similarity_search_ft_list(object_fts, query_clip_ft, top_k=5, similarity_threshold=0.1):

    """
    Perform cosine similarity search on a list of objects based on their CLIP features.
    
    Args:
        objects (list): List of objects with 'clip_ft' attributes.
        query_clip_ft (torch.Tensor): The query CLIP feature tensor.
        top_k (int): Number of top similar objects to return.
        
    Returns:
        list: Indices of the top_k most similar objects.
    """
    # Convert query_clip_ft to a tensor if it's not already
    if not isinstance(query_clip_ft, torch.Tensor):
        query_clip_ft = torch.tensor(query_clip_ft, dtype=torch.float32).to(device='cuda')

    # Normalize the query feature
    query_clip_ft = query_clip_ft / query_clip_ft.norm(dim=-1, keepdim=True)

    # Compute similarities
    similarities = []
    for obj_ft in object_fts:
        obj_clip_ft = torch.tensor(obj_ft, dtype=torch.float32).to(device='cuda')
        obj_clip_ft = obj_clip_ft / obj_clip_ft.norm(dim=-1, keepdim=True)
        sim = F.cosine_similarity(query_clip_ft.unsqueeze(0), obj_clip_ft.unsqueeze(0), dim=-1)
        similarities.append(sim.item())

    # Get top_k indices
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # top k similarities
    top_k_similarities = [similarities[i] for i in top_k_indices]

#    return top_k_indices, top_k_similarities

    pairs = list(zip(top_k_indices, top_k_similarities))
    # keep only those with sim > threshold
    filtered = [(i, s) for i, s in pairs if s > similarity_threshold]
    if not filtered:
        return [], []
    # unzip back into two lists
    filt_indices, filt_sims = zip(*filtered)
    return list(filt_indices), list(filt_sims)




def cosine_similarity_search_softmax(object_fts, query_clip_ft, top_k=5):

    """
    Perform cosine similarity search on a list of objects based on their CLIP features.
    
    Args:
        objects (list): List of objects with 'clip_ft' attributes.
        query_clip_ft (torch.Tensor): The query CLIP feature tensor.
        top_k (int): Number of top similar objects to return.
        
    Returns:
        list: Indices of the top_k most similar objects.
    """
    # Convert query_clip_ft to a tensor if it's not already
    if not isinstance(query_clip_ft, torch.Tensor):
        query_clip_ft = torch.tensor(query_clip_ft, dtype=torch.float32).to(device='cuda')

    # Normalize the query feature
    query_clip_ft = query_clip_ft / query_clip_ft.norm(dim=-1, keepdim=True)


    objects_clip_fts = torch.stack(object_fts, dim=0)

    objects_clip_fts = objects_clip_fts.to(device='cuda')
    query_clip_ft    = query_clip_ft.to(device='cuda')

    similarities = F.cosine_similarity(
        query_clip_ft.unsqueeze(0),   # (1, D)
        objects_clip_fts,             # (N, D)
        dim=-1
    )

    print(similarities)

    #max_value = similarities.max()
    #min_value = similarities.min()
    #normalized_similarities = (similarities - min_value) / (max_value - min_value)

    #print(normalized_similarities)

    probs = F.softmax(similarities, dim=1)

    print("probs: ", probs.shape, probs.device, probs.dtype)

    print(probs)

    # top k similarities
    top_k_indices = torch.argsort(probs, descending=True)[:top_k].cpu().numpy()
    
    top_k_indices   = top_k_indices.tolist()                      # now [int, int, …]

    print("top_k_indices: ", top_k_indices)

    # top k probabilities
    top_k_probs = [probs[i].item() for i in top_k_indices]

    return top_k_indices, top_k_probs



def cosine_similarity_search_mapobjects(map_objects, query_clip_ft, top_k=5):

    # Convert query_clip_ft to a tensor if it's not already
    if not isinstance(query_clip_ft, torch.Tensor):
        query_clip_ft = torch.tensor(query_clip_ft, dtype=torch.float32)

    # Normalize the query feature
    query_clip_ft = query_clip_ft / query_clip_ft.norm(dim=-1, keepdim=True)

    similarities = map_objects.compute_similarities(query_clip_ft)
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_k_indices



def visualize_segmentation(object_list):
    """
    Visualize ground truth segmentation objects in Open3D.
    
    Args:
        object_list (list): List of Open3D PointCloud objects representing segments.
    """

    np.random.seed(0)  # for reproducibility

    # Create a list of Open3D geometries for visualization
    geometries = []
    for i, obj in enumerate(object_list):
        obj.paint_uniform_color(np.random.rand(3))  # Assign a random color to each object
        geometries.append(obj)

    # Visualize the objects
    o3d.visualization.draw_geometries(geometries, window_name="Ground Truth Segmentation")


