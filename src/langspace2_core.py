import open3d as o3d
import numpy as np

from langspace2_utils import *
from langspace2_core import *
from langspace2_ransac_icp import *
from langspace2_predator import *
from langspace2_eval import *
from langspace2_data import *


def multi_method_registration(src_pcd, tgt_pcd, ransac_icp_registration, predator_registration):
    
    for _ in range(3):    
        src_pcd_copy = copy.deepcopy(src_pcd)
        voxel_size = 0.02
        transformation = ransac_icp_registration.register(src_pcd_copy, tgt_pcd, voxel_size)
        src_pcd_copy = src_pcd_copy.transform(transformation)
        fitness, _ = eval_registraion(src_pcd_copy, tgt_pcd)
        print(f"Fitness after RANSAC ICP (0.02): {fitness}")
        if fitness > 0.7:
            return transformation

    for _ in range(3):    
        src_pcd_copy = copy.deepcopy(src_pcd)
        voxel_size = 0.1
        transformation = ransac_icp_registration.register(src_pcd_copy, tgt_pcd, voxel_size)
        src_pcd_copy = src_pcd_copy.transform(transformation)
        fitness, _ = eval_registraion(src_pcd_copy, tgt_pcd)
        print(f"Fitness after RANSAC ICP (0.1): {fitness}")
        if fitness > 0.7:
            return transformation

    for _ in range(3):    
        src_pcd_copy = copy.deepcopy(src_pcd)
        voxel_down_sample=0.025
        transformation = predator_registration.register(src_pcd_copy, tgt_pcd, voxel_down_sample)
        src_pcd_copy = src_pcd_copy.transform(transformation)
        fitness, _ = eval_registraion(src_pcd_copy, tgt_pcd)
        print(f"Fitness after PREDATOR (0.025): {fitness}")
        if fitness > 0.4:
            return transformation

    for _ in range(3):    
        src_pcd_copy = copy.deepcopy(src_pcd)
        voxel_down_sample=0.08
        transformation = predator_registration.register(src_pcd_copy, tgt_pcd, voxel_down_sample)
        src_pcd_copy = src_pcd_copy.transform(transformation)
        fitness, _ = eval_registraion(src_pcd_copy, tgt_pcd)
        print(f"Fitness after PREDATOR (0.08): {fitness}")
        if fitness > 0.4:
            return transformation

    # If we reach here, it means none of the methods succeeded
    transformation = None

    return transformation


def langspace2_registration(ref_objects, ref_labels, cg_objects, cg_clip_fts):


    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to("cuda").eval()    # eval mode

    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")


    ransac_icp_reg = RANSAC_ICP_Registration()
    predator_reg = PREDATOR_Registration()

    updated_cg_objects = copy.deepcopy(cg_objects)


    for i, ref_object in enumerate(ref_objects):
        

        # 2. Prepare your text as a *list* and tokenize
        text_query = ref_labels[i]
        text_tokens = clip_tokenizer([text_query])   # note the list!
        # text_tokens is a LongTensor of shape (1, seq_len)

        # 3. Move tokens to GPU
        text_tokens = text_tokens.to("cuda")

        # 4. Encode
        with torch.no_grad():
            text_query_ft = clip_model.encode_text(text_tokens)
            # text_query_ft is (1, D)

        # 5. If you want a (D,) vector rather than (1, D):
        text_query_ft = text_query_ft.squeeze(0)      # shape (D,)

        candidate_indices, _ = cosine_similarity_search_ft_list(cg_clip_fts, text_query_ft, top_k=10, similarity_threshold=0.27)

        for candidate_i in candidate_indices:

            src_pcd = copy.deepcopy(ref_object)
            tgt_pcd = copy.deepcopy(cg_objects[candidate_i])
            
            transformation = multi_method_registration(src_pcd, tgt_pcd, ransac_icp_reg, predator_reg)

            if transformation is None:
                print(f"Registration failed for ref object: {ref_labels[i]}, CG object index: {candidate_i}")
                continue
            
            src_pcd = src_pcd.transform(transformation)

            #objects[candidate_i]['pcd'] = src_pcd
            updated_cg_objects[candidate_i] = src_pcd

            fitness, rmse = eval_registraion(src_pcd, tgt_pcd)
            print(f"[UPDATED] Ref object: {ref_labels[i]}, CG object index: {candidate_i}, Fitness: {fitness}, RMSE: {rmse}")

    #object_list = [objects[i]['pcd'] for i in range(len(objects))]

    return updated_cg_objects



def single_method_registration(src_pcd, tgt_pcd, ransac_icp_registration, predator_registration):
    
    for _ in range(3):    
        src_pcd_copy = copy.deepcopy(src_pcd)
        voxel_down_sample=0.08
        transformation = predator_registration.register(src_pcd_copy, tgt_pcd, voxel_down_sample)
        src_pcd_copy = src_pcd_copy.transform(transformation)
        fitness, _ = eval_registraion(src_pcd_copy, tgt_pcd)
        print(f"Fitness after PREDATOR (0.08): {fitness}")
        if fitness > 0.4:
            return transformation

    # If we reach here, it means none of the methods succeeded
    transformation = None

    return transformation




def single_registration(ref_objects, ref_labels, cg_objects, cg_clip_fts):


    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to("cuda").eval()    # eval mode

    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")


    ransac_icp_reg = RANSAC_ICP_Registration()
    predator_reg = PREDATOR_Registration()

    updated_cg_objects = copy.deepcopy(cg_objects)


    for i, ref_object in enumerate(ref_objects):
        

        # 2. Prepare your text as a *list* and tokenize
        text_query = ref_labels[i]
        text_tokens = clip_tokenizer([text_query])   # note the list!
        # text_tokens is a LongTensor of shape (1, seq_len)

        # 3. Move tokens to GPU
        text_tokens = text_tokens.to("cuda")

        # 4. Encode
        with torch.no_grad():
            text_query_ft = clip_model.encode_text(text_tokens)
            # text_query_ft is (1, D)

        # 5. If you want a (D,) vector rather than (1, D):
        text_query_ft = text_query_ft.squeeze(0)      # shape (D,)

        candidate_indices, _ = cosine_similarity_search_ft_list(cg_clip_fts, text_query_ft, top_k=10, similarity_threshold=0.27)

        for candidate_i in candidate_indices:

            src_pcd = copy.deepcopy(ref_object)
            tgt_pcd = copy.deepcopy(cg_objects[candidate_i])
            
            transformation = multi_method_registration(src_pcd, tgt_pcd, ransac_icp_reg, predator_reg)

            if transformation is None:
                print(f"Registration failed for ref object: {ref_labels[i]}, CG object index: {candidate_i}")
                continue
            
            src_pcd = src_pcd.transform(transformation)

            #objects[candidate_i]['pcd'] = src_pcd
            updated_cg_objects[candidate_i] = src_pcd

            fitness, rmse = eval_registraion(src_pcd, tgt_pcd)
            print(f"[UPDATED] Ref object: {ref_labels[i]}, CG object index: {candidate_i}, Fitness: {fitness}, RMSE: {rmse}")

    #object_list = [objects[i]['pcd'] for i in range(len(objects))]

    return updated_cg_objects

