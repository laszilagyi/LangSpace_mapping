

import os, torch, time, shutil, json,glob,sys,copy, argparse
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch import optim, nn
import open3d as o3d

sys.path.append('/home/laszlo/Stanford/OverlapPredator')

from datasets.indoor import IndoorDataset
from datasets.dataloader import get_dataloader
from models.architectures import KPFCNN
from lib.utils import load_obj, setup_seed,natural_key, load_config
from lib.benchmark_utils import ransac_pose_estimation, to_o3d_pcd, get_blue, get_yellow, to_tensor
from lib.trainer import Trainer
from lib.loss import MetricLoss
import shutil
setup_seed(0)

class ThreeDMatchDemo(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self,config, src_pcd, tgt_pcd, voxel_down_sample=0.025):
        super(ThreeDMatchDemo,self).__init__()
        self.config = config
        self.src_pcd = src_pcd
        self.tgt_pcd = tgt_pcd

        self.voxel_down_sample = voxel_down_sample

    def __len__(self):
        return 1

    def __getitem__(self,item): 
        # get pointcloud
        #src_pcd = torch.load(self.src_path).astype(np.float32)
        #tgt_pcd = torch.load(self.tgt_path).astype(np.float32)   


        #src_pcd = o3d.io.read_point_cloud("/home/laszlo/Stanford/concept-graphs_data/object_pointclouds/sofa_gt.ply")
        #tgt_pcd = o3d.io.read_point_cloud("/home/laszlo/Stanford/concept-graphs_data/object_pointclouds/sofa_cg.ply")

        src_pcd = self.src_pcd
        tgt_pcd = self.tgt_pcd

        #src_pcd = src_pcd.voxel_down_sample(0.025)
        #tgt_pcd = tgt_pcd.voxel_down_sample(0.025)

        src_pcd = src_pcd.voxel_down_sample(self.voxel_down_sample)
        tgt_pcd = tgt_pcd.voxel_down_sample(self.voxel_down_sample)
        
        #src_pcd = o3d.io.read_point_cloud(self.src_path)
        #tgt_pcd = o3d.io.read_point_cloud(self.tgt_path)
        #src_pcd = src_pcd.voxel_down_sample(0.025)
        #tgt_pcd = tgt_pcd.voxel_down_sample(0.025)
        src_pcd = np.array(src_pcd.points).astype(np.float32)
        tgt_pcd = np.array(tgt_pcd.points).astype(np.float32)


        src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
        tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

        # fake the ground truth information
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3,1)).astype(np.float32)
        correspondences = torch.ones(1,2).long()

        return src_pcd,tgt_pcd,src_feats,tgt_feats,rot,trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)

def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1,1,1)'''
    color = np.array(color)
    white = np.array([1, 1, 1])
    vector = white-color
    return color + vector * percent

def draw_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, src_saliency, tgt_saliency, tsfm):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. overlap colors
    rot, trans = to_tensor(tsfm[:3,:3]), to_tensor(tsfm[:3,3][:,None])
    src_overlap = src_overlap[:,None].repeat(1,3).numpy()
    tgt_overlap = tgt_overlap[:,None].repeat(1,3).numpy()
    src_overlap_color = lighter(get_yellow(), 1 - src_overlap)
    tgt_overlap_color = lighter(get_blue(), 1 - tgt_overlap)
    src_pcd_overlap = copy.deepcopy(src_pcd_before)
    src_pcd_overlap.transform(tsfm)
    tgt_pcd_overlap = copy.deepcopy(tgt_pcd_before)
    src_pcd_overlap.colors = o3d.utility.Vector3dVector(src_overlap_color)
    tgt_pcd_overlap.colors = o3d.utility.Vector3dVector(tgt_overlap_color)

    ########################################
    # 3. draw registrations
    src_pcd_after = copy.deepcopy(src_pcd_before)
    src_pcd_after.transform(tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input', width=960, height=540, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='Inferred overlap region', width=960, height=540, left=0, top=600)
    vis2.add_geometry(src_pcd_overlap)
    vis2.add_geometry(tgt_pcd_overlap)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name ='Our registration', width=960, height=540, left=960, top=0)
    vis3.add_geometry(src_pcd_after)
    vis3.add_geometry(tgt_pcd_before)
    
    while True:
        vis1.update_geometry(src_pcd_before)
        vis3.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_overlap)
        vis2.update_geometry(tgt_pcd_overlap)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        vis3.update_geometry(src_pcd_after)
        vis3.update_geometry(tgt_pcd_before)
        if not vis3.poll_events():
            break
        vis3.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()  



def refine_registration(source, target, global_reg, voxel_size):
    
    # compute normals on the original clouds
    radius_normal = voxel_size * 2
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, global_reg,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


#def ransac_icp_registration(gt_object, cg_object):
class PREDATOR_Registration():
    def __init__(self):
        # TODO: move to constructor
        path = "/home/laszlo/Stanford/OverlapPredator/configs/test/indoor.yaml"
        self.config = load_config(path)
        self.config = edict(self.config)
        self.config.device = torch.device('cuda')

        #print(config)

        # model initialization
        self.config.architecture = [
            'simple',
            'resnetb',
        ]
        for i in range(self.config.num_layers-1):
            self.config.architecture.append('resnetb_strided')
            self.config.architecture.append('resnetb')
            self.config.architecture.append('resnetb')
        for i in range(self.config.num_layers-2):
            self.config.architecture.append('nearest_upsample')
            self.config.architecture.append('unary')
        self.config.architecture.append('nearest_upsample')
        self.config.architecture.append('last_unary')
        self.config.model = KPFCNN(self.config).to(self.config.device)

        # load pretrained weights
        assert self.config.pretrain != None
        state = torch.load(self.config.pretrain)
        self.config.model.load_state_dict(state['state_dict'])


    def register(self, src_pcd_o3d, tgt_pcd_o3d, voxel_down_sample=0.025):
        # TODO: put in a method

        # create dataset and dataloader
        info_train = load_obj(self.config.train_info)
        train_set = IndoorDataset(info_train,self.config,data_augmentation=True)
        demo_set = ThreeDMatchDemo(self.config, src_pcd_o3d, tgt_pcd_o3d, voxel_down_sample)

        _, neighborhood_limits = get_dataloader(dataset=train_set,
                                            batch_size=self.config.batch_size,
                                            shuffle=True,
                                            num_workers=self.config.num_workers,
                                            )
        demo_loader, _ = get_dataloader(dataset=demo_set,
                                            batch_size=self.config.batch_size,
                                            shuffle=False,
                                            num_workers=1,
                                            neighborhood_limits=neighborhood_limits)


        self.config.model.eval()
        c_loader_iter = demo_loader.__iter__()
        with torch.no_grad():
            #inputs = c_loader_iter.next()
            inputs = next(c_loader_iter)
            ##################################
            # load inputs to device.
            for k, v in inputs.items():  
                if type(v) == list:
                    inputs[k] = [item.to(self.config.device) for item in v]
                else:
                    inputs[k] = v.to(self.config.device)

            ###############################################
            # forward pass
            feats, scores_overlap, scores_saliency = self.config.model(inputs)  #[N1, C1], [N2, C2]
            pcd = inputs['points'][0]
            len_src = inputs['stack_lengths'][0][0]
            c_rot, c_trans = inputs['rot'], inputs['trans']
            correspondence = inputs['correspondences']
            
            src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
            src_raw = copy.deepcopy(src_pcd)
            tgt_raw = copy.deepcopy(tgt_pcd)
            src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
            src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[:len_src].detach().cpu()
            tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[len_src:].detach().cpu()

            ########################################
            # do probabilistic sampling guided by the score
            src_scores = src_overlap * src_saliency
            tgt_scores = tgt_overlap * tgt_saliency

            if(src_pcd.size(0) > self.config.n_points):
                idx = np.arange(src_pcd.size(0))
                probs = (src_scores / src_scores.sum()).numpy().flatten()
                idx = np.random.choice(idx, size= self.config.n_points, replace=False, p=probs)
                src_pcd, src_feats = src_pcd[idx], src_feats[idx]
            if(tgt_pcd.size(0) > self.config.n_points):
                idx = np.arange(tgt_pcd.size(0))
                probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
                idx = np.random.choice(idx, size= self.config.n_points, replace=False, p=probs)
                tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

            ########################################
            # run ransac and draw registration
            tsfm = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)
            #draw_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, src_saliency, tgt_saliency, tsfm)

            # refine registration with ICP
            voxel_size = 0.02
            result_icp = refine_registration(src_pcd_o3d, tgt_pcd_o3d, tsfm, voxel_size)

        transformation = result_icp.transformation

        return transformation









