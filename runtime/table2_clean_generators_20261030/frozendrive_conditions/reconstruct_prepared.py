import os,json,time,sys
from pathlib import Path
import numpy as np
import torch,pickle,yaml
from chamferdist.chamfer import knn_points
from generate_occ_mini import nksr_mesh_normal
W=Path(__file__).parent
prepared=Path(sys.argv[1]);save_path=sys.argv[2]
z=np.load(prepared);point=z['point'];normal=z['normal'];scene_semantic_points=z['semantic_points'];dict={'sample_token':str(z['sample_token']), 'lidar_token':str(z['lidar_token'])}
config=yaml.safe_load((W/'config800.yaml').read_text());voxel_size=config['voxel_size'];pc_range=config['pc_range'];occ_size=config['occ_size'];smart_path_join=os.path.join
g=torch.Generator().manual_seed(7)
a=torch.randn(1,103,3,generator=g).cuda();b=torch.randn(1,317,3,generator=g).cuda()
k=knn_points(a,b,K=1);d=((a[:,:,None,:]-b[:,None,:,:])**2).sum(-1);dv,di=d.min(-1)
assert torch.equal(k.idx[...,0],di)
assert torch.allclose(k.dists[...,0],dv,atol=1e-6,rtol=1e-6)
print('CUDA nearest-neighbor operation passed',flush=True)
del a,b,k,d,dv,di
start=time.time()
point = torch.from_numpy(point).float().cuda()
normal = torch.from_numpy(normal).float().cuda()

with torch.no_grad():
    nksr_mesh = nksr_mesh_normal(point, normal, detail_level=0.5, mise_iter=1, cpu_=False)

scene_points = np.asarray(nksr_mesh.v.cpu(), dtype=float)

################## remain points with a spatial range ##############
mask = (
    (np.abs(scene_points[:, 0]) < 50)
    & (np.abs(scene_points[:, 1]) < 50)
    & (scene_points[:, 2] > -5.0)
    & (scene_points[:, 2] < 3.0)
)
scene_points = scene_points[mask]

################## convert points to voxels ##############
pcd_np = scene_points
pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
pcd_np = np.floor(pcd_np).astype(np.int32)
voxel = np.zeros(occ_size)
voxel[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = 1

################## convert voxel coordinates to LiDAR system  ##############
gt_ = voxel
x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
z = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
vv = np.stack([X, Y, Z], axis=-1)
fov_voxels = vv[gt_ > 0]
fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
fov_voxels[:, 0] += pc_range[0]
fov_voxels[:, 1] += pc_range[1]
fov_voxels[:, 2] += pc_range[2]

################## get semantics of sparse points  ##############
mask = (
    (np.abs(scene_semantic_points[:, 0]) < 50)
    & (np.abs(scene_semantic_points[:, 1]) < 50)
    & (scene_semantic_points[:, 2] > -5.0)
    & (scene_semantic_points[:, 2] < 3.0)
)
scene_semantic_points = scene_semantic_points[mask]

################## Nearest Neighbor to assign semantics ##############
dense_voxels = fov_voxels
sparse_voxels_semantic = scene_semantic_points

x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float()
y = torch.from_numpy(sparse_voxels_semantic[:, :3]).cuda().unsqueeze(0).float()
idx1 = knn_points(x, y, K=1).idx[..., 0]
indices = idx1[0].cpu().numpy()

dense_semantic = sparse_voxels_semantic[:, 3][np.array(indices)]
dense_voxels_with_semantic = np.concatenate([fov_voxels, dense_semantic[:, np.newaxis]], axis=1)

# to voxel coordinate
pcd_np = dense_voxels_with_semantic
pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
dense_voxels_with_semantic = np.floor(pcd_np).astype(np.int32)

# dirs = os.path.join(save_path, 'dense_voxels_with_semantic/')
dirs = smart_path_join(save_path, "dense_voxels_with_semantic/")
if not os.path.exists(dirs):
    os.makedirs(dirs)
# np.save(os.path.join(dirs, dict['pc_file_name'] + '.npy'), dense_voxels_with_semantic)
lidar_data_path = smart_path_join(dirs, dict["sample_token"], dict["lidar_token"] + ".npy")
os.makedirs(os.path.dirname(lidar_data_path), exist_ok=True)
with open(lidar_data_path, "wb") as f:
    pickle.dump(dense_voxels_with_semantic, f)

print(json.dumps({'status':'passed','seconds':time.time()-start,'sample_token':dict['sample_token'],'output':lidar_data_path,'shape':list(dense_voxels_with_semantic.shape),'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3}),flush=True)
