import numpy as np
from mesh_to_sdf import sample_sdf_near_surface, mesh_to_sdf
import trimesh

# Load and normalize the mesh to [-1,1]^3
mesh = trimesh.load('sdfs/greek_column.obj')
bbox_min, bbox_max = mesh.bounds
center = (bbox_min + bbox_max) / 2.0
scale = (bbox_max - bbox_min).max() / 2.0
mesh.vertices = (mesh.vertices - center) / scale

num_surface_points = 500000
num_uniform_points = 1

points_surface, sdf_surface = sample_sdf_near_surface(
    mesh, number_of_points=num_surface_points
)

# Do not sample uniform points, this is experimental
points_uniform = np.random.uniform(
    low=-1.0, high=1.0, size=(num_uniform_points, 3)
)
sdf_uniform = mesh_to_sdf(mesh, points_uniform, surface_point_method='sample')


all_points = np.vstack([points_surface, points_uniform])
all_sdf = np.hstack([sdf_surface, sdf_uniform])

# Shuffle
perm = np.random.permutation(all_points.shape[0])
all_points = all_points[perm]
all_sdf = all_sdf[perm]

data = np.hstack([all_points, all_sdf[:, None]]).astype(np.float32) 
data.tofile("greek_column.bin")