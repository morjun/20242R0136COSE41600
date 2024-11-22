# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# pcd 파일 불러오기, 필요에 맞게 경로 수정
file_path = "test_data/1727320101-665925967.pcd"
# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# 빠른 연산 및 전처리를 위한 Voxel downsampling
voxel_size = 0.5  # 필요에 따라 voxel 크기를 조정하세요.
voxel_downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Statistical Outlier Removal (SOR) 적용
cl, ind = voxel_downsample_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
sor_downsampled_pcd = voxel_downsample_pcd.select_by_index(ind)

# Radius Outlier Removal (ROR) 적용
cl, ind = voxel_downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.0)
ror_downsampled_pcd = voxel_downsample_pcd.select_by_index(ind)

# 각 포인트 클라우드에 색상 지정
original_pcd.paint_uniform_color([0, 0, 0])
voxel_downsample_pcd.paint_uniform_color([0, 0.5, 1])  # 파랑
sor_downsampled_pcd.paint_uniform_color([1, 0, 0])  # 빨강
ror_downsampled_pcd.paint_uniform_color([0, 1, 0])  # 초록

# 포인트 클라우드 시각화 함수
def visualize_point_clouds(pcd_list, window_name="Point Cloud Visualization", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 시각화 확인 (포인트 크기를 원하는 크기로 조절 가능)
visualize_point_clouds([original_pcd, voxel_downsample_pcd], 
                       window_name="Original (Black), Voxel Downsampled (Red)", point_size=2.0)

visualize_point_clouds([original_pcd, sor_downsampled_pcd], 
                       window_name="Original (Black), SOR (Blue)", point_size=2.0)

visualize_point_clouds([original_pcd, ror_downsampled_pcd], 
                       window_name="Original (Black), ROR (Green)", point_size=2.0)

