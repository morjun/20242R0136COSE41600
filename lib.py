import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import KDTree
from time import sleep

SCENARIOS = {
    1: "01_straight_walk",
    2: "02_straight_duck_walk",
    3: "03_straight_crawl",
    4: "04_zigzag_walk",
    5: "05_straight_duck_walk",
    6: "06_straight_crawl",
    7: "07_straight_walk",
}


class PointProcessor:
    def __init__(self, args, **kwargs):
        # 기본값 설정
        self.params = {
            "scenario": args.scenario,
            "floor_zvalue": 0.0,
            "scores": {},
        }
        # 전달된 값을 덮어씌움
        self.params.update(kwargs)

    def set_attribute(self, **kwargs):
        self.params.update(kwargs)

    def __getattr__(self, item):
        return self.params.get(item, None)

    def voxel_downsampling(self, pcd, voxel_size=0.5):
        # 빠른 연산 및 전처리를 위한 Voxel downsampling
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    def sor(self, pcd, nb_neighbors=20, std_ratio=1.0):
        # Statistical Outlier Removal (SOR) 적용
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return pcd.select_by_index(ind)

    def ror(self, pcd, nb_points=6, radius=1.0):
        # Radius Outlier Removal (ROR) 적용
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        return pcd.select_by_index(ind)

    def remove_floor(self, pcd):
        # RANSAC을 사용하여 평면 추정
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.1, ransac_n=3, num_iterations=2000
        )

        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        # 도로에 속하는 포인트 (inliers)
        road_pcd = pcd.select_by_index(inliers)

        # 도로에 속하지 않는 포인트 (outliers)
        non_road_pcd = pcd.select_by_index(inliers, invert=True)
        self.floor_zvalue = road_pcd.get_center()[2]
        print(f"floor.z = {self.floor_zvalue}")
        return non_road_pcd

    def dbscan(self, pcd, eps=0.4, min_points=6, print_progress=False):
        # DBSCAN 클러스터링 적용
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            labels = np.array(
                pcd.cluster_dbscan(
                    eps=eps, min_points=min_points, print_progress=print_progress
                )
            )
        return labels

    def recursive_cluster_matching(self, prev_clusters_centers_frames, curr_clusters_centers, num_history_frames, movement_threshold):
        """
        이전 n 프레임의 클러스터 중심점을 재귀적으로 현재 프레임과 매칭하여 최종 경로와 누적 거리 계산.
        
        Args:
            history_clusters (list): 이전 프레임들의 클러스터 중심점 리스트 (각 프레임별 Nx3 ndarray).
            curr_clusters (list): 현재 프레임의 클러스터 중심점 리스트 (Nx3 ndarray).
            movement_threshold (float): 중심점 간 거리 기준.
            
        Returns:
            matched_paths (dict): 각 현재 클러스터와 매칭된 경로 정보 {curr_idx: [(history_idx, cluster_idx), ...]}.
            unmatched_curr_clusters (list): 매칭되지 않은 현재 클러스터 인덱스 리스트.
            total_distances (dict): 각 현재 클러스터의 누적 거리 정보 {curr_idx: 누적 거리}.
        """

        # 현재 프레임 클러스터를 위한 결과 초기화
        matched_paths = {i: [] for i in range(len(curr_clusters_centers))} # (prev_idx, curr_idx) 형태로 수정
        unmatched_curr_clusters = set(range(len(curr_clusters_centers)))
        displacements = {i: 0 for i in range(len(curr_clusters_centers))}
        prev_indices = []

        if not prev_clusters_centers_frames:
            # 이전 클러스터가 없으면, 모든 현재 클러스터를 새로운 클러스터로 간주
            return matched_paths, unmatched_curr_clusters, displacements

        # 직전 프레임의 클러스터와 매칭
        prev_clusters_centers = prev_clusters_centers_frames[-1]
        if not prev_clusters_centers:
            return matched_paths, unmatched_curr_clusters, displacements
        kdtree = KDTree(prev_clusters_centers)

        for curr_idx, curr_center in enumerate(curr_clusters_centers):
            distance, prev_idx = kdtree.query(curr_center)

            if distance <= movement_threshold and prev_idx > -1 and prev_idx < len(prev_clusters_centers):
                # print(f"Matched: {prev_idx} -> {curr_idx}", f"Distance: {distance}, Current_position: {curr_center}, Initial_position: {prev_clusters[prev_idx]}")
                matched_paths[curr_idx].append((prev_idx, curr_idx))
                unmatched_curr_clusters.discard(curr_idx)
                prev_indices.append(prev_idx)

        # 이전 프레임의 매칭 정보를 재귀적으로 이어가기
        if len(prev_clusters_centers_frames) > 1:
            prev_matched_paths, _, prev_displacements = self.recursive_cluster_matching(
                prev_clusters_centers_frames[:-1],
                prev_clusters_centers,
                num_history_frames,
                movement_threshold
            )

            # 이전 매칭 정보와 현재 매칭 정보를 병합
            # path 형태: [(prev_idx, curr_idx), ...]
            for curr_idx, path in matched_paths.items():
                # 경로 병합
                prev_idx = path[-1][0] if path else None
                if prev_idx in prev_matched_paths:
                    matched_paths[curr_idx] = prev_matched_paths[prev_idx] + path # list concatenation
                    if len(matched_paths[curr_idx]) > num_history_frames - 1:
                        matched_paths[curr_idx].pop(0)

        # 초기 위치 추적
        for curr_idx, path in matched_paths.items():
            path_length = len(path)
            if path_length > 0:
                try:
                    initial_position = prev_clusters_centers_frames[-path_length][matched_paths[curr_idx][0][0]]
                except IndexError:
                    print(f"IndexError: {matched_paths[curr_idx]}")
                    for prev_clusters_center_frame in prev_clusters_centers_frames:
                        print(f"Length: {len(prev_clusters_center_frame)}")
                    initial_position = curr_clusters_centers[curr_idx]

                # 변위 계산
                displacement = np.linalg.norm(curr_clusters_centers[curr_idx] - initial_position)
                displacements[curr_idx] = displacement
                if displacement > 1:
                    print(f"Matched path: {matched_paths[curr_idx]}")
                    print(f"Initial position: {initial_position}")
                    print(f"Positions: ")
                    for idx, (prev_idx, curr_idx) in enumerate(matched_paths[curr_idx]):
                        if idx < path_length - 1:
                            # print(f"frame: {-(path_length) + idx}, idx: {prev_idx} -> {curr_idx}")
                            print(f"{idx}: {prev_clusters_centers_frames[-(path_length) + idx][prev_idx]} -> {prev_clusters_centers_frames[-(path_length)+ idx + 1][curr_idx]}")
                        else:
                            print(f"{idx}: {prev_clusters_centers_frames[-(path_length) + idx][prev_idx]} -> {curr_clusters_centers[curr_idx]}")
                    # print(f"Current position {curr_clusters[curr_idx]}, Initial position {initial_position}")
                    print(f"Displacement: {displacement}")

        return matched_paths, list(unmatched_curr_clusters), displacements

    def get_score(self, cluster_pcd, cluster_indices):
        score = 0

        # 필터링 기준 1. 클러스터 내 최대 최소 포인트 수
        min_points_in_cluster = 8  # 클러스터 내 최소 포인트 수
        max_points_in_cluster = 20  # 클러스터 내 최대 포인트 수

        # 필터링 기준 2. 클러스터 내 최소 최대 Z값
        min_z_value = -1.5  # 클러스터 내 최소 Z값
        max_z_value = self.floor_zvalue + 5.0  # 클러스터 내 최대 Z값

        # 필터링 기준 3. 클러스터 내 최소 최대 Z값 차이
        min_height = 0.5  # Z값 차이의 최소값 (50cm?)
        max_height = 2.5  # Z값 차이의 최대값 (2m?)

        max_distance = 30.0  # 원점으로부터의 최대 거리

        # max_volume = 5.0  # 클러스터의 최대 부피

        # 학습을 통해 조정 가능
        score_weights = {
            "points_in_cluster": 1,
            "z_value_range": 0.1,
            "height_diff": 0.4,
            "distance": 0,
        }

        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            score += score_weights["points_in_cluster"]
            # 부피 계산
            # if len(cluster_indices) >= 4:
            #     volume = cluster_pcd.get_oriented_bounding_box().volume()
            #     score += (1 / volume)

        points = np.asarray(cluster_pcd.points)
        z_values = points[:, 2]  # Z값 추출
        z_min = z_values.min()
        z_max = z_values.max()

        if min_z_value <= z_min and z_max <= max_z_value:
            score += score_weights["z_value_range"]

        height_diff = z_max - z_min
        if min_height <= height_diff <= max_height:
            score += score_weights["height_diff"]

        distances = np.linalg.norm(points, axis=1)
        if distances.max() <= max_distance:
            score += score_weights["distance"]

        return score

    def generate_bbox(self, pcd, labels, matched_paths, unmatched_curr_clusters, displacements):
        """
        누적 거리와 매칭 경로를 기반으로 Bounding Box 생성.
        """
        # 점수 기준
        threshold_score = 10 # 총 점수가 이 값을 넘으면 Bounding Box 생성

        bboxes_scored = []
        displacements_log = {}
        average_displacements = {}
        cluster_pcds = {}
        scores = {}

        # Unmatched clusters scoring
        for cluster_id in unmatched_curr_clusters:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_pcd = pcd.select_by_index(cluster_indices)
            cluster_pcds[cluster_id] = cluster_pcd
            score = self.get_score(cluster_pcd, cluster_indices)
            scores[cluster_id] = score

        # Matched clusters scoring with accumulated distance
        for curr_idx, path in matched_paths.items():
            cluster_indices = np.where(labels == curr_idx)[0]
            cluster_pcd = pcd.select_by_index(cluster_indices)
            cluster_pcds[curr_idx] = cluster_pcd

            # 기본 점수
            score = self.get_score(cluster_pcd, cluster_indices)

            # 누적 거리 기반 점수 추가
            if curr_idx in displacements:
                # print(f"Displacement: {displacements[curr_idx]}")
                score += displacements[curr_idx] * 10.0  # 가중치 조정 가능
                if curr_idx not in displacements_log:
                    displacements_log[curr_idx] = [displacements[curr_idx]]

                displacements_log[curr_idx].append(displacements[curr_idx])
                average_displacements[curr_idx] = sum(displacements_log[curr_idx]) / len(displacements_log[curr_idx])
                score += average_displacements[curr_idx] * 60.0  # 가중치 조정 가능
                if average_displacements[curr_idx] > 0.1:
                    variance = sum([(d - average_displacements[curr_idx]) ** 2 for d in displacements_log[curr_idx]]) / len(displacements_log[curr_idx])
                    score +=  (100.0 / variance)

            scores[curr_idx] = score

        # Threshold에 따라 Bounding Box 생성
        for cluster_id, score in scores.items():
            if score >= threshold_score:
                bbox = cluster_pcds[cluster_id].get_axis_aligned_bounding_box()
                bbox.color = (1, 0, 0)
                bboxes_scored.append(bbox)

        return bboxes_scored

    def bbox_filtering(self, bboxes_sequence):
        # 문제: 이전 scene의 bounding box와 현재 scene의 bounding box를 어떻게 대응시킬 것인가?
        NotImplemented

    def color_clusters(self, pcd, labels):
        # 각 클러스터를 색으로 표시
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")

        # 노이즈를 제거하고 각 클러스터에 색상 지정
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        return pcd

    # pcd 파일 불러오고 시각화하는 함수
    def load_and_visualize_pcd(self, file_path, window_name="PCD TEST", point_size=1.0):
        # pcd 파일 로드
        pcd = o3d.io.read_point_cloud(file_path)
        print(f"Point cloud has {len(pcd.points)} points.")

        # 시각화 설정
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)

        vis.add_geometry(pcd)

        vis.get_render_option().point_size = point_size
        vis.run()
        vis.destroy_window()

    # PCD 파일 불러오기 및 데이터 확인 함수
    def load_and_inspect_pcd(self, file_path):
        # PCD 파일 로드
        pcd = o3d.io.read_point_cloud(file_path)

        # 점 구름 데이터를 numpy 배열로 변환
        points = np.asarray(pcd.points)

        # 점 데이터 개수 및 일부 점 확인
        print(f"Number of points: {len(points)}")
        print("First 5 points:")
        print(points[:5])  # 처음 5개의 점 출력

        # 점의 x, y, z 좌표의 범위 확인
        print("X coordinate range:", np.min(points[:, 0]), "to", np.max(points[:, 0]))
        print("Y coordinate range:", np.min(points[:, 1]), "to", np.max(points[:, 1]))
        print("Z coordinate range:", np.min(points[:, 2]), "to", np.max(points[:, 2]))

    def visualize_with_bounding_boxes(
        self,
        pcd_list,
        bounding_boxes,
        window_name="Filtered Clusters and Bounding Boxes",
        point_size=1.0,
        fps=10,
    ):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1920, height=1080)
        vis.get_render_option().point_size = point_size

        for i in range(len(pcd_list)):
            vis.clear_geometries()
            vis.add_geometry(pcd_list[i])
            if bounding_boxes[i] is not None:
                for bbox in bounding_boxes[i]:
                    vis.add_geometry(bbox)

            vis.poll_events()
            vis.update_renderer()

            sleep(1.0 / fps)

        vis.destroy_window()

    def run(self, num_history_frames=10, movement_threshold=0.6):
        """
        이전 n 프레임을 고려하여 클러스터 매칭 및 경로 추적.
        """
        pcd_path = f"data/{SCENARIOS[self.scenario]}/pcd/"
        pcd_files = os.listdir(pcd_path)
        pcd_files.sort()

        result_pcds = []
        labels_sequence = []
        bboxes_sequence = []
        history_clusters = []  # 히스토리 프레임 클러스터 중심점 리스트

        for pcd_file in pcd_files:
            pcd_file_path = os.path.join(pcd_path, pcd_file)
            pcd = o3d.io.read_point_cloud(pcd_file_path)
            print(f"Point cloud has {len(pcd.points)} points.")

            # 전처리
            voxel_pcd = self.voxel_downsampling(pcd)
            ror_cpd = self.ror(voxel_pcd)
            non_road_pcd = self.remove_floor(ror_cpd)

            labels = self.dbscan(non_road_pcd)
            colored_pcd = self.color_clusters(non_road_pcd, labels)
            result_pcds.append(colored_pcd)

            # 현재 프레임의 클러스터 중심점 계산
            cluster_centers = []
            for i in range(labels.max() + 1):
                cluster_indices = np.where(labels == i)[0]
                cluster_pcd = colored_pcd.select_by_index(cluster_indices)
                cluster_centers.append(np.array(cluster_pcd.get_center()))

            # 클러스터 매칭 및 변위 계산
            matched_paths, unmatched_curr_clusters, displacements = self.recursive_cluster_matching(
                history_clusters, cluster_centers, num_history_frames, movement_threshold
            )

            # 히스토리 업데이트
            history_clusters.append(cluster_centers)
            if len(history_clusters) > num_history_frames:
                history_clusters.pop(0)

            # Bounding Box 생성
            bboxes = self.generate_bbox(
                colored_pcd, labels, matched_paths, unmatched_curr_clusters, displacements
            )
            bboxes_sequence.append(bboxes)

        # 결과 시각화
        self.visualize_with_bounding_boxes(result_pcds, bboxes_sequence, point_size=2.0)
