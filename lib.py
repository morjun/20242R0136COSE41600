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

    def dbscan(self, pcd, eps=0.4, min_points=7, print_progress=False):
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

    def recursive_cluster_matching(
        self, history_clusters_frames, curr_clusters, movement_threshold=0.5
    ):
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
        if not history_clusters_frames:
            # 이전 클러스터가 없으면, 모든 현재 클러스터를 새로운 클러스터로 간주
            return (
                {i: [] for i in range(len(curr_clusters))},
                list(range(len(curr_clusters))),
                {i: 0 for i in range(len(curr_clusters))},
            )

        # 현재 프레임 클러스터를 위한 결과 초기화
        matched_paths = {}
        total_distances = {}
        path_distances = {}
        unmatched_curr_clusters = set(range(len(curr_clusters)))

        # 직전 프레임의 클러스터와 매칭
        prev_clusters = history_clusters_frames[-1]
        kdtree = KDTree(prev_clusters)

        for curr_idx, curr_center in enumerate(curr_clusters):
            distance, prev_idx = kdtree.query(curr_center)

            if distance <= movement_threshold:
                # 초기 매칭 경로 생성
                matched_paths[curr_idx] = [(len(history_clusters_frames) - 1, prev_idx)]

                # 누적 거리 계산
                total_distances[curr_idx] = total_distances.get(curr_idx, 0) + distance

                unmatched_curr_clusters.discard(curr_idx)

        # 이전 프레임의 매칭 정보를 재귀적으로 이어가기
        if len(history_clusters_frames) > 1:
            prev_matched_paths, _, prev_total_distances = (
                self.recursive_cluster_matching(
                    history_clusters_frames[:-1],
                    [prev_clusters[path[-1][1]] for path in matched_paths.values()],
                    movement_threshold,
                )
            )

            # 이전 매칭 정보와 현재 매칭 정보를 병합
            for curr_idx, path in matched_paths.items():
                # 경로 병합
                if curr_idx in prev_matched_paths:
                    matched_paths[curr_idx] = (
                        prev_matched_paths[curr_idx] + path
                    )  # list concatenation
                # 거리 병합
                if curr_idx in prev_total_distances:
                    total_distances[curr_idx] += prev_total_distances.get(curr_idx, 0)

        return matched_paths, list(unmatched_curr_clusters), total_distances

    def get_score(self, cluster_pcd, cluster_indices):
        score = 0

        # 필터링 기준 1. 클러스터 내 최대 최소 포인트 수
        min_points_in_cluster = 5  # 클러스터 내 최소 포인트 수
        max_points_in_cluster = 20  # 클러스터 내 최대 포인트 수

        # 필터링 기준 2. 클러스터 내 최소 최대 Z값
        min_z_value = -1.5  # 클러스터 내 최소 Z값
        max_z_value = self.floor_zvalue + 5.0  # 클러스터 내 최대 Z값

        # 필터링 기준 3. 클러스터 내 최소 최대 Z값 차이
        min_height = 0.5  # Z값 차이의 최소값 (50cm?)
        max_height = 2.0  # Z값 차이의 최대값 (2m?)

        max_distance = 30.0  # 원점으로부터의 최대 거리

        # 학습을 통해 조정 가능
        score_weights = {
            "points_in_cluster": 1,
            "z_value_range": 0.5,
            "height_diff": 0.5,
            "distance": 0,
        }

        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            score += score_weights["points_in_cluster"]

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

    def generate_bbox(
        self, pcd, labels, matched_paths, unmatched_curr_clusters, total_distances
    ):
        """
        누적 거리와 매칭 경로를 기반으로 Bounding Box 생성.
        """
        # 점수 기준
        threshold_score = 3  # 총 점수가 이 값을 넘으면 Bounding Box 생성

        bboxes_scored = []
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
            if curr_idx in total_distances:
                score += total_distances[curr_idx] * 6  # 가중치 조정 가능
                if total_distances[curr_idx] > 0.3:
                    print(
                        f"Cluster {curr_idx} total distance: {total_distances[curr_idx]:.2f}"
                    )

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
        vis.create_window(window_name=window_name, width=1280, height=720)
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

    def run(self, history_frames=10, movement_threshold=0.5):
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

            # 클러스터 매칭 및 누적 거리 계산
            matched_paths, unmatched_curr_clusters, total_distances = (
                self.recursive_cluster_matching(
                    history_clusters, cluster_centers, movement_threshold
                )
            )

            # 히스토리 업데이트
            history_clusters.append(cluster_centers)
            if len(history_clusters) > history_frames:
                history_clusters.pop(0)

            # Bounding Box 생성
            bboxes = self.generate_bbox(
                colored_pcd,
                labels,
                matched_paths,
                unmatched_curr_clusters,
                total_distances,
            )
            bboxes_sequence.append(bboxes)

        # 결과 시각화
        self.visualize_with_bounding_boxes(result_pcds, bboxes_sequence, point_size=2.0)
