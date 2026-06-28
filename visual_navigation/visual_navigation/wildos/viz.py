# 本文件实现 WildOS 目标无关前沿评分的调试可视化。
# 它把导航图节点、边、前沿点、探索半径和多朝向评分转换成 RViz Marker 或图像面板，
# 用于检查视觉评分是否被正确写入导航图，以及不同朝向 bin 的评分是否符合预期。

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

import numpy as np
import cv2
import matplotlib.pyplot as plt

from visual_navigation.geofrontier_nav.viz import VisualizeGeoFrontierScoring
from visual_navigation.utils.viz import (
    make_subplot_grid, overlay_heatmap, draw_point, draw_text, draw_path, make_colorbar, pad_image, show_mask
)


class VisualizeGoalAgnosticGeoFrontierScoring(VisualizeGeoFrontierScoring):
    """
    WildOS 目标无关评分可视化器。

    父类已经提供几何前沿评分的通用绘图能力，本类额外处理按角度离散的评分环：
    每个导航图前沿节点可以携带多个朝向 bin 的视觉分数，RViz 中会显示前沿点、节点连线、
    自由半径和探索半径，图像调试界面则显示当前相机朝向对应的评分热力图。
    """

    def __init__(
            self,
            angular_bins: np.ndarray,
            flatten_nav_graph_viz: bool = True,
            nav_graph_viz_z: float = 0.08,
            **kwargs,
    ):
        """
        初始化角度分桶和 Marker ID 状态。

        参数：
            angular_bins: 每个目标无关评分方向 bin 的起始角，单位为弧度。
            flatten_nav_graph_viz: 是否将 NavigationGraph 的 RViz 调试图压平成 XY 平面。
            nav_graph_viz_z: 压平显示时使用的固定 Z 高度，单位为米。
            **kwargs: 传给 `VisualizeGeoFrontierScoring` 的通用可视化配置。
        """
        super().__init__(**kwargs)
        self.discretization_angle = 2 * np.pi / len(angular_bins)
        self.bin_starts = angular_bins
        self.num_bins = len(angular_bins)
        self.flatten_nav_graph_viz = bool(flatten_nav_graph_viz)
        self.nav_graph_viz_z = float(nav_graph_viz_z)

        # 图像调试时，每个相机只显示与当前相对朝向最接近的评分 bin。
        self.goal_cam_relative_headings = 0.0
        # 评分环位于导航图世界坐标系中，因此半径单位是米，而不是图像像素。
        self.ring_radius = 1.0

        # 每个“前沿 UUID + 方向 bin”必须长期复用同一个 Marker ID，RViz 才能执行
        # MODIFY/DELETE，而不会在连续帧中叠加出重复评分环。
        self.marker_id = 0
        self.uuid_to_marker_id = {}

    def nav_graph_display_point(self, point, z_offset: float = 0.0) -> Point:
        """返回 `/nav_graph_viz` 使用的显示点；可选把真实 2.5D 高度压平成 XY 平面。"""
        z = self.nav_graph_viz_z if self.flatten_nav_graph_viz else point.z
        return Point(
            x=float(point.x),
            y=float(point.y),
            z=float(z + z_offset),
        )

    def visualize_navgraph(self, navgraph_msg, frame_id, stamp):
        """
        将导航图消息转换为 RViz MarkerArray。

        参数：
            navgraph_msg: 已包含节点、边、前沿属性和视觉评分的导航图消息。
            frame_id: Marker 所在坐标系。
            stamp: Marker 时间戳，通常沿用导航图消息时间。

        返回：
            MarkerArray: 包含普通节点、前沿节点、边、探索半径、自由半径和评分文本/颜色标记。
        """
        marker_array = MarkerArray()

        # 普通节点合并到一个 SPHERE_LIST 中发布，显著减少 Marker 数量和 ROS 序列化开销。
        node_marker = Marker()
        node_marker.header.frame_id = str(frame_id)
        node_marker.header.stamp = stamp
        node_marker.frame_locked = True
        node_marker.ns = "nodes"
        node_marker.id = 0
        node_marker.type = Marker.SPHERE_LIST
        node_marker.action = Marker.ADD
        node_marker.scale.x = 0.5
        node_marker.scale.y = 0.5
        node_marker.scale.z = 0.5
        node_marker.color = ColorRGBA()
        node_marker.color.a = 1.0
        node_marker.color.r = 0.0
        node_marker.color.g = 1.0
        node_marker.color.b = 0.0

        # 不同 traversability class 分别建立前沿节点 Marker，方便在多类别导航图中独立着色。
        frontier_node_markers = {}

        # frontier_points 表示节点附近真实探索边界采样点；LINE_LIST 将这些点连接回所属节点，
        # 用于确认前沿方向、节点中心和后续默认高斯方向评分是否一致。
        frontier_point_marker = Marker()
        frontier_point_marker.header.frame_id = str(frame_id)
        frontier_point_marker.header.stamp = stamp
        frontier_point_marker.frame_locked = True
        frontier_point_marker.ns = "frontier_points"
        frontier_point_marker.id = 0
        frontier_point_marker.type = Marker.CUBE_LIST
        frontier_point_marker.action = Marker.ADD
        frontier_point_marker.scale.x = 0.3
        frontier_point_marker.scale.y = 0.3
        frontier_point_marker.scale.z = 0.3
        frontier_point_marker.color = ColorRGBA()
        frontier_point_marker.color.a = 1.0
        frontier_point_marker.color.r = 1.0
        frontier_point_marker.color.g = 0.0
        frontier_point_marker.color.b = 1.0

        frontier_point_to_node_marker = Marker()
        frontier_point_to_node_marker.header.frame_id = str(frame_id)
        frontier_point_to_node_marker.header.stamp = stamp
        frontier_point_to_node_marker.frame_locked = True
        frontier_point_to_node_marker.ns = "frontier_point_to_node"
        frontier_point_to_node_marker.id = 0
        frontier_point_to_node_marker.type = Marker.LINE_LIST
        frontier_point_to_node_marker.action = Marker.ADD
        frontier_point_to_node_marker.scale.x = 0.06
        frontier_point_to_node_marker.color = ColorRGBA()
        frontier_point_to_node_marker.color.a = 1.0
        frontier_point_to_node_marker.color.b = 1.0

        # 每个命名空间的 Marker ID 必须唯一。free radius 按可通行性类别分别计数，
        # explored radius 则共用一个命名空间和计数器。
        free_radius_counters = {}
        explored_radius_counter = 0

        # 每帧先清空旧 explored_radius，避免节点删除或数量减少后遗留历史球体。
        explored_radius_reset = Marker()
        explored_radius_reset.header.frame_id = str(frame_id)
        explored_radius_reset.header.stamp = stamp
        explored_radius_reset.ns = "explored_radius"
        explored_radius_reset.id = explored_radius_counter
        explored_radius_reset.action = Marker.DELETEALL
        marker_array.markers.append(explored_radius_reset)

        # edges 显示导航图拓扑；relative_positions 预留用于显示相对位姿约束。
        # 两者使用不同命名空间，便于在 RViz 中单独开关。
        edge_marker = Marker()
        edge_marker.header.frame_id = str(frame_id)
        edge_marker.header.stamp = stamp
        edge_marker.frame_locked = True
        edge_marker.ns = "edges"
        edge_marker.id = 1
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.01
        edge_marker.color = ColorRGBA()
        edge_marker.color.a = 1.0
        edge_marker.color.r = 1.0
        edge_marker.color.g = 0.0
        edge_marker.color.b = 0.0

        rel_pos_marker = Marker()
        rel_pos_marker.header.frame_id = str(frame_id)
        rel_pos_marker.header.stamp = stamp
        rel_pos_marker.frame_locked = True
        rel_pos_marker.ns = "relative_positions"
        rel_pos_marker.id = 2
        rel_pos_marker.type = Marker.LINE_LIST
        rel_pos_marker.action = Marker.ADD
        rel_pos_marker.scale.x = 0.05
        rel_pos_marker.color = ColorRGBA()
        rel_pos_marker.color.a = 1.0
        rel_pos_marker.color.r = 0.0
        rel_pos_marker.color.g = 0.0
        rel_pos_marker.color.b = 1.0

        # Edge 消息只保存节点索引，先缓存索引到显示位置的映射，后续构造线段时无需重复取值。
        node_positions = []
        for node in navgraph_msg.nodes:
            p = node.pose.position
            node_positions.append(self.nav_graph_display_point(p))

        # 遍历节点及其每个可通行性类别属性，构造节点、半径和前沿点 Marker。
        for idx, node in enumerate(navgraph_msg.nodes):
            p = node.pose.position
            node_position = self.nav_graph_display_point(p)

            for trav_idx, trav_prop in enumerate(node.trav_properties):
                try:
                    trav_class = str(navgraph_msg.trav_classes[trav_idx])
                except Exception:
                    trav_class = str(f"class_{trav_idx}")

                # 只把非前沿节点加入绿色普通节点列表；前沿节点由分类 Marker 单独展示。
                if not trav_prop.is_frontier:
                    node_marker.points.append(node_position)

                # explored_radius 表示该节点周围已观测/已建图区域。使用扁平半透明球体，
                # 从俯视角看近似二维圆盘，同时不遮挡节点和边。
                explored_radius = Marker()
                explored_radius.header.frame_id = str(frame_id)
                explored_radius.header.stamp = stamp
                explored_radius.ns = "explored_radius"
                explored_radius.id = explored_radius_counter + 1
                explored_radius_counter += 1
                explored_radius.action = Marker.ADD
                explored_radius.type = Marker.SPHERE
                explored_radius.scale.x = getattr(trav_prop, 'explored_radius', 0.0) * 2.0
                explored_radius.scale.y = getattr(trav_prop, 'explored_radius', 0.0) * 2.0
                explored_radius.scale.z = 0.01
                explored_radius.color = ColorRGBA()
                explored_radius.color.a = 0.2
                explored_radius.color.r = 0.0
                explored_radius.color.g = 0.0
                explored_radius.color.b = 1.0
                # 半径圆盘以导航图节点位置为中心。
                explored_radius.pose.position = node_position
                marker_array.markers.append(explored_radius)

                # free_radius 表示该类别下节点周围可安全通行的局部范围。
                if trav_class not in free_radius_counters:
                    # 每个类别首次出现时清空对应命名空间，避免上一帧多余 Marker 残留。
                    free_radius_reset = Marker()
                    free_radius_reset.header.frame_id = str(frame_id)
                    free_radius_reset.header.stamp = stamp
                    free_radius_reset.ns = f"free_radius_{trav_class}"
                    free_radius_reset.id = 0
                    free_radius_reset.action = Marker.DELETEALL
                    marker_array.markers.append(free_radius_reset)
                    free_radius_counters[trav_class] = 1

                free_radius = Marker()
                free_radius.header.frame_id = str(frame_id)
                free_radius.header.stamp = stamp
                free_radius.ns = f"free_radius_{trav_class}"
                free_radius.id = free_radius_counters[trav_class]
                free_radius_counters[trav_class] += 1
                free_radius.action = Marker.ADD
                free_radius.type = Marker.SPHERE
                free_radius.scale.x = getattr(trav_prop, 'free_radius', 0.0) * 2.0
                free_radius.scale.y = getattr(trav_prop, 'free_radius', 0.0) * 2.0
                free_radius.scale.z = 0.01
                free_radius.color = ColorRGBA()
                free_radius.color.a = 0.2
                free_radius.color.r = 1.0
                free_radius.color.g = 0.0
                free_radius.color.b = 0.0
                free_radius.pose.position = node_position
                marker_array.markers.append(free_radius)

                # 前沿节点按可通行性类别聚合。当前统一使用蓝色，命名空间仍保留类别信息，
                # 便于后续在 RViz 中筛选或扩展为不同颜色。
                if trav_prop.is_frontier:
                    if trav_class not in frontier_node_markers:
                        m = Marker()
                        m.header.frame_id = str(frame_id)
                        if stamp is not None:
                            m.header.stamp = stamp
                        m.frame_locked = True
                        m.ns = f"frontier_nodes_{trav_class}"
                        m.id = 0
                        m.type = Marker.SPHERE_LIST
                        m.action = Marker.ADD
                        m.scale.x = 0.6
                        m.scale.y = 0.6
                        m.scale.z = 0.6
                        m.color = ColorRGBA()
                        m.color.a = 1.0
                        m.color.b = 1.0
                        frontier_node_markers[trav_class] = m
                    frontier_node_markers[trav_class].points.append(node_position)
                    # SPHERE_LIST 支持逐点颜色；即使当前颜色一致，也需与 points 数量一一对应。
                    c = ColorRGBA()
                    c.a = 1.0
                    c.b = 1.0
                    frontier_node_markers[trav_class].colors.append(c)

                # 每个前沿采样点既加入紫色方块列表，也追加“节点 -> 前沿点”的两端点线段。
                for fp in trav_prop.frontier_points:
                    fp_point = self.nav_graph_display_point(fp)
                    frontier_point_marker.points.append(fp_point)
                    frontier_point_to_node_marker.points.append(node_position)
                    frontier_point_to_node_marker.points.append(fp_point)

        # 导航图边通过 from_idx/to_idx 引用节点；越界或暂时不一致的边跳过，避免可视化
        # 阻断主导航回调。
        for edge in navgraph_msg.edges:
            try:
                from_p = node_positions[edge.from_idx]
                to_p = node_positions[edge.to_idx]
            except Exception:
                continue
            edge_marker.points.append(from_p)
            edge_marker.points.append(to_p)

        # 相对位置线是兼容旧版导航图可视化的预留结构。当前消息未提供独立参考点时，
        # 线段两端相同，因此不会产生可见长度，但保留命名空间以兼容现有 RViz 配置。
        for node in navgraph_msg.nodes:
            try:
                ref_p = node.pose.position
                node_p = node.pose.position
                ref_pt = self.nav_graph_display_point(ref_p)
                node_pt = self.nav_graph_display_point(node_p)
                rel_pos_marker.points.append(ref_pt)
                rel_pos_marker.points.append(node_pt)
            except Exception:
                pass

        # UUID 文本用于在 RViz 中把评分环、日志和具体导航图节点对应起来。
        # 先 DELETEALL，再按当前节点集合重建，避免节点移除后文字残留。
        if hasattr(navgraph_msg, 'nodes'):
            text_delete_all = Marker()
            text_delete_all.header.frame_id = frame_id
            if stamp is not None:
                text_delete_all.header.stamp = stamp
            text_delete_all.frame_locked = True
            text_delete_all.ns = "ids"
            text_delete_all.id = 0
            text_delete_all.action = Marker.DELETEALL
            marker_array.markers.append(text_delete_all)
            id_counter = 1
            for node in navgraph_msg.nodes:
                try:
                    p = node.pose.position
                    pos_pt = self.nav_graph_display_point(p)
                    text_marker = Marker()
                    text_marker.header.frame_id = frame_id
                    if stamp is not None:
                        text_marker.header.stamp = stamp
                    text_marker.frame_locked = True
                    text_marker.ns = "ids"
                    text_marker.id = id_counter
                    id_counter += 1
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD
                    text_marker.scale.z = 0.05

                    try:
                        uuid_val = node.uuid
                        text_marker.text = "".join([f"{x:03}" for x in uuid_val.id])
                    except Exception:
                        text_marker.text = ""
                    text_marker.color = ColorRGBA()
                    text_marker.color.a = 1.0
                    text_marker.color.r = 1.0
                    text_marker.color.g = 1.0
                    text_marker.color.b = 1.0
                    text_marker.pose.position.x = pos_pt.x
                    text_marker.pose.position.y = pos_pt.y
                    text_marker.pose.position.z = pos_pt.z + 0.1
                    marker_array.markers.append(text_marker)
                except Exception:
                    continue

        # 最后追加聚合型 Marker；逐节点半径和文本 Marker 已在遍历过程中直接加入数组。
        marker_array.markers.append(node_marker)
        for _, m in frontier_node_markers.items():
            marker_array.markers.append(m)
        marker_array.markers.append(frontier_point_marker)
        marker_array.markers.append(frontier_point_to_node_marker)
        marker_array.markers.append(edge_marker)
        marker_array.markers.append(rel_pos_marker)

        return marker_array

    def visualize_all_heading_scores(self, frontier_uuid_to_scores, removed_uuids, updated_uuids, frame_id, stamp):
        """
        为每个前沿节点生成一圈按朝向分段的评分 Marker。

        参数：
            frontier_uuid_to_scores: UUID 到 `(scores, node)` 的映射，`scores` 长度等于角度 bin 数。
            removed_uuids: 本轮被移除的前沿 UUID，函数会发布 DELETE marker 清理旧评分环。
            updated_uuids: 本轮新增或更新的前沿 UUID。
            frame_id: Marker 所在坐标系。
            stamp: Marker 时间戳。

        返回：
            MarkerArray: 每个朝向 bin 使用一段 LINE_STRIP 表示，颜色由评分映射得到。
        """
        marker_array = MarkerArray()

        # 对已从导航图消失的前沿，逐 bin 发布 DELETE。使用历史 ID 而不是新 ID，
        # 才能精确删除 RViz 中对应的评分弧段。
        for uuid in removed_uuids:
            for i in range(self.num_bins):
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = stamp
                marker.ns = "geofrontier_score_ring"
                marker.id = self.uuid_to_marker_id[uuid][i]
                marker.action = Marker.DELETE
                marker_array.markers.append(marker)

        # 新前沿分配连续 ID，已有前沿沿用原 ID 并发布 MODIFY，保证动画更新稳定。
        for uuid in updated_uuids:
            scores, node = frontier_uuid_to_scores[uuid]

            node_pos = np.array([
                node.pose.position.x,
                node.pose.position.y,
                node.pose.position.z
            ])

            marker_action = Marker.MODIFY
            if uuid not in self.uuid_to_marker_id:
                self.uuid_to_marker_id[uuid] = np.arange(self.marker_id, self.marker_id + self.num_bins).tolist()
                self.marker_id += self.num_bins
                marker_action = Marker.ADD

            for bin_idx, score in enumerate(scores):
                # Jet 色表仅用于调试：低分偏蓝，高分偏红；不会反馈到规划分数。
                color = plt.cm.jet(score)
                angle_st = self.bin_starts[bin_idx]
                angle_end = angle_st + self.discretization_angle

                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = stamp
                marker.ns = "geofrontier_score_ring"
                marker.action = marker_action
                marker.id = self.uuid_to_marker_id[uuid][bin_idx]
                marker.type = Marker.LINE_STRIP

                start_pt = node_pos + self.ring_radius * np.array([np.cos(angle_st), np.sin(angle_st), 0])
                end_pt = node_pos + self.ring_radius * np.array([np.cos(angle_end), np.sin(angle_end), 0])
                start_pt = start_pt.astype(np.float64)
                end_pt = end_pt.astype(np.float64)
                marker.points.append(Point(x=start_pt[0], y=start_pt[1], z=start_pt[2]))
                marker.points.append(Point(x=end_pt[0], y=end_pt[1], z=end_pt[2]))

                marker.scale.x = 0.5  # LINE_STRIP 使用 scale.x 表示线宽，单位为米。
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 1.0
                marker_array.markers.append(marker)

        return marker_array

    def visualize_model_det_front(self, nav_data, all_cam_data):
        """
        生成紧凑的前向相机检测调试图。

        参数：
            nav_data: 每个相机的图像、前沿图、可通行性图和可选目标掩码。
            all_cam_data: 保留的接口参数，当前函数主要使用 `nav_data`。

        返回：
            np.ndarray: 拼接后的调试图，包含前沿叠加、可通行性叠加和可选目标掩码。

        说明：
            该视图只展示最关键的前向检测结果，适合在线调试时减少屏幕占用。
        """
        img_grid = {}
        num_rows = 1
        num_cols = 2
        fig_resize_factor = 0.75

        rgb_img = nav_data[0]["image"]
        # rgb_img = cv2.resize(rgb_img, (0,0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
        frontier_overlay = rgb_img.copy()
        trav_overlay = rgb_img.copy()

        frontier_map = nav_data[0]["img_frontiers"].astype(np.float32)
        traversability_map = nav_data[0]["traversability"].astype(np.float32)

        frontier_map[frontier_map < 0.6] = 0.0
        frontier_map[traversability_map < 0.9] = 0.0

        # 先做形态学开运算去除面积较小的孤立前沿响应，避免在线总览图被噪声点覆盖。
        # 这里只影响可视化，不会修改导航评分器使用的原始概率图。
        kernel = np.ones((20, 20), np.uint8)
        valid = cv2.morphologyEx((frontier_map > 0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        frontier_map[valid == 0] = 0.0

        frontier_hm = overlay_heatmap(rgb_img, frontier_map, alpha=1.0)
        valid_frontier = (frontier_map > 0)
        valid_mask_3d = np.stack([valid_frontier] * 3, axis=-1)
        frontier_overlay[valid_mask_3d] = frontier_hm[valid_mask_3d]
        frontier_overlay = cv2.resize(frontier_overlay, (0, 0), fx=fig_resize_factor, fy=fig_resize_factor)
        img_grid[(0, 0)] = (frontier_overlay, "Frontiers Overlay")
        # return frontier_overlay

        # 可通行性总览只显示高置信区域；热力图使用 1-probability，使危险区域颜色更醒目。
        traversability_map[traversability_map < 0.9] = 0.0

        trav_hm = overlay_heatmap(rgb_img, 1 - traversability_map, alpha=0.5)
        valid_traversability = (traversability_map > 0)
        valid_mask_3d = np.stack([valid_traversability] * 3, axis=-1)
        trav_overlay[valid_mask_3d] = trav_hm[valid_mask_3d]
        trav_overlay = cv2.resize(trav_overlay, (0, 0), fx=fig_resize_factor, fy=fig_resize_factor)
        img_grid[(0, 1)] = (trav_overlay, "Traversability Overlay")
        # return trav_overlay

        fin_img_chosen = None
        if "object_mask" in nav_data[0] and nav_data[0]["object_mask"] is not None:
            # 多相机中选择目标掩码像素最多的视角作为目标检测总览，避免固定展示前相机
            # 而目标实际只出现在侧相机时给出空白结果。
            chosen_cam_idx = -1
            max_obj_pix = -1
            for i in range(self.num_cameras):
                img = nav_data[i]["image"]
                H, W, C = img.shape

                obj_mask_2d = nav_data[i]["object_mask"].squeeze()
                valid_mask_2d = obj_mask_2d > 0
                num_obj_pixels = np.sum(valid_mask_2d)

                if num_obj_pixels <= max_obj_pix:
                    continue

                fin_img = show_mask(img, obj_mask_2d)

                DARK_GRAY = (50, 50, 50)
                WHITE = (255, 255, 255)
                FONT = cv2.FONT_HERSHEY_DUPLEX
                FONT_SCALE = 0.7
                FONT_THICKNESS = 1
                PADDING = 10  # 文本与深灰背景框边缘之间的像素留白。
                text_to_display = f"Current View: {self.camera_mapping[i].title()} Camera"
                (text_w, text_h), baseline = cv2.getTextSize(
                    text_to_display, FONT, FONT_SCALE, FONT_THICKNESS
                )

                text_x = PADDING
                text_y = PADDING + text_h

                p1 = (PADDING, PADDING)  # 文本背景左上角。
                p2 = (PADDING + text_w + PADDING, PADDING + text_h + baseline + PADDING)  # 右下角。

                cv2.rectangle(fin_img, p1, p2, DARK_GRAY, -1)  # thickness=-1 表示填充矩形。

                cv2.putText(
                    fin_img,
                    text_to_display,
                    (text_x, text_y),
                    FONT,
                    FONT_SCALE,
                    WHITE,
                    FONT_THICKNESS,
                    cv2.LINE_AA
                )

                max_obj_pix = num_obj_pixels
                chosen_cam_idx = i
                fin_img_chosen = fin_img

            if fin_img_chosen is not None:
                fin_img_chosen = cv2.resize(fin_img_chosen, (0, 0), fx=fig_resize_factor, fy=fig_resize_factor)
                img_grid[(0, 2)] = (fin_img_chosen, "Object Detection Overlay")
                num_cols = 3

        grid = make_subplot_grid(img_grid, (num_rows, num_cols), pad=15)

        return grid

    def visualize_model_det(self, nav_data, all_cam_data):
        """
        生成多相机完整模型检测和评分调试图。

        参数：
            nav_data: 每个相机的原图、目标掩码、前沿图、可通行性图和评分图。
            all_cam_data: 多相机原始数据，保留给父类接口和后续扩展。

        返回：
            np.ndarray: 拼接后的 RGB 调试图像。单相机时使用 2x2 布局，多相机时按相机列显示。

        说明：
            单相机时四个面板分别为原图/目标掩码、前沿置信度、可通行性和朝向评分；
            多相机时每列对应一个相机，行内展示同样四类信息，便于横向对比。
        """
        img_grid = {}
        num_rows = 4
        num_cols = self.num_cameras

        for i in range(self.num_cameras):
            plt_idx = self.cam_order[i]

            rgb_img = nav_data[i]["image"]
            rgb_img = cv2.resize(rgb_img, (0, 0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            img_grid[(0, plt_idx)] = (rgb_img, f"Image {self.camera_mapping[i]}")

            if "object_mask" in nav_data[i] and nav_data[i]["object_mask"] is not None:
                obj_mask = nav_data[i]["object_mask"].astype(np.float32)
                obj_mask = cv2.resize(obj_mask, (0, 0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
                mask_overlay = show_mask(rgb_img, obj_mask)
                img_grid[(0, plt_idx)] = (mask_overlay, f"Image {self.camera_mapping[i]} + Obj Mask")

            # 前沿概率直接作为热力图叠加，便于检查网络响应是否贴合深度视野边界。
            frontier_map = nav_data[i]["img_frontiers"].astype(np.float32)
            frontier_map = cv2.resize(frontier_map, (0, 0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            frontier_overlay = overlay_heatmap(rgb_img, frontier_map)
            img_grid[(1, plt_idx)] = (frontier_overlay, "Frontier Conf.")

            # 可通行性概率使用相同尺寸缩放后叠加，确保多相机各列布局一致。
            traversability_map = nav_data[i]["traversability"].astype(np.float32)
            traversability_map = cv2.resize(traversability_map, (0, 0), fx=self.fig_resize_factor,
                                            fy=self.fig_resize_factor)
            trav_overlay = overlay_heatmap(rgb_img, traversability_map)
            img_grid[(2, plt_idx)] = (trav_overlay, "Traversability Conf.")

            # 最后一行展示几何前沿投影、当前相机朝向对应的评分热力图以及 MCP 可达路径。
            path_overlay = rgb_img.copy()
            if "geo_frontiers" not in nav_data[i]:
                dummy_img = path_overlay
                cv2.putText(
                    dummy_img,
                    "No Valid Geometric Frontiers",
                    (20, dummy_img.shape[0] // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.45,
                    (0, 0, 0),
                    0
                )
                img_grid[(3, plt_idx)] = (dummy_img, "Frontier Nodes")
                continue

            geo_frontiers = nav_data[i]["geo_frontiers"] * self.fig_resize_factor
            cam_heading = all_cam_data[i]["R_wc"].astype(np.float32) @ np.array([0, 0, 1], dtype=np.float32)
            cam_heading = cam_heading[:2]
            cam_heading = cam_heading / np.linalg.norm(cam_heading)
            cam_angle = np.arctan2(cam_heading[1], cam_heading[0])
            goal_heading = cam_angle + self.goal_cam_relative_headings
            # 将世界系相机朝向量化到评分器使用的方向 bin，随后只显示该切片，
            # 避免把 [H,W,num_bins] 全部方向分数同时叠加到一张图。
            heading_bin = int(goal_heading / self.discretization_angle) % len(self.bin_starts)

            scores = nav_data[i]["scores"]
            paths = nav_data[i]["paths"]
            score_map = nav_data[i]["score_map"][0][:, :, heading_bin].astype(np.float32)
            score_map = cv2.resize(score_map, (0, 0), fx=self.fig_resize_factor, fy=self.fig_resize_factor)
            path_overlay_hm = overlay_heatmap(path_overlay, score_map, alpha=0.5)
            valid_map = (score_map > 0)
            valid_mask_3d = np.stack([valid_map] * 3, axis=-1)
            path_overlay[valid_mask_3d] = path_overlay_hm[valid_mask_3d]

            for ((y, x), score, path) in zip(geo_frontiers, scores, paths):
                path = np.array(path[heading_bin]) * self.fig_resize_factor
                color = plt.cm.jet(score[heading_bin])
                color = tuple(int(c * 255) for c in color[:3])

                # 调试时曾显示路径末端和数值文本；当前只保留路径及红色几何前沿起点，
                # 以减少多前沿重叠时的画面遮挡。
                # draw_point(path_overlay, (y,x), color, radius=6)
                # draw_point(path_overlay, path[-1], (255,255,255), radius=2)  # goal point
                # draw_text(path_overlay, (y,x), f"{score[heading_bin]:.2f}", color=(255,255,255))
                draw_path(path_overlay, path, color)
                draw_point(path_overlay, (y, x), (0, 0, 255), radius=8)
            img_grid[(3, plt_idx)] = (path_overlay, "Frontier Nodes")

        if self.num_cameras == 1:
            # 单相机时 4x1 竖排在 RViz Image 面板里会非常窄，重排为 2x2 更容易观察。
            img_grid = {
                (0, 0): img_grid[(0, 0)],
                (0, 1): img_grid[(1, 0)],
                (1, 0): img_grid[(2, 0)],
                (1, 1): img_grid[(3, 0)],
            }
            num_rows = 2
            num_cols = 2

        grid = make_subplot_grid(img_grid, (num_rows, num_cols), pad=15)
        cbar = make_colorbar(
            height=grid.shape[0] - 100,
            width=20,
            vmin=0,
            vmax=1,
            cmap=cv2.COLORMAP_JET,
            num_ticks=10,
            font_scale=0.5,
            pad=50
        )
        grid = cv2.hconcat([grid, cbar])

        return grid
