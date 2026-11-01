import os
import json
import argparse
import numpy as np
from typing import List
from pyquaternion import Quaternion
from shapely.geometry import LineString, Polygon, Point
import networkx as nx
from scipy.interpolate import interp1d
import torch

from nuscenes.nuscenes import NuScenes
import lanelet2
from lanelet2.projection import UtmProjector
from lanelet2.io import Origin

# from attack_toolkit.src.utils.utils_attack import chamfer_distance

np.set_printoptions(precision=3, suppress=True)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Search Region Parameters
SEARCH_X_RANGE = 30
SEARCH_Y_RANGE = 60

# Route Finding Parameters
MAX_ROUTE_DEPTH = 100
ROUTE_X_RANGE = (-15, 15)
ROUTE_Y_RANGE = (0, 30)
ROUTE_INTERVAL = 0.5
DIST2BOUNDARY = 0.5

# Goal Selection Parameters
NUM_GOALS = 3
DIST2RANGE = 2.0

# Map Origins for different locations
MAP_ORIGINS = {
    'boston-seaport': [42.336849169438615, -71.05785369873047],
    'singapore-onenorth': [1.2882100868743724, 103.78475189208984],
    'singapore-hollandvillage': [1.2993652317780957, 103.78217697143555],
    'singapore-queenstown': [1.2782562240223188, 103.76741409301758]
}


# =============================================================================
# COORDINATE TRANSFORMATION UTILITIES
# =============================================================================

def global_to_lidar(points, nusc, sample_token):
    """Convert points from global coordinates to lidar coordinates."""
    def get_matrix(calibrated_data, inverse=False):
        output = np.eye(4)
        output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
        output[:3,  3] = calibrated_data["translation"]
        if inverse:
            output = np.linalg.inv(output)
        return output
    
    sample = nusc.get('sample', sample_token)
    pointsensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    global_to_ego = get_matrix(poserecord, inverse=True)
    
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    ego_to_lidar = get_matrix(cs_record, inverse=True)
    
    global_to_lidar_transform = ego_to_lidar @ global_to_ego
    
    hom_points = np.hstack((points, np.ones((points.shape[0], 1))))
    lidar_points = hom_points @ global_to_lidar_transform.T
    
    return lidar_points[:, :-1]

def lidar_to_global(points, nusc, sample_token, ground=False):
    """Convert points from lidar coordinates to global coordinates."""
    def get_matrix(calibrated_data, inverse=False):
        output = np.eye(4)
        output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
        output[:3,  3] = calibrated_data["translation"]
        if inverse:
            output = np.linalg.inv(output)
        return output

    sample = nusc.get('sample', sample_token)
    pointsensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    lidar_to_ego = get_matrix(cs_record)

    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    ego_to_global = get_matrix(poserecord)

    lidar_to_global_transform = ego_to_global @ lidar_to_ego
    
    hom_points = np.hstack((points, np.ones((points.shape[0], 1))))
    global_points = hom_points @ lidar_to_global_transform.T
    
    if ground:
        global_points[:, 2] = 0
    
    return global_points[:, :-1]

def is_in_range(points, x_range=(-15, 15), y_range=(0, 30), mode='any'):
    """Check if points are within specified ranges."""
    x_in_range = (x_range[0] <= points[:, 0]) & (points[:, 0] <= x_range[1])
    y_in_range = (y_range[0] <= points[:, 1]) & (points[:, 1] <= y_range[1])
    points_in_range = x_in_range & y_in_range
    
    if mode == 'any':
        return points_in_range.any()
    elif mode == 'all':
        return points_in_range.all()
    
def chamfer_distance(p1, p2, device=None):
    """Calculate chamfer distance between two point sets.
    
    Args:
        p1 (np.ndarray or torch.Tensor): First point set of shape (n1, d)
        p2 (np.ndarray or torch.Tensor): Second point set of shape (n2, d)
        device (torch.device, optional): Device to use if converting to torch.Tensor
    
    Returns:
        float: Chamfer distance between the two point sets
    """
    # Convert to torch tensors if they aren't already
    if not isinstance(p1, torch.Tensor):
        p1 = torch.from_numpy(p1).to(device)
    if not isinstance(p2, torch.Tensor):
        p2 = torch.from_numpy(p2).to(device)
    
    # Compute pairwise distances using broadcasting
    distances = torch.norm(p1.unsqueeze(1) - p2.unsqueeze(0), dim=2)
    
    # Compute minimum distances in both directions
    min_p1_to_p2 = distances.min(dim=1)[0].mean()  # Mean of min distances from p1 to p2
    min_p2_to_p1 = distances.min(dim=0)[0].mean()  # Mean of min distances from p2 to p1
    
    # Return the symmetric Chamfer distance
    return min_p1_to_p2 + min_p2_to_p1


# =============================================================================
# LANELET2 UTILITIES
# =============================================================================
def find_lanelets_in_region(lanelet_map, sample_token, nusc, x_range=30, y_range=60):
    """Find all lanelets within a rectangular region around the ego vehicle."""
    corners_lidar = np.array([
        [-x_range/2, y_range/2, 0],
        [x_range/2, y_range/2, 0],
        [x_range/2, -y_range/2, 0],
        [-x_range/2, -y_range/2, 0],
    ])
    corners_global = lidar_to_global(corners_lidar, nusc, sample_token, ground=True)
    
    min_x, max_x = np.min(corners_global[:, 0]), np.max(corners_global[:, 0])
    min_y, max_y = np.min(corners_global[:, 1]), np.max(corners_global[:, 1])
    
    p_min = lanelet2.core.BasicPoint2d(min_x, min_y)
    p_max = lanelet2.core.BasicPoint2d(max_x, max_y)
    bounding_box_2d = lanelet2.core.BoundingBox2d(p_min, p_max)
    
    lanelets_in_region = lanelet_map.laneletLayer.search(bounding_box_2d)
    
    return lanelets_in_region

def get_adjacent_lanelets(lanes, sample_token, nusc):
    """Get lanelets that are adjacent to the ego vehicle (crossing y=0)."""
    adjacent_lanes = []
    
    for lane in lanes:
        center_points = np.array([[p.x, p.y, p.z] for p in lane.centerline])
        center_points_lidar = global_to_lidar(center_points, nusc, sample_token)
        
        for i in range(len(center_points_lidar)-1):
            y1 = center_points_lidar[i][1]
            y2 = center_points_lidar[i+1][1]
            
            if y1 * y2 <= 0:
                direction = 'next' if y1 <= 0 and y2 > 0 else 'prev'
                adjacent_lanes.append((lane, direction))
                break
        
    return adjacent_lanes

def create_lane_graph(lanes: List[lanelet2.core.Lanelet], graph: lanelet2.routing.RoutingGraph) -> nx.DiGraph:
    """Create a directed graph representation of lane connections."""
    G = nx.DiGraph()
    lane_map = {lane.id: lane for lane in lanes}
    
    for lane in lanes:
        G.add_node(lane.id, lane_obj=lane)
        
        for next_lane in graph.following(lane):
            if next_lane.id in lane_map:
                G.add_edge(lane.id, next_lane.id, direction='next')
        
        for prev_lane in graph.previous(lane):
            if prev_lane.id in lane_map:
                G.add_edge(lane.id, prev_lane.id, direction='prev')
    
    return G

def find_all_routes(
    start_lanes: List[lanelet2.core.Lanelet],
    lane_graph: nx.DiGraph,
    gt_boundaries: np.ndarray,
    sample_token: str,
    nusc,
    max_depth: int = 50
) -> List[List[lanelet2.core.Lanelet]]:
    """Find all valid routes from given start points using graph search."""
    
    def is_valid_route(route: List[lanelet2.core.Lanelet], gt_boundaries: np.ndarray,
                       x_range=(-15, 15), y_range=(-30, 30),
                       resolution=0.5) -> bool:
        """Check if route stays within bounds and is reachable from ego position."""
        points = []
        for lane in route:
            lane_points = np.array([[p.x, p.y, p.z] for p in lane.centerline])
            points_lidar = global_to_lidar(lane_points, nusc, sample_token)
            points.extend(points_lidar)
        points = np.array(points)
        
        if not is_in_range(points, x_range=x_range, y_range=(0, 30), mode='any'):
            return False
        
        x_size = int((x_range[1] - x_range[0]) / resolution)
        y_size = int((y_range[1] - y_range[0]) / resolution)
        grid = np.zeros((y_size, x_size), dtype=np.uint8)
        
        boundaries = []
        for boundary_points in gt_boundaries:
            if np.allclose(boundary_points[0], boundary_points[-1]):
                boundaries.append(Polygon(boundary_points))
            else:
                boundaries.append(LineString(boundary_points))
        boundaries.append(LineString([(-15, -5), (15, -5)]))
        
        def world_to_grid(x, y):
            grid_x = int((x - x_range[0]) / resolution)
            grid_y = int((y - y_range[0]) / resolution)
            return np.clip(grid_x, 0, x_size-1), np.clip(grid_y, 0, y_size-1)
    
        def is_in_boundary(x: float, y: float) -> bool:
            point = Point(x, y)
            for boundary in boundaries:
                if isinstance(boundary, Polygon):
                    if point.within(boundary):
                        return True
                else:
                    if point.distance(boundary) < resolution:
                        return True
            return False
        
        for i in range(y_size):
            for j in range(x_size):
                x = -15 + j * resolution
                y = -30 + i * resolution
                if is_in_boundary(x, y):
                    grid[i, j] = 1
        
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        
        start_x, start_y = world_to_grid(0, 0)
        visited = np.zeros_like(grid)
        queue = [(start_x, start_y)]
        visited[start_y, start_x] = 1
        
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
        
        while queue:
            x, y = queue.pop(0)
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < x_size and 0 <= new_y < y_size and 
                    not visited[new_y, new_x] and not grid[new_y, new_x]):
                    queue.append((new_x, new_y))
                    visited[new_y, new_x] = 1
                    
        for point in points:
            grid_x, grid_y = world_to_grid(point[0], point[1])
            if visited[grid_y, grid_x]:
                return True
                
        return False
    
    def is_subset_of_existing_routes(new_route: List[lanelet2.core.Lanelet], existing_routes: List[List[lanelet2.core.Lanelet]]) -> bool:
        """Check if the new route is a subset of any existing route."""
        new_route_ids = [lane.id for lane in new_route]
        for existing_route in existing_routes:
            existing_route_ids = [lane.id for lane in existing_route]
            
            str_new = ','.join([str(id) for id in new_route_ids])
            str_existing = ','.join([str(id) for id in existing_route_ids])
            if str_new in str_existing:
                return True
        return False
    
    valid_routes = []
    temp_routes = []
    
    for start_lane, direction in start_lanes:
        stack = [(start_lane.id, [start_lane])]
        while stack:
            current_id, current_route = stack.pop()
            
            for next_id in lane_graph.neighbors(current_id):
                if lane_graph.adj[current_id][next_id]['direction'] == direction:
                    next_lane = lane_graph.nodes[next_id]['lane_obj']
                    new_route = current_route + [next_lane]
                    
                    if len(new_route) > max_depth:
                        continue
                        
                    if is_valid_route(new_route, gt_boundaries):
                        temp_routes.append(new_route)
                        stack.append((next_id, new_route))
    
    temp_routes.sort(key=lambda x: len(x), reverse=True)
    
    for route in temp_routes:
        if not is_subset_of_existing_routes(route, valid_routes):
            valid_routes.append(route)
    
    return valid_routes

def convert_routes_to_points(
    routes: List[List[lanelet2.core.Lanelet]],
    gt_boundaries: np.ndarray,
    sample_token: str,
    nusc,
    interval=1.0,
    x_range=(-15, 15),
    y_range=(0, 30),
    dist2boundary=0.5
) -> List[np.ndarray]:
    """Convert route sequences to point sequences in lidar coordinates."""
    point_sequences = []
    
    boundary_lines = []
    for boundary in gt_boundaries:
        line = LineString(boundary)
        boundary_lines.append(line)
    
    for route in routes:
        points = []
        
        start_lane = route[0]
        start_lane_centers = np.array([[p.x, p.y, p.z] for p in start_lane.centerline])
        start_lane_centers_lidar = global_to_lidar(start_lane_centers, nusc, sample_token)
        direction = 'next' if start_lane_centers_lidar[0, 1] < 0 else 'prev'
        
        for lane in route:
            lane_centers = np.array([[p.x, p.y, p.z] for p in lane.centerline])
            lane_centers_lidar = global_to_lidar(lane_centers, nusc, sample_token)
            if direction == 'prev':
                lane_centers_lidar = lane_centers_lidar[::-1]
            points.extend(lane_centers_lidar)
            
        line = LineString([(p[0], p[1]) for p in points])
        
        line_length = line.length
        distances = np.arange(0, line_length, interval)
        points_interpolated = []
        for dist in distances:
            point = line.interpolate(dist)
            points_interpolated.append([point.x, point.y, 0])

        if line_length > distances[-1]:
            point = line.interpolate(line_length)
            points_interpolated.append([point.x, point.y, 0])
            
        points_interpolated = np.array(points_interpolated)
        
        mask = (points_interpolated[:, 0] >= x_range[0]) & (points_interpolated[:, 0] <= x_range[1]) & \
               (points_interpolated[:, 1] >= y_range[0]) & (points_interpolated[:, 1] <= y_range[1])
        points_interpolated = points_interpolated[mask]
        
        points_wo_collision = []
        for point in points_interpolated:
            x, y, z = point
            point_shapely = Point(x, y)
            collision = False
            
            for boundary in boundary_lines:
                if point_shapely.distance(boundary) < dist2boundary:
                    collision = True
                    break
            if not collision:
                points_wo_collision.append([x, y, z])
            else:
                break
        
        point_sequences.append(np.array(points_wo_collision))
    
    return point_sequences

# =============================================================================
# ROUTE PROCESSING UTILITIES
# =============================================================================

def select_goal_states(route_center_points, interval=1.0, n=3, dist2range=2.0):
    """Select n diverse goal points from route centerlines."""
    candidate_points = []
    for points in route_center_points:
        if len(points) < 2:
            continue
            
        last_point = points[-1]
        prev_point = points[-2]
        heading = np.arctan2(last_point[1] - prev_point[1],
                           last_point[0] - prev_point[0])
            
        candidate_points.append((last_point, heading, points, len(points)-1))
    
    if not candidate_points:
        return []
        
    selected = []
    remaining = candidate_points.copy()
    
    farthest = max(remaining, key=lambda x: np.linalg.norm(x[0]))
    selected.append(farthest)
    remaining = [r for r in remaining if not np.array_equal(r[0], farthest[0])]
    
    while len(selected) < n and remaining:
        max_min_dist = -1
        best_candidate = None
        
        for candidate in remaining:
            min_dist = min(np.linalg.norm(candidate[0] - s[0]) for s in selected)
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_candidate = candidate
                
        if best_candidate:
            selected.append(best_candidate)
            remaining = [r for r in remaining if not np.array_equal(r[0], best_candidate[0])]
    
    final_selected = []
    steps_back = int(dist2range / interval)
    
    for point, heading, route_points, idx in selected[:n]:
        new_idx = max(0, idx - steps_back)
        new_point = route_points[new_idx]
        
        if new_idx + 1 < len(route_points):
            next_point = route_points[new_idx + 1]
            new_heading = np.arctan2(next_point[1] - new_point[1],
                                   next_point[0] - new_point[0])
        else:
            prev_point = route_points[new_idx - 1]
            new_heading = np.arctan2(new_point[1] - prev_point[1],
                                   new_point[0] - prev_point[0])
                                   
        final_selected.append((new_point, new_heading))
    
    return final_selected


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate centerlines from map data')
    parser.add_argument('--data_root', type=str, default='data/nuscenes',
                       help='Path to nuScenes data root directory')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       help='NuScenes version')
    parser.add_argument('--dataset_dir', type=str, default='dataset/',
                       help='Path to dataset directory')
    parser.add_argument('--dataset_tag', type=str, default='asymmetric',
                       help='Dataset tag identifier')
    return parser.parse_args()


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def main():
    """Main processing function."""
    args = parse_arguments()
    
    # Initialize nuScenes
    nusc = NuScenes(
        version=args.version,
        dataroot=args.data_root,
        verbose=True
    )
    
    # Setup directories
    scene_labels_dir = os.path.join(args.dataset_dir, f'scenes_{args.dataset_tag}')
    gt_dir = os.path.join(args.dataset_dir, 'maptr-bevpool', f'train_blind_rsa_{args.dataset_tag}', 'results/map/gt')
    output_dir = os.path.join(args.dataset_dir, f'diverge_route_centerlines_{args.dataset_tag}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample tokens
    with open(os.path.join(args.dataset_dir, f'sample_tokens_{args.dataset_tag}.txt'), 'r') as f:
        sample_tokens = f.readlines()
    sample_tokens = [token.strip() for token in sample_tokens]
    
    valid_routes_num = []
    for sample_idx, sample_token in enumerate(sample_tokens):
        
        # Initialize scene and map data
        sample = nusc.get('sample', sample_token)
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        location = log['location']
        
        # Read ground truth road boundaries
        gt_path = os.path.join(gt_dir, f'{sample_token}.json')
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        gt_boundaries = [
            bbox for bbox, label in zip(gt_data['bboxes'], gt_data['labels'])
            if label == 2
        ]
        gt_vectorized_map = np.array(gt_boundaries)
        
        # Load OSM map
        map_path = f'data/lanelet2_for_nuScenes/{location}.osm'
        origin = Origin(MAP_ORIGINS[location][0], MAP_ORIGINS[location][1])
        projector = UtmProjector(origin)
        
        lanelet_map = lanelet2.io.load(map_path, projector)
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, 
                                            lanelet2.traffic_rules.Participants.Vehicle)
        lanelet_graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)
        
        # Step 1: Get lanelets
        lanelets = find_lanelets_in_region(lanelet_map, sample_token, nusc, x_range=SEARCH_X_RANGE, y_range=SEARCH_Y_RANGE)
        ego_adjacent_lanelets = get_adjacent_lanelets(lanelets, sample_token, nusc)
        
        # Step 2: Find valid routes
        graph = create_lane_graph(lanelets, lanelet_graph)
        start_lanelets = ego_adjacent_lanelets
        routes = find_all_routes(start_lanelets, graph, gt_vectorized_map, sample_token, nusc, max_depth=MAX_ROUTE_DEPTH)
        print(f'{sample_token} valid routes: {len(routes)}')
        valid_routes_num.append(len(routes))
        
        # Step 3: Convert routes to point sequences
        route_center_points = convert_routes_to_points(routes, gt_vectorized_map, sample_token, nusc,
                                                      interval=ROUTE_INTERVAL, x_range=ROUTE_X_RANGE, 
                                                      y_range=ROUTE_Y_RANGE, dist2boundary=DIST2BOUNDARY)

        
        # Step 4: Select the route closest to diverge boundary
        with open(os.path.join(scene_labels_dir, f'{sample_token}.json'), 'r') as f:
            scene_label = json.load(f)
        diverge_boundary_tag, confidence, left_total_score, right_total_score = scene_label['diverge_boundary_tag']
        for boundary in scene_label['map_elements']:
            if boundary['tag'] == diverge_boundary_tag:
                diverge_boundary_pts = np.array(boundary['coordinates'])
            else:
                reference_boundary_pts = np.array(boundary['coordinates'])
                
        # Find route closest to diverge boundary using chamfer distance
        best_route_idx = None
        min_distance = float('inf')
        
        for route_idx, center_points in enumerate(route_center_points):
            distance = chamfer_distance(center_points[:, :2], diverge_boundary_pts)
            
            if distance < min_distance:
                min_distance = distance
                best_route_idx = route_idx
                
        if best_route_idx is not None:
            selected_route = route_center_points[best_route_idx]
        else:
            print("No valid route found")
            continue
        
        # Step 5: Re-interpolate the selected route with fixed number of 20 points
        distances = np.zeros(len(selected_route))
        for i in range(1, len(selected_route)):
            distances[i] = distances[i-1] + np.linalg.norm(selected_route[i] - selected_route[i-1])
        
        x_interp = interp1d(distances, selected_route[:, 0])
        y_interp = interp1d(distances, selected_route[:, 1])
        
        new_distances = np.linspace(0, distances[-1], 20)
        
        new_x = x_interp(new_distances)
        new_y = y_interp(new_distances)
        
        if selected_route.shape[1] > 2:
            new_z = np.zeros_like(new_x)
            selected_route = np.column_stack((new_x, new_y, new_z))
        else:
            selected_route = np.column_stack((new_x, new_y))
        
        # Save selected route
        selected_route_path = os.path.join(output_dir, f'{sample_token}.json')
        with open(selected_route_path, 'w') as f:
            json.dump(selected_route[:, :2].tolist(), f)
    
    print('Average valid routes num:', np.mean(valid_routes_num))


if __name__ == '__main__':
    main()