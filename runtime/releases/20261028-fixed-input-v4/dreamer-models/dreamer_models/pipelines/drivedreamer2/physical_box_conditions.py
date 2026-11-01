"""Single-keyframe nuScenes world boxes -> native DD2 19-channel conditions.

Hard pixels execute the repository's original NumPy/OpenCV renderer. Gradients
are an explicitly different smooth raster surrogate through a straight-through
estimator (STE); they are NOT derivatives of integer OpenCV drawing. This pilot
cannot assert collision/road legality or supply missing HD maps/temporal frames.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F

from .dd2_fixed_input_adapter import RenderedDD2Conditions
from .generation_pairing import tensor_fingerprint


PANORAMA_CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                   'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT')
SOURCE_SIZE = (900, 1600)


def _compile_functions(path, names, namespace):
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError('Required native geometry functions are missing: ' + str(path))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeBoxRasterizer:
    """Execute exact local native functions; no heavyweight package imports."""
    def __init__(self, repository_root=None):
        import cv2
        root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[4]
        transforms_path = root / 'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py'
        source = transforms_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DriveDreamer2_Transform')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_canvas_box')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'init_map')
        assignments = {n.targets[0].attr: ast.literal_eval(n.value) for n in init.body
                       if isinstance(n, ast.Assign) and len(n.targets) == 1
                       and isinstance(n.targets[0], ast.Attribute)
                       and n.targets[0].attr in {'name_map', 'box_skeleton'}}
        self.name_map, self.skeleton = assignments['name_map'], assignments['box_skeleton']
        namespace = {'np': np, 'cv2': cv2, 'HEIGHT': 900, 'WIDTH': 1600}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(transforms_path), 'exec'), namespace)
        self._canvas = namespace['generate_canvas_box']
        self._receiver = SimpleNamespace(box_normal=True, box_skeleton=self.skeleton)
        self.source_hashes = {transforms_path.relative_to(root).as_posix(): hashlib.sha256(transforms_path.read_bytes()).hexdigest()}
        for relative in ('dreamer-datasets/dd_scripts/converters/nuscenes_converter.py',
                         'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_utils.py'):
            self.source_hashes[relative] = hashlib.sha256((root/relative).read_bytes()).hexdigest()
        utils = root / 'dreamer-datasets/dreamer_datasets/structures/utils'
        points = {'np': np, 'torch': torch}
        self.source_hashes['points3d_utils.py'] = _compile_functions(utils/'points3d_utils.py',
            {'_get_rot_mat_t', 'get_rot_mat_t', 'rotate_points3d'}, points)
        boxes2d = {'np': np, 'torch': torch}
        self.source_hashes['boxes_utils.py'] = _compile_functions(utils/'boxes_utils.py', {'corners_in_image'}, boxes2d)
        boxes = {'np': np, 'torch': torch, 'points3d_utils': SimpleNamespace(**points),
                 'boxes_utils': SimpleNamespace(**boxes2d)}
        self.source_hashes['boxes3d_utils.py'] = _compile_functions(utils/'boxes3d_utils.py',
            {'dims3d_to_norm_corners3d', 'boxes3d_to_corners3d', 'corners3d_to_corners2d',
             'corners3d_in_image', 'crop_corners3d'}, boxes)
        self.boxes = SimpleNamespace(**boxes)

    def canvas(self, corners, class_ids):
        value = np.asarray(corners)
        if not np.isfinite(value).all() or (value.size and np.abs(value).max() >= 2 ** 30):
            raise ValueError('Projected corners cannot safely enter the native int32 rasterizer.')
        return self._canvas(self._receiver, value, list(class_ids), thickness=10, debug=False)

    def resized(self, corners, class_ids, output_size, *, antialias):
        if not isinstance(antialias, bool):
            raise TypeError('Native bilinear antialias choice must be explicit.')
        # Native canvas is float64 CHW. Native ToTensor(canvas.transpose(1,2,0))
        # yields these same values/dtype, with no /255 conversion for floats.
        value = torch.from_numpy(self.canvas(corners, class_ids))
        return F.interpolate(value[None], size=output_size, mode='bilinear',
                             align_corners=False, antialias=antialias)[0].sub(.5).div(.5)


def quaternion_matrix(value):
    value = torch.as_tensor(value, dtype=torch.float64)
    if value.shape != (4,) or not bool(torch.isfinite(value).all()) or value.norm() < 1e-9:
        raise ValueError('Require a finite source wxyz quaternion.')
    w, x, y, z = value / value.norm()
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))).reshape(3, 3)


def pose_matrix(record):
    result = torch.eye(4, dtype=torch.float64)
    result[:3, :3] = quaternion_matrix(record['rotation'])
    result[:3, 3] = torch.tensor(record['translation'], dtype=torch.float64)
    return result


def world_to_camera_boxes(centers_world, sizes_wlh, rotations_world, camera_to_world):
    """Exact native converter convention: camera center, lhw, Ry(-quat yaw)."""
    world_to_camera = torch.linalg.inv(camera_to_world.to(centers_world))
    rotation = world_to_camera[:3, :3]
    centers = centers_world @ rotation.T + world_to_camera[:3, 3]
    orientations = rotation[None] @ rotations_world
    # PyQuaternion yaw = atan2(-R01,R00); the converter stores its negative.
    yaw = torch.atan2(orientations[:, 0, 1], orientations[:, 0, 0])
    zero = torch.zeros_like(yaw)
    boxes = torch.cat((centers, sizes_wlh[:, [1, 2, 0]], torch.stack((zero, yaw, zero), dim=-1)), dim=-1)
    return boxes, centers, orientations


def full_camera_corners(centers, sizes_wlh, orientations):
    # nuScenes Box.corners uses local (length,width,height), before the full rotation.
    signs = centers.new_tensor([[1,1,1],[1,-1,1],[1,-1,-1],[1,1,-1],
                                [-1,1,1],[-1,-1,1],[-1,-1,-1],[-1,1,-1]])
    local = signs[None] * sizes_wlh[:, [1,0,2], None].transpose(1, 2) / 2
    return local @ orientations.transpose(-1, -2) + centers[:, None]


def native_camera_corners(boxes):
    normalized = boxes.new_tensor([[0,0,0],[0,0,1],[0,1,1],[0,1,0],
                                   [1,0,0],[1,0,1],[1,1,1],[1,1,0]]) - .5
    local = normalized[None] * boxes[:, None, 3:6]
    angle = boxes[:, 7]
    c, s, z, one = angle.cos(), angle.sin(), torch.zeros_like(angle), torch.ones_like(angle)
    rotation_transpose = torch.stack((c,z,-s,z,one,z,s,z,c), dim=-1).reshape(-1,3,3)
    return local @ rotation_transpose + boxes[:, None, :3]


def crop_native_corners(corners, *, epsilon=1e-8):
    behind = corners[..., 2] < 0
    opposite = corners[:, [4,5,6,7,0,1,2,3]]
    denominator = opposite[..., 2] - corners[..., 2]
    if bool((behind & (denominator.detach().abs() < epsilon)).any()):
        raise ValueError('Native near-plane crop has an undefined opposite-edge intersection.')
    safe = torch.where(behind, denominator, torch.ones_like(denominator))
    scale = (opposite[..., 2] - .5) / safe
    xy = opposite[..., :2] - scale[..., None] * (opposite[..., :2] - corners[..., :2])
    clipped = torch.cat((xy, torch.full_like(corners[..., 2:], .5)), dim=-1)
    return torch.where(behind[..., None], clipped, corners)


def project_corners(corners, intrinsic, epsilon=1e-8):
    points = corners @ intrinsic.to(corners).T
    if bool((points[..., 2].detach().abs() < epsilon).any()):
        raise ValueError('Projection lies on the camera plane.')
    return points[..., :2] / points[..., 2:]


def _inside(points):
    return ((points[..., 0] >= 0) & (points[..., 0] < 1600)
            & (points[..., 1] >= 0) & (points[..., 1] < 900))


def soft_box_raster(corners, class_ids, *, output_size, skeleton, sigma_pixels,
                    hard_corners=None):
    """Smooth line/convex-face compositing in native source-pixel coordinates.

    Background and selected boxes share native far-to-near order. Discrete native
    skip/draw-order decisions use detached hard corners. This is a surrogate;
    output is not an approximation guarantee for OpenCV anti-aliasing/coverage.
    """
    if sigma_pixels <= 0:
        raise ValueError('sigma_pixels must be positive.')
    height, width = output_size
    yy = (torch.arange(height, dtype=corners.dtype, device=corners.device)+.5)*900/height-.5
    xx = (torch.arange(width, dtype=corners.dtype, device=corners.device)+.5)*1600/width-.5
    y, x = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack((x, y), dim=-1)
    channels = [corners.new_zeros((height, width)) for _ in range(19)]
    decisions = corners.detach() if hard_corners is None else torch.as_tensor(hard_corners, device=corners.device)
    for index, label in enumerate(class_ids):
        points, hard = corners[index], decisions[index]
        in_image = _inside(hard)
        face_first = bool(((hard[0,0]-hard[2,0]).abs()*(hard[0,1]-hard[2,1]).abs()
                           >= (hard[4,0]-hard[6,0]).abs()*(hard[4,1]-hard[6,1]).abs()).item())
        def face(value):
            if not bool(in_image[4:8].any()):
                return value
            polygon = points[4:8]
            following = polygon.roll(-1, 0)
            area = (polygon[:,0]*following[:,1]-polygon[:,1]*following[:,0]).sum()
            direction = torch.where(area.detach() >= 0, area.new_tensor(1.), area.new_tensor(-1.))
            edge = following-polygon
            offset = grid[None]-polygon[:,None,None]
            distance = direction*(edge[:,None,None,0]*offset[...,1]-edge[:,None,None,1]*offset[...,0])
            distance = distance / edge.square().sum(-1).clamp_min(1e-12).sqrt()[:,None,None]
            alpha = torch.sigmoid(distance/sigma_pixels).prod(0)
            return value*(1-alpha)+.8*alpha
        def lines(value):
            for begin, end in skeleton:
                if not bool(in_image[begin] | in_image[end]):
                    continue
                edge = points[end]-points[begin]
                offset = grid-points[begin]
                fraction = (offset*edge).sum(-1)/edge.square().sum().clamp_min(1e-12)
                closest = points[begin]+fraction.clamp(0,1)[...,None]*edge
                distance = ((grid-closest).square().sum(-1)+1e-12).sqrt()
                alpha = torch.sigmoid((5-distance)/sigma_pixels)
                value = value*(1-alpha)+alpha
            return value
        current = channels[int(label)]
        channels[int(label)] = lines(face(current)) if face_first else face(lines(current))
    return torch.stack(channels).sub(.5).div(.5)


def _depends_on(value, parameter):
    if value is parameter:
        return True
    target = parameter.grad_fn
    pending = [value.grad_fn]
    visited = set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        if (target is not None and node is target) or getattr(node, 'variable', None) is parameter:
            return True
        pending.extend(item[0] for item in node.next_functions)
    return False


class PilotPhysicalBoxConditions:
    """One actual PilotSample; no synthetic frames and no synthetic HD maps.

    build_geometry(u) uses normalized [K,3] world dx/dy/yaw. All supported raw
    annotations, including detector-filtered backgrounds, remain in generation.
    world_state selects the detector's existing token subset from those exact
    current tensors. Collision/road validity needs an explicit external callback.
    """
    def __init__(self, sample, *, selected_tokens, xy_budget_m, yaw_budget_rad,
                 output_size, antialias, fixed_hdmap=None, fixed_hdmap_provenance=None,
                 physical_legality=None, sigma_pixels=8., native=None, allow_cuda=False):
        self.sample, self.native = sample, native or NativeBoxRasterizer()
        if not isinstance(allow_cuda, bool):
            raise TypeError('allow_cuda must be explicit boolean.')
        self.allow_cuda = allow_cuda
        if sample.source_image_size != SOURCE_SIZE:
            raise ValueError('Native DD2 canvas contract requires original 900x1600 cameras.')
        self.output_size = tuple(output_size)
        if len(self.output_size) != 2 or any(not isinstance(x, int) or x <= 0 for x in self.output_size):
            raise ValueError('Explicit positive output camera size is required.')
        if not isinstance(antialias, bool):
            raise TypeError('Specify the installed native Resize antialias behavior explicitly.')
        self.antialias, self.sigma_pixels = antialias, float(sigma_pixels)
        if not np.isfinite(self.sigma_pixels) or self.sigma_pixels <= 0:
            raise ValueError('Positive finite soft sigma is required.')
        if any(not np.isfinite(x) or x <= 0 for x in (xy_budget_m, yaw_budget_rad)):
            raise ValueError('The exposed XY/yaw parameters need positive finite budgets.')
        self.budgets = torch.tensor([xy_budget_m, xy_budget_m, yaw_budget_rad], dtype=torch.float64)
        self.physical_legality = physical_legality
        if physical_legality is not None and not callable(physical_legality):
            raise TypeError('Physical legality must be an explicit callback.')
        table = sample.bundle.tables
        tokens, centers, sizes, rotations, class_ids = [], [], [], [], []
        self.unsupported_tokens = []
        for token in sample.entry['sample_annotation_tokens']:
            ann = table['sample_annotation'][token]
            name = table['category'][table['instance'][ann['instance_token']]['category_token']]['name']
            label = self.native.name_map.get(name)
            if label is None:
                self.unsupported_tokens.append(token)
                continue
            center, size = torch.tensor(ann['translation'], dtype=torch.float64), torch.tensor(ann['size'], dtype=torch.float64)
            if center.shape != (3,) or size.shape != (3,) or not bool(torch.isfinite(center).all() and torch.isfinite(size).all() and (size > 0).all()):
                raise ValueError('Invalid supported raw annotation geometry: ' + token)
            tokens.append(token); centers.append(center); sizes.append(size)
            rotations.append(quaternion_matrix(ann['rotation'])); class_ids.append(label)
        self.annotation_tokens = tuple(tokens)
        if not tokens:
            raise ValueError('No supported raw annotations are available.')
        self.base = {'centers_world': torch.stack(centers), 'sizes_wlh': torch.stack(sizes),
                     'rotations_world': torch.stack(rotations)}
        self.class_ids = tuple(class_ids)
        self.selected_tokens = tuple(selected_tokens)
        if not self.selected_tokens or len(set(self.selected_tokens)) != len(self.selected_tokens):
            raise ValueError('Declare a nonempty unique token perturbation set.')
        if not set(self.selected_tokens) <= set(tokens):
            raise ValueError('Selected token lacks a supported raw DD2 box.')
        self.selected_indices = tuple(tokens.index(token) for token in self.selected_tokens)
        if not set(sample.annotation_tokens) <= set(tokens):
            raise ValueError('A detector target cannot be mapped into the shared raw DD2 annotation set.')
        self.detector_indices = tuple(tokens.index(token) for token in sample.annotation_tokens)
        self.cameras = []
        for name in PANORAMA_CAMERAS:
            calib, pose = sample._sensor_records(sample.cameras[name], name)
            self.cameras.append((pose_matrix(pose) @ pose_matrix(calib),
                                 torch.tensor(calib['camera_intrinsic'], dtype=torch.float64)))
        self._hdmap = None
        self.hdmap_provenance = None
        if fixed_hdmap is not None:
            if not isinstance(fixed_hdmap, torch.Tensor) or fixed_hdmap.requires_grad or fixed_hdmap.device.type != 'cpu':
                raise ValueError('Fixed HDMap must be a detached CPU source tensor.')
            if tuple(fixed_hdmap.shape) != (1, 3, self.output_size[0], self.output_size[1]*6):
                raise ValueError('Fixed HDMap must already have the declared single-frame native panorama layout.')
            if not fixed_hdmap_provenance or fixed_hdmap_provenance.get('sample_token') != sample.sample_token:
                raise ValueError('A real source identity for the fixed HDMap is required.')
            required = {'source_path', 'source_sha256', 'tensor_fingerprint', 'camera_names', 'preprocessing'}
            if not required <= set(fixed_hdmap_provenance):
                raise ValueError('Fixed HDMap requires a verified source artifact and preprocessing receipt.')
            source_path = Path(fixed_hdmap_provenance['source_path'])
            if not source_path.is_file():
                raise ValueError('Fixed HDMap source artifact is missing.')
            digest = hashlib.sha256()
            with source_path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            if (digest.hexdigest() != fixed_hdmap_provenance['source_sha256']
                    or tensor_fingerprint(fixed_hdmap) != fixed_hdmap_provenance['tensor_fingerprint']
                    or tuple(fixed_hdmap_provenance['camera_names']) != PANORAMA_CAMERAS
                    or not fixed_hdmap_provenance['preprocessing']):
                raise ValueError('Fixed HDMap source/tensor hash, camera order or preprocessing identity differs.')
            if not bool(torch.isfinite(fixed_hdmap).all()) or bool(((fixed_hdmap < -1)|(fixed_hdmap > 1)).any()):
                raise ValueError('Fixed HDMap must use native normalized [-1,1] values.')
            self._hdmap = fixed_hdmap.detach().clone()
            self.hdmap_provenance = dict(fixed_hdmap_provenance)
        elif fixed_hdmap_provenance is not None:
            raise ValueError('HDMap provenance without its source tensor is invalid.')
        self._baseline = self._project(self.base)
        for selected in self.selected_indices:
            if not any(selected in view['indices'] for view in self._baseline):
                raise ValueError('Selected token is not visible in any source camera: ' + tokens[selected])
        self._check_projection_domain(self._baseline, baseline=True)

    def build_geometry(self, u):
        if not isinstance(u, torch.Tensor) or not u.is_floating_point():
            raise ValueError('World-geometry parameters must be a floating torch tensor.')
        if u.device.type == 'cuda' and not self.allow_cuda:
            raise PermissionError('CUDA world rendering requires separate explicit authorization.')
        if u.device.type not in {'cpu', 'cuda'}:
            raise ValueError('World rendering supports CPU or explicitly authorized CUDA only.')
        if tuple(u.shape) != (len(self.selected_indices), 3) or not bool(torch.isfinite(u.detach()).all()):
            raise ValueError('u must be finite [selected_objects, dx/dy/yaw] normalized parameters.')
        delta = u*self.budgets.to(u)
        indices = torch.tensor(self.selected_indices, device=u.device)
        full = u.new_zeros((len(self.annotation_tokens), 3)).index_copy(0, indices, delta)
        center_delta = torch.cat((full[:, :2], torch.zeros_like(full[:, :1])), dim=-1)
        angle = full[:, 2]
        c, s, zero, one = angle.cos(), angle.sin(), torch.zeros_like(angle), torch.ones_like(angle)
        rz = torch.stack((c,-s,zero,s,c,zero,zero,zero,one), dim=-1).reshape(-1,3,3)
        state = {'centers_world': self.base['centers_world'].to(u)+center_delta,
                 'sizes_wlh': self.base['sizes_wlh'].to(u),
                 'rotations_world': rz @ self.base['rotations_world'].to(u)}
        detector = {name: value[list(self.detector_indices)] for name, value in state.items()}
        detector['velocities_world'] = self.sample.original_world_state['velocities_world'].to(u)
        return {'normalized_u': u, 'all_world_state': state, 'world_state': detector,
                'annotation_tokens': self.annotation_tokens, 'sample_token': self.sample.sample_token,
                'scene_id': self.sample.scene_token, 'single_keyframe_only': True}

    def _project(self, state):
        views = []
        for camera_to_world, intrinsic in self.cameras:
            boxes, centers, rotations = world_to_camera_boxes(**state, camera_to_world=camera_to_world)
            full = full_camera_corners(centers, state['sizes_wlh'], rotations)
            full_array = full.detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(full_array).all() or np.any(np.abs(full_array[...,2]) < 1e-8):
                raise ValueError('Full box visibility projection is singular.')
            visible = self.native.boxes.corners3d_in_image(full_array, intrinsic.numpy(), SOURCE_SIZE)
            array = boxes.detach().cpu().numpy().astype(np.float32)
            indices = [i for i in range(len(array)) if visible[i]]
            indices.sort(key=lambda i: array[i,0]**2+array[i,1]**2+array[i,2]**2, reverse=True)
            hard3d = self.native.boxes.boxes3d_to_corners3d(array[indices], rot_axis=1)
            if len(indices) and not (hard3d[...,2].mean(-1) > 0).all():
                raise ValueError('Native approximated camera boxes lie behind the camera.')
            hard3d_cropped = self.native.boxes.crop_corners3d(hard3d.copy(), z_b=.5, threshold=0)
            hard2d = self.native.boxes.corners3d_to_corners2d(hard3d_cropped, intrinsic.numpy())
            if not np.isfinite(hard2d).all() or (hard2d.size and np.abs(hard2d).max() >= 2 ** 30):
                raise ValueError('Projected native raster coordinates are not finite safe int32 values.')
            soft3d = native_camera_corners(boxes[indices])
            soft2d = project_corners(crop_native_corners(soft3d), intrinsic)
            views.append({'indices': tuple(indices), 'visible': tuple(bool(x) for x in visible),
                          'hard_corners': hard2d, 'hard_corners3d': hard3d,
                          'soft_corners': soft2d, 'full_corners': full,
                          'class_ids': tuple(self.class_ids[i] for i in indices)})
        return views

    def _check_projection_domain(self, views, baseline=False):
        for camera_index, view in enumerate(views):
            original = view if baseline else self._baseline[camera_index]
            for token_index in self.selected_indices:
                if view['visible'][token_index] != original['visible'][token_index]:
                    raise ValueError('Selected box changed its discrete source-camera visibility set.')
                if token_index not in view['indices']:
                    continue
                index, original_index = view['indices'].index(token_index), original['indices'].index(token_index)
                if np.min(view['hard_corners3d'][index,:,2]) <= .55:
                    raise ValueError('Selected box approaches the native near-plane crop; choose a stable token/domain.')
                if bool((view['full_corners'][token_index,:,2].detach().abs() < .05).any()):
                    raise ValueError('Selected full-orientation box approaches a visibility projection singularity.')
                hard = view['hard_corners'][index].astype(np.int32)
                old = original['hard_corners'][original_index].astype(np.int32)
                face_order = lambda p: abs(p[0,0]-p[2,0])*abs(p[0,1]-p[2,1]) >= abs(p[4,0]-p[6,0])*abs(p[4,1]-p[6,1])
                if face_order(hard) != face_order(old):
                    raise ValueError('Selected box crossed a discrete native face/edge layering decision.')
                if not np.array_equal(_inside(torch.from_numpy(hard)).numpy(), _inside(torch.from_numpy(old)).numpy()):
                    raise ValueError('Selected box crossed a discrete raster image-boundary decision.')
            # Lock far-to-near ordering: a changed order can alter occlusion discontinuously.
            if view['indices'] != original['indices']:
                raise ValueError('Native visible-box ordering changed within the bounded pilot domain.')

    def _validated_views(self, geometry):
        u = geometry.get('normalized_u')
        if isinstance(u, torch.Tensor) and u.device.type == 'cuda' and not self.allow_cuda:
            raise PermissionError('CUDA world rendering requires separate explicit authorization.')
        if (geometry.get('sample_token') != self.sample.sample_token
                or geometry.get('scene_id') != self.sample.scene_token
                or geometry.get('annotation_tokens') != self.annotation_tokens):
            raise ValueError('Geometry does not identify this exact raw sample/token set.')
        if bool((geometry['normalized_u'].detach().abs() > 1).any()):
            raise ValueError('Normalized world perturbation exceeds its declared budget.')
        # Rebuild from u so separately overwritten all_world_state cannot silently desynchronize labels.
        rebuilt = self.build_geometry(geometry['normalized_u'])
        for section in ('all_world_state', 'world_state'):
            if set(geometry[section]) != set(rebuilt[section]):
                raise ValueError('Current world-state fields are incomplete.')
            for name, expected in rebuilt[section].items():
                actual = geometry[section][name]
                if actual.shape != expected.shape or not torch.equal(actual.detach(), expected.detach()):
                    raise ValueError('Geometry/label state diverged from the shared physical perturbation.')
                if expected.requires_grad:
                    if not actual.requires_grad:
                        raise ValueError('Current world state detached the physical perturbation graph.')
                    if not _depends_on(actual, geometry['normalized_u']):
                        raise ValueError('Current world state is disconnected from physical perturbations.')
        views = self._project(geometry['all_world_state'])
        self._check_projection_domain(views)
        return views

    def geometry_is_valid(self, geometry):
        try:
            self._validated_views(geometry)
        except (ValueError, TypeError, KeyError):
            return False
        if self.physical_legality is None:
            return False
        result = self.physical_legality(geometry)
        if not isinstance(result, bool):
            raise TypeError('External collision/road legality must return an explicit bool.')
        return result

    def __call__(self, geometry):
        views = self._validated_views(geometry)
        images = []
        for view in views:
            hard = self.native.resized(view['hard_corners'], view['class_ids'], self.output_size,
                                       antialias=self.antialias).to(geometry['normalized_u'])
            if torch.is_grad_enabled() and geometry['normalized_u'].requires_grad:
                soft = soft_box_raster(view['soft_corners'], view['class_ids'], output_size=self.output_size,
                    skeleton=self.native.skeleton, sigma_pixels=self.sigma_pixels,
                    hard_corners=view['hard_corners'].astype(np.int32))
                # Parentheses are intentional: zero forward value, exact hard pixels.
                hard = hard + (soft-soft.detach())
            images.append(hard)
        raw = {'box_downsampler_input': torch.cat(images, dim=-1)[None]}
        if self._hdmap is not None:
            raw['grounding_downsampler_input'] = self._hdmap.to(geometry['normalized_u']).clone()
        return RenderedDD2Conditions(self.sample.scene_token, PANORAMA_CAMERAS, raw)

    def prepare_detector_inputs(self, video, geometry, *, pixel_transforms):
        self._validated_views(geometry)
        return self.sample.prepare_inputs(video=video, world_state=geometry['world_state'],
            input_camera_order=PANORAMA_CAMERAS, pixel_transforms=pixel_transforms)

    def audit(self):
        return {'single_keyframe_only': True, 'temporal_generation_validated': False,
                'raw_supported_annotations': len(self.annotation_tokens),
                'detector_annotation_subset': len(self.detector_indices),
                'detector_filtered_background_boxes_retained': len(self.annotation_tokens)-len(self.detector_indices),
                'unsupported_raw_annotation_tokens': list(self.unsupported_tokens),
                'selected_tokens': list(self.selected_tokens), 'camera_order': list(PANORAMA_CAMERAS),
                'hard_renderer_source_sha256': dict(self.native.source_hashes),
                'gradient': 'soft_line_convex_face_STE_not_hard_raster_derivative',
                'antialias': self.antialias, 'soft_sigma_source_pixels': self.sigma_pixels,
                'fixed_hdmap_present': self._hdmap is not None,
                'fixed_hdmap_provenance': self.hdmap_provenance,
                'collision_road_legality_callback_present': self.physical_legality is not None,
                'physical_legality_implemented_here': False, 'research_result': False}
