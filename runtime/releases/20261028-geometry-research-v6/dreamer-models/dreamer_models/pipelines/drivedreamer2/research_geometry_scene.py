"""Assemble actual raw-world geometry renderers for native research runs."""
import json
from pathlib import Path

def build_research_renderer(bundle,root,release,kind='box',map_selection=None):
    if kind=='box':
        from .temporal_box_conditions import TemporalBoxConditions
        from .temporal_box_legality import TemporalBoxLegality
        renderer=TemporalBoxConditions(bundle,repository_root=release)
        renderer.physical_legality=TemporalBoxLegality(renderer)
        return renderer
    if kind=='joint':
        from .joint_world_conditions import JointWorldConditions
        return JointWorldConditions(build_research_renderer(bundle,root,release,'divider',map_selection),build_research_renderer(bundle,root,release,'box'))
    if kind=='divider':
        if map_selection is None:raise ValueError('Raw map element selection required')
        from .world_divider_conditions import WorldDividerConditions
        selection=json.loads(Path(map_selection).read_text()) if isinstance(map_selection,(str,Path)) else map_selection
        return WorldDividerConditions(bundle,raw_root=Path(root)/'assets/nuscenes_mini_temporal_camera_v1',repository_root=release,location=selection['location'],line_token=selection['line_token'],node_token=selection['node_token'],xy_budget_m=selection['budget_xy_m'])
    raise ValueError('Unsupported geometry representation')

def geometry_report(renderer):
    return renderer.last_report if hasattr(renderer,'last_report') else renderer.physical_legality.last_report
