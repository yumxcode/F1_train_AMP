# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This module provides functionality to convert motion data from GMR format to Isaac Lab format.

Ref: 
    - https://github.com/xbpeng/MimicKit/blob/main/tools/gmr_to_mimickit/gmr_to_mimickit.py
    - https://github.com/HybridRobotics/whole_body_tracking/blob/main/scripts/csv_to_npz.py

GMR Format:
    The input GMR format should be a pickle file containing a dictionary with keys:
    - 'fps': Frame rate (int)
    - 'root_pos': Root position array, shape (num_frames, 3)
    - 'root_rot': Root rotation quaternions, shape (num_frames, 4), format (x, y, z, w)
    - 'dof_pos': Degrees of freedom positions, shape (num_frames, num_dofs)
    - 'local_body_pos': Currently unused (can be None)
    - 'link_body_list': Currently unused (can be None)

Output Isaac Lab Format:
    The output Isaac Lab format is a dictionary with keys:
    - 'fps': Frame rate (int)
    - 'root_pos': Root position array, shape (num_frames, 3)
    - 'root_rot': Root rotation quaternions, shape (num_frames, 4), format (w, x, y, z)
    - 'dof_pos': Degrees of freedom positions, shape (num_frames, num_dofs)
    - 'loop_mode': Loop mode (int, 0 for clamp, 1 for wrap)
    - 'key_body_pos': Key body positions in world frame, shape (num_frames, num_key_bodies, 3)


"""


import pickle
import numpy as np
import enum
import torch


import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1


def extract_gmr_data(
    gmr_file_path: str, 
    gmr_dof_names: list[str],
    lab_dof_names: list[str],
    loop_mode: LoopMode,
    start_frame: int = 0,
    end_frame: int = -1,
    gmr_to_lab_name_map: dict[str, str] | None = None,
    default_dof_value: float | None = None,
):
    """Convert GMR motion data to Isaac Lab format.

    When ``gmr_to_lab_name_map`` is provided, each GMR DOF name is translated to
    its Lab equivalent before lookup.  Lab DOFs that have no corresponding GMR
    entry are filled with ``default_dof_value`` (0.0 if ``None`` but map is set).

    When ``gmr_to_lab_name_map`` is *not* provided (legacy mode), Lab DOF names
    must match GMR names exactly.  If ``default_dof_value`` is ``None`` (default),
    unmapped DOFs raise ``ValueError`` (backward compatible with rpo).
    """
    with open(gmr_file_path, 'rb') as f:
        gmr_data = pickle.load(f)
        
    # Extract data from GMR format
    fps = gmr_data['fps']
    root_pos = gmr_data['root_pos']  # Shape: (num_frames, 3)
    root_rot_quat = gmr_data['root_rot']  # Shape: (num_frames, 4), quaternion format
    dof_pos = gmr_data['dof_pos']    # Shape: (num_frames, num_dofs)

    # Log the type and shape of each extracted term
    print("\n" + "="*60)
    print("LOADED GMR DATA")
    print("="*60)
    print(f"  FPS:           type={type(fps).__name__}, value={fps}")
    print(f"  Root Position: type={type(root_pos).__name__}, shape={root_pos.shape}")
    print(f"  Root Rotation: type={type(root_rot_quat).__name__}, shape={root_rot_quat.shape}")
    print(f"  DOF Position:  type={type(dof_pos).__name__}, shape={dof_pos.shape}")
    print("="*60 + "\n")

    # Verify shapes
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (num_frames, 3), got {root_pos.shape}")
        
    if root_rot_quat.ndim != 2 or root_rot_quat.shape[1] != 4:
        raise ValueError(f"Expected root_rot_quat shape (num_frames, 4), got {root_rot_quat.shape}")
        
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos to be 2D array, got {dof_pos.ndim}D")
    
    num_frames = dof_pos.shape[0]
    if end_frame == -1 or end_frame > num_frames:
        end_frame = num_frames
    assert 0 <= start_frame < end_frame <= num_frames, "Invalid start_frame or end_frame."

    # ------------------------------------------------------------------
    # Build a lookup: lab_dof_name -> gmr_dof_index (or None if unmapped)
    # ------------------------------------------------------------------
    lab_to_gmr_idx: dict[str, int | None] = {}
    if gmr_to_lab_name_map is not None:
        # Resolve effective default for unmapped DOFs when map is provided
        effective_default = 0.0 if default_dof_value is None else default_dof_value
        for gmr_name, lab_name in gmr_to_lab_name_map.items():
            if gmr_name not in gmr_dof_names:
                raise ValueError(
                    f"GMR DOF '{gmr_name}' (from name map) not found in gmr_dof_names."
                )
            lab_to_gmr_idx[lab_name] = gmr_dof_names.index(gmr_name)
    else:
        effective_default = default_dof_value  # may be None → raise on missing

    # Fill dof_pos_lab column by column
    dof_pos_lab = np.zeros((num_frames, len(lab_dof_names)), dtype=np.float64)
    for i, lab_dof in enumerate(lab_dof_names):
        if lab_dof in lab_to_gmr_idx and lab_to_gmr_idx[lab_dof] is not None:
            dof_pos_lab[:, i] = dof_pos[:, lab_to_gmr_idx[lab_dof]]
        elif gmr_to_lab_name_map is None and lab_dof in gmr_dof_names:
            dof_pos_lab[:, i] = dof_pos[:, gmr_dof_names.index(lab_dof)]
        else:
            if effective_default is None:
                raise ValueError(f"DOF name '{lab_dof}' not found in GMR DOF names.")
            dof_pos_lab[:, i] = effective_default

    # set the elbow yaw joint to 0.0 (these joints do not need action)
    for i, lab_dof in enumerate(lab_dof_names):
        if lab_dof.endswith("_elbow_yaw_joint"):
            dof_pos_lab[:, i] = 0.0

    output_data = {
        'fps': fps,
        'root_pos': root_pos[start_frame:end_frame],
        'root_rot': root_rot_quat[start_frame:end_frame],
        'dof_pos': dof_pos_lab[start_frame:end_frame],
        'loop_mode': loop_mode.value,
    }
    
    return output_data

def run_simulator(
        simulation_app, 
        sim: sim_utils.SimulationContext, 
        scene: InteractiveScene, 
        motion_data_dicts: list[dict[str, np.ndarray]], 
        key_body_names: list[str],
        lab_dof_names: list[str] | None = None):
    
    robot: Articulation = scene["robot"]
    # marker
    marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/FrameVisualizerFromScript",
        markers={
            "red_sphere": sim_utils.SphereCfg(
                radius=0.03, 
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            ),
        }
    )
    marker: VisualizationMarkers = VisualizationMarkers(marker_cfg)
    
    # get the motion data
    num_motions = len(motion_data_dicts)
    assert num_motions == scene.num_envs, "Number of motions must match number of environments."
    fps = motion_data_dicts[0]['fps']
    root_pos_w_list = []
    root_quat_list = []
    dof_pos_list = []
    num_frames_list = []
    
    for motion_data in motion_data_dicts:
        # assert motion_data['fps'] == fps, "All motions must have the same fps."
        root_pos_w_list.append(torch.from_numpy(motion_data['root_pos']).to(scene.device).float())
        
        root_quat_tensor = torch.from_numpy(motion_data['root_rot']).to(scene.device).float()
        root_quat_tensor = math_utils.convert_quat(root_quat_tensor, "wxyz") # convert to w, x, y, z format
        root_quat_tensor = math_utils.quat_unique(root_quat_tensor)
        root_quat_tensor = math_utils.normalize(root_quat_tensor)
        root_quat_list.append(root_quat_tensor)
        
        dof_pos_list.append(torch.from_numpy(motion_data['dof_pos']).to(scene.device).float())
        num_frames_list.append(motion_data['dof_pos'].shape[0])

    max_num_frames = max(num_frames_list)
    
    lab_body_names = robot.data.body_names
    lab_joint_names = robot.data.joint_names

    # ------------------------------------------------------------------
    # DOF remapping: motion data may be in a different order than the
    # robot's internal joint order.  Build a remap tensor so we can index
    # motion dof_pos by robot order.
    # ------------------------------------------------------------------
    if lab_dof_names is not None:
        dof_remap = []
        print("\n" + "=" * 70)
        print("DOF ORDER VERIFICATION")
        print("=" * 70)
        print(f"{'Idx':>3}  {'Robot Joint Name (runtime)':<40} {'YAML lab_dof_names Idx':>6}")
        print("-" * 70)
        for ri, rj_name in enumerate(lab_joint_names):
            if rj_name not in lab_dof_names:
                print(f"{'':>3}  {rj_name:<40} *** NOT FOUND IN YAML ***")
                raise ValueError(
                    f"Robot joint '{rj_name}' (index {ri}) not found in lab_dof_names. "
                    f"Update the YAML config to include this joint."
                )
            yi = lab_dof_names.index(rj_name)
            flag = "" if yi == ri else f"  <- remapped from yaml[{yi}]"
            print(f"{ri:>3}  {rj_name:<40} {yi:>6}{flag}")
            dof_remap.append(yi)
        print("-" * 70)
        if dof_remap == list(range(len(lab_joint_names))):
            print("YAML order matches robot order exactly (identity remap).")
        else:
            print(f"WARNING: YAML order differs from robot order. Remap applied: {dof_remap}")
        print("=" * 70 + "\n")
        dof_remap_tensor = torch.tensor(dof_remap, device=scene.device, dtype=torch.long)
    else:
        dof_remap_tensor = None

    # print(f"[INFO]: Isaac Lab body names: {lab_body_names}")
    key_body_indices = []
    for name in key_body_names:
        if name in lab_body_names:
            key_body_indices.append(lab_body_names.index(name))
        else:
            raise ValueError(f"Key body name '{name}' not found in Isaac Lab body names.")
    key_body_pos_w_list = [
        torch.zeros((num_frames, len(key_body_indices), 3), device=scene.device) 
        for num_frames in num_frames_list
    ]
    
    count = 0
    sim_time = 0.0
    dt = sim.cfg.dt
    
    while simulation_app.is_running():
        
        root_states = robot.data.default_root_state.clone()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel)
        
        for motion_idx in range(num_motions):
            num_frames = num_frames_list[motion_idx]            
            frame_idx = count if count < num_frames else num_frames - 1
            
            # set root state
            root_states[motion_idx, :3] = root_pos_w_list[motion_idx][frame_idx, :]
            root_states[motion_idx, :3] += scene.env_origins[motion_idx, :3]
            root_states[motion_idx, 3:7] = root_quat_list[motion_idx][frame_idx, :]
            root_states[motion_idx, 7:10] = 0.0  # zero linear velocity
            root_states[motion_idx, 10:13] = 0.0  # zero angular velocity
            
            # set joint state (remap if needed)
            raw_dof = dof_pos_list[motion_idx][frame_idx, :]
            if dof_remap_tensor is not None:
                joint_pos[motion_idx, :] = raw_dof[dof_remap_tensor]
            else:
                joint_pos[motion_idx, :] = raw_dof
            
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        
        # step without physics
        sim.render()
        scene.update(dt)
        
        for motion_idx in range(num_motions):
            num_frames = num_frames_list[motion_idx]
            if count < num_frames:
                key_body_pos_w_tensor = robot.data.body_pos_w[motion_idx, key_body_indices, :] - scene.env_origins[motion_idx, :3]
                key_body_pos_w_list[motion_idx][count, :, :] = key_body_pos_w_tensor
        
        vis_key_body_pos_w = robot.data.body_pos_w[:, key_body_indices, :]
        marker.visualize(
            translations=vis_key_body_pos_w.reshape(-1, 3)
        )
        
        count += 1
        sim_time += dt
        if count >= max_num_frames:
            break
        
    print(f"[INFO]: Simulation completed in {count} steps, total time: {sim_time:.2f} seconds.")
    
    for motion_data_dict, root_quat in zip(motion_data_dicts, root_quat_list):
        motion_data_dict['root_rot'] = root_quat.cpu().numpy()
        
    for motion_data_dict, key_body_pos_w in zip(motion_data_dicts, key_body_pos_w_list):
        motion_data_dict['key_body_pos'] = key_body_pos_w.cpu().numpy()

    # Remap dof_pos to robot order so the saved pkl matches training robot DOFs
    if dof_remap_tensor is not None:
        for motion_data_dict, dof_tensor in zip(motion_data_dicts, dof_pos_list):
            remapped = dof_tensor[:, dof_remap_tensor].cpu().numpy()
            motion_data_dict['dof_pos'] = remapped
        print("[run_simulator] dof_pos remapped to robot joint order in output.")
        
    return motion_data_dicts
    
    
@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = None

    
