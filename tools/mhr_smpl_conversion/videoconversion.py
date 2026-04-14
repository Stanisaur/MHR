import os
import torch
import numpy as np
import smplx
import trimesh
from mhr.mhr import MHR
from conversion import Conversion
import json
from tqdm import tqdm

def bake_smplx_to_mhr(npz_path, output_dir, smplx_model_path="data/SMPLX_NEUTRAL.npz", num_viz_frames=30):
    # 1. Environment & Device Setup
    device = torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading SMPL-X model from {smplx_model_path}...")
    smplx_model = smplx.SMPLX(
        model_path=smplx_model_path, 
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,                 # Enforce 10 betas (standard shape)
        num_expression_coeffs=100     # FIX: Tell SMPL-X to expect the 100-dim FLAME expressions from BEAT2
    ).to(device)
    
    print("Loading MHR model...")
    mhr_model = MHR.from_files(lod=1, device=device)

    print("Initializing PyMomentum Converter...")
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smplx_model,
        method="pytorch"
    )

    # 2. Data Parsing
    print(f"Loading sequence data from {npz_path}...")
    beat_data = np.load(npz_path)
    
    # Safely find keys (handles both 'poses' and 'poses.npy' string formats)
    poses_key = next((k for k in beat_data.files if 'poses' in k), None)
    betas_key = next((k for k in beat_data.files if 'betas' in k), None)
    exp_key = next((k for k in beat_data.files if 'exp' in k or 'expression' in k), None)

    if poses_key:
        poses = beat_data[poses_key][:30] # Shape: [1763, 165]
        num_frames = poses.shape[0]
        
        # SMPL-X 165 index slicing:
        smplx_params = {
            "global_orient": torch.from_numpy(poses[:, 0:3]).to(torch.float32).to(device),
            "body_pose": torch.from_numpy(poses[:, 3:66]).to(torch.float32).to(device),
            "jaw_pose": torch.from_numpy(poses[:, 66:69]).to(torch.float32).to(device),
            # Indices 69:75 are eye poses, which we skip unless explicitly needed
            "left_hand_pose": torch.from_numpy(poses[:, 75:120]).to(torch.float32).to(device),
            "right_hand_pose": torch.from_numpy(poses[:, 120:165]).to(torch.float32).to(device),
        }
        
        # Parse Betas (Shape data)
        if betas_key:
            b = beat_data[betas_key]
            if len(b.shape) == 1: # Single static shape (e.g. shape: (300,))
                smplx_params["betas"] = torch.from_numpy(b[:smplx_model.num_betas]).to(torch.float32).unsqueeze(0).repeat(num_frames, 1).to(device)
            else: # Per-frame shape
                smplx_params["betas"] = torch.from_numpy(b[:, :smplx_model.num_betas]).to(torch.float32).to(device)
        else:
            smplx_params["betas"] = torch.zeros((num_frames, smplx_model.num_betas)).to(device)
            
        # Parse Expressions (100-dim FLAME)
        if exp_key:
            exp_data = beat_data[exp_key]
            smplx_params["expression"] = torch.from_numpy(exp_data[:, :smplx_model.num_expression_coeffs]).to(torch.float32).to(device)
        else:
            smplx_params["expression"] = torch.zeros((num_frames, smplx_model.num_expression_coeffs)).to(device)
            
    else:
        raise ValueError(f"Could not find poses array in NPZ. Available keys: {beat_data.files}")

    # 3. Conversion using PyMomentum
    print(f"Baking sequence ({num_frames} frames)... this will take some time on CPU.")
    results = converter.convert_smpl2mhr(
        smpl_parameters=smplx_params,
        single_identity=True,     # Solves for one consistent body shape across the video
        is_tracking=True,         # CRITICAL: Uses previous frame to initialize the next frame
        exclude_expression=False, # Map FLAME expressions to MHR blendshapes
        return_mhr_parameters=True,
        return_mhr_meshes=True,   # Required for side-by-side comparison
        return_fitting_errors=True
    )

    print(f"Conversion complete. Average vertex error: {np.mean(results.result_errors):.4f} cm")

    # 4. JSON Export (For later UE5 Live Link)
    mhr_params = results.result_parameters
    export_data = {
        "lbs_model_params": mhr_params["lbs_model_params"].detach().cpu().numpy().tolist(),
        "face_expr_coeffs": mhr_params["face_expr_coeffs"].detach().cpu().numpy().tolist(),
        "identity_coeffs": mhr_params["identity_coeffs"].detach().cpu().numpy().tolist()
    }

    json_path = os.path.join(output_dir, "unreal_ready_sequence.json")
    with open(json_path, 'w') as f:
        json.dump(export_data, f)
    print(f"Exported MHR parameters to {json_path}")

    # 5. Side-by-Side Visualization Export
    print(f"\nExporting first {min(num_viz_frames, num_frames)} frames for side-by-side visualization...")
    viz_dir = os.path.join(output_dir, "visualization")
    os.makedirs(viz_dir, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(min(num_viz_frames, num_frames)), desc="Generating .PLY sequences"):
            # A. Generate SMPL-X Mesh
            smplx_out = smplx_model(
                global_orient=smplx_params["global_orient"][i:i+1],
                body_pose=smplx_params["body_pose"][i:i+1],
                jaw_pose=smplx_params.get("jaw_pose", torch.zeros(1,3).to(device))[i:i+1],
                left_hand_pose=smplx_params["left_hand_pose"][i:i+1],
                right_hand_pose=smplx_params["right_hand_pose"][i:i+1],
                betas=smplx_params["betas"][i:i+1],
                expression=smplx_params["expression"][i:i+1]
            )
            
            # SMPL-X operates in Meters. MHR operates in Centimeters.
            # Multiply by 100 so they overlap perfectly in Blender.
            smplx_verts = smplx_out.vertices.detach().cpu().numpy()[0] * 100.0 
            smplx_mesh = trimesh.Trimesh(smplx_verts, smplx_model.faces, process=False)
            smplx_mesh.export(os.path.join(viz_dir, f"frame_{i:04d}_smplx.ply"))

            # B. Save MHR Mesh (already generated by PyMomentum)
            mhr_mesh = results.result_meshes[i]
            mhr_mesh.export(os.path.join(viz_dir, f"frame_{i:04d}_mhr.ply"))

    print(f"Done! Sequences saved to {viz_dir}.")

if __name__ == "__main__":
    # Ensure these paths correctly point to your local assets
    INPUT_NPZ = "../../data/flamesmplx30fps/testmotionvid.npz"
    OUTPUT_DIR = "../../tmp_results/conversion_out"
    SMPLX_MODEL = "../../models/smplx/SMPLX_NEUTRAL.npz" 
    
    bake_smplx_to_mhr(INPUT_NPZ, OUTPUT_DIR, SMPLX_MODEL, num_viz_frames=30)