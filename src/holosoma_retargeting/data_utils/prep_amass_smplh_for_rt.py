from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import smplx

THIS_DIR = Path(__file__).resolve()
# Add src/ to sys.path so holosoma_retargeting is importable
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))



def load_behave_sequence(seq_dir: Path):
    """
    Load a BEHAVE sequence folder that contains:
      - smpl_fit_all.npz
      - info.json (with "gender")
    """
    info_path = seq_dir / "info.json"
    info = {}
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    gender = str(info.get("gender", "neutral")).lower()

    smpl_npz = np.load(seq_dir / "smpl_fit_all.npz", allow_pickle=True)
    poses = smpl_npz["poses"]
    trans = smpl_npz["trans"]
    betas = smpl_npz["betas"]
    if "gender" in smpl_npz.files and not info.get("gender"):
        gender = str(smpl_npz["gender"]).lower()
    # Use a single beta vector; BEHAVE stores per-frame duplicates.
    if betas.ndim == 2:
        if betas.shape[0] > 1 and not np.allclose(betas, betas[0], atol=1e-6):
            print(f"Warning: betas vary across frames in {seq_dir.name}; using first frame.")
        betas = betas[0]

    return {
        "gender": gender,
        "trans": trans,
        "poses": poses,
        "betas": betas,
    }


def build_smplh_model(model_root_folder: str, gender: str, num_betas: int):
    base_dir = Path(model_root_folder) / "smplh"
    gender_dir = base_dir / gender
    model_path = gender_dir / "model.npz"
    if not model_path.exists():
        gender_upper = gender.upper()
        alt_in_gender = gender_dir / f"SMPLH_{gender_upper}.npz"
        alt_root = base_dir / f"SMPLH_{gender_upper}.npz"
        if alt_in_gender.exists():
            model_path = alt_in_gender
        elif alt_root.exists():
            model_path = alt_root
        else:
            raise FileNotFoundError(
                f"SMPL-H model not found: {model_path}, {alt_in_gender}, or {alt_root}"
            )
    model_path = _patch_smplh_model_path(model_path)
    return smplx.create(
        str(model_path),
        model_type="smplh",
        gender=gender,
        ext="npz",
        use_pca=False,
        create_transl=False,
        use_pose_mean=False,
        num_betas=num_betas,
    )


def _patch_smplh_model_path(model_path: Path) -> Path:
    """
    Some SMPL-H npz files miss hand PCA/mean fields expected by smplx.
    Create a patched copy with the missing fields if needed.
    """
    data = np.load(model_path, allow_pickle=True)
    required = [
        "hands_componentsl",
        "hands_componentsr",
        "hands_meanl",
        "hands_meanr",
        "pose_mean",
        "global_orient_mean",
        "body_pose_mean",
        "left_hand_mean",
        "right_hand_mean",
    ]
    if all(k in data.files for k in required):
        return model_path

    patched = model_path.with_name(model_path.stem + "_patched.npz")
    data_dict = {k: data[k] for k in data.files}
    zeros_comp = np.zeros((12, 45), dtype=np.float32)
    zeros_mean = np.zeros((45,), dtype=np.float32)
    data_dict["hands_componentsl"] = data_dict.get("hands_componentsl", zeros_comp)
    data_dict["hands_componentsr"] = data_dict.get("hands_componentsr", zeros_comp)
    data_dict["hands_meanl"] = np.reshape(data_dict.get("hands_meanl", zeros_mean), (45,))
    data_dict["hands_meanr"] = np.reshape(data_dict.get("hands_meanr", zeros_mean), (45,))
    data_dict["pose_mean"] = np.zeros((156,), dtype=np.float32)
    data_dict["global_orient_mean"] = np.zeros((3,), dtype=np.float32)
    data_dict["body_pose_mean"] = np.zeros((63,), dtype=np.float32)
    data_dict["left_hand_mean"] = np.zeros((45,), dtype=np.float32)
    data_dict["right_hand_mean"] = np.zeros((45,), dtype=np.float32)

    np.savez(str(patched), **data_dict)
    return patched


def compute_height(model, betas):
    num_betas = betas.shape[-1]
    betas_t = torch.tensor(betas, dtype=torch.float32).view(1, num_betas)
    zeros = torch.zeros(1, 3, dtype=torch.float32)
    with torch.no_grad():
        out = model(
            betas=betas_t,
            body_pose=torch.zeros(1, 63),
            global_orient=torch.zeros(1, 3),
            left_hand_pose=torch.zeros(1, 45),
            right_hand_pose=torch.zeros(1, 45),
            transl=zeros,
            pose2rot=True,
        )
    verts = out.vertices[0].detach().cpu().numpy()
    min_z = np.min(verts[:, 1])
    max_z = np.max(verts[:, 1])
    return max_z - min_z


def get_sequence_dirs(params_root, sequences=None):
    params_path = Path(params_root)
    if sequences:
        return [params_path / seq for seq in sequences]
    return [p for p in params_path.iterdir() if p.is_dir()]


@dataclass
class Config:
    """Configuration for processing BEHAVE SMPL-H data."""

    params_root: str = "/home/sitongchen/Documents/Master's Thesis/behaveDATA/behave-30fps-params-v1"
    """Root folder containing BEHAVE sequence folders with smpl_fit_all.npz."""

    output_folder: str = "/home/sitongchen/Documents/Master's Thesis/behaveDATA/behave-30fps-smplh-processed"
    """Output folder for processed data."""

    model_root_folder: str = "/home/sitongchen/Documents/Master's Thesis/behaveDATA/smplx"
    """Root folder containing SMPL-H model files (expects smplh/SMPLH_*.npz)."""

    sequences: list[str] | None = None
    """Optional list of sequence folder names to process."""

    plot_debug: bool = False
    """If True, plot joint positions for the first processed sequence."""

    plot_frame: int = 0
    """Frame index to plot when plot_debug is enabled."""

    plot_path: str | None = None
    """Optional path to save the debug plot as a PNG."""

    print_joint_names: bool = False
    """If True, print the joint name order used by the SMPL-H model."""


def main(cfg: Config):
    seq_dirs = get_sequence_dirs(cfg.params_root, cfg.sequences)
    os.makedirs(cfg.output_folder, exist_ok=True)

    model_cache: dict[tuple[int, str], object] = {}
    num_body_joints = 52

    for seq_idx, seq_dir in enumerate(seq_dirs):
        data = load_behave_sequence(Path(seq_dir))
        gender = data["gender"]
        betas = data["betas"]
        root_trans = data["trans"]
        aa_rot_rep = data["poses"]  # T X 156 (52*3)
        aa_rot_52 = aa_rot_rep.reshape(-1, 52, 3)

        num_betas = betas.shape[-1]
        model_key = (num_betas, gender)
        if model_key not in model_cache:
            model_cache[model_key] = build_smplh_model(cfg.model_root_folder, gender, num_betas=num_betas)
        smplh_model = model_cache[model_key]

        pose_t = torch.from_numpy(aa_rot_rep).float()
        trans_t = torch.from_numpy(root_trans).float()
        betas_t = torch.from_numpy(betas).float().view(1, -1).repeat(pose_t.shape[0], 1)
        global_orient = pose_t[:, :3]
        body_pose = pose_t[:, 3:66]
        left_hand_pose = pose_t[:, 66:111]
        right_hand_pose = pose_t[:, 111:156]
        with torch.no_grad():
            out = smplh_model(
                betas=betas_t,
                body_pose=body_pose,
                global_orient=global_orient,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                transl=trans_t,
                pose2rot=True,
            )
        if cfg.print_joint_names and seq_idx == 0:
            joint_names = getattr(smplh_model, "joint_names", None)
            if joint_names is None:
                joint_names = getattr(smplh_model, "JOINT_NAMES", None)
            if joint_names is None:
                try:
                    from smplx.joint_names import SMPLH_JOINT_NAMES as joint_names  # type: ignore[import-not-found]
                except Exception:
                    try:
                        from smplx.joint_names import JOINT_NAMES as joint_names  # type: ignore[import-not-found]
                    except Exception:
                        joint_names = None
            if joint_names is None:
                print("Joint names not available from model; update SMPLH_DEMO_JOINTS manually.")
            else:
                print("SMPL-H joint names order:")
                for idx, name in enumerate(list(joint_names)[:num_body_joints]):
                    print(f"{idx:02d}: {name}")
        global_joint_positions = out.joints.detach().cpu().numpy()[:, :num_body_joints, :]
        # Convert from (x, y, z) to (x, z, -y)
        x = global_joint_positions[..., 0]
        y = global_joint_positions[..., 1]
        z = global_joint_positions[..., 2]
        global_joint_positions = np.stack([x, z, -y], axis=-1)

        height = compute_height(smplh_model, betas)

        if cfg.plot_debug and seq_idx == 0:
            try:
                import matplotlib.pyplot as plt

                frame_idx = min(max(cfg.plot_frame, 0), global_joint_positions.shape[0] - 1)
                pts = global_joint_positions[frame_idx]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, c="orange")
                ax.set_title(f"{Path(seq_dir).name} frame {frame_idx}")
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.set_zlabel("-y")
                if cfg.plot_path:
                    out_path = Path(cfg.plot_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(out_path, dpi=200)
                    plt.close(fig)
                    print(f"Saved debug plot to {out_path}")
                else:
                    plt.show()
            except Exception as e:
                print(f"Plot skipped: {e}")

        seq_name = Path(seq_dir).name
        output_file_path = os.path.join(cfg.output_folder, f"{seq_name}.npz")
        np.savez(output_file_path, global_joint_positions=global_joint_positions, height=height)
        print(f"Saved processed data to {output_file_path}")

    print("All data processed successfully")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
