import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# viz
from pytorch_msssim import ssim
import lpips
from dreamsim import dreamsim

# pos
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
from evo.core import metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

def coords_to_evo_traj(coords_xy_yaw: np.ndarray) -> PoseTrajectory3D:
    positions_xyz = np.zeros((len(coords_xy_yaw), 3))
    positions_xyz[:, :2] = coords_xy_yaw[:, :2]

    orientations_quat_wxyz = np.zeros((len(coords_xy_yaw), 4))
    yaws = coords_xy_yaw[:, 2]
    orientations_quat_wxyz[:, 0] = np.cos(yaws / 2)
    orientations_quat_wxyz[:, 3] = np.sin(yaws / 2)

    timestamps = np.arange(len(coords_xy_yaw), dtype=np.float64)

    return PoseTrajectory3D(
        positions_xyz=positions_xyz,
        orientations_quat_wxyz=orientations_quat_wxyz,
        timestamps=timestamps
    )

def eval_ate_rpe(traj_ref: PoseTrajectory3D, traj_pred: PoseTrajectory3D) -> tuple[float, float, float]:
    traj_ref, traj_pred = sync.associate_trajectories(traj_ref, traj_pred)

    ate_result = main_ape.ape(
        traj_ref, traj_pred, est_name='traj',
        pose_relation=PoseRelation.translation_part,
        align=False, correct_scale=False
    )
    ate = ate_result.stats['rmse']

    rpe_trans_result = main_rpe.rpe(
        traj_ref, traj_pred, est_name='traj',
        pose_relation=PoseRelation.translation_part,
        align=False, correct_scale=False,
        delta=1, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1
    )
    rpe_trans = rpe_trans_result.stats['rmse']

    rpe_rot_result = main_rpe.rpe(
        traj_ref, traj_pred, est_name='traj',
        pose_relation=PoseRelation.rotation_part,
        align=False, correct_scale=False,
        delta=1, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1
    )
    rpe_rot = rpe_rot_result.stats['rmse']

    return ate, rpe_trans, rpe_rot



class ImageMetricsCalculator:
    def __init__(self, device='cuda'):
        print(f"Initializing ImageMetricsCalculator on device: {device}")
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.lpips_metric = lpips.LPIPS(net='alex').to(self.device).eval()
        self.dreamsim_metric, _ = dreamsim(pretrained=True, device=self.device)
        self.dreamsim_metric.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True)
        ])

    def _preprocess(self, image):
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        
        while image.dim() > 3:
            image = image.squeeze(0)
            
        image = image.to(self.device).float()
        
        if image.max() > 1.0:
            image = image / 255.0
            
        return self.transform(image).unsqueeze(0)

    @torch.no_grad()
    def calculate(self, pred_image, gt_image):
        pred_tensor_0_1 = self._preprocess(pred_image)
        goal_tensor_0_1 = self._preprocess(gt_image)

        ssim_score = ssim(pred_tensor_0_1, goal_tensor_0_1, data_range=1.0, size_average=True).item()
        mse = F.mse_loss(pred_tensor_0_1, goal_tensor_0_1)
        psnr_score = 10 * torch.log10(1.0 / mse).item() if mse > 0 else float('inf')

        pred_tensor_neg1_1 = (pred_tensor_0_1 * 2) - 1
        goal_tensor_neg1_1 = (goal_tensor_0_1 * 2) - 1

        lpips_score = self.lpips_metric(pred_tensor_neg1_1, goal_tensor_neg1_1).item()
        dreamsim_score = self.dreamsim_metric(pred_tensor_neg1_1, goal_tensor_neg1_1).item()

        return {
            'psnr': psnr_score,
            'ssim': ssim_score,
            'lpips': lpips_score,
            'dreamsim': dreamsim_score
        }