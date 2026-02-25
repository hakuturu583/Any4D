#!/usr/bin/env python3
# --------------------------------------------------------
# PhysicalAI-AV → Any4D Integration Script
#
# NVIDIA PhysicalAI-Autonomous-Vehicles データセット（HuggingFace）から
# 単一シーケンスをダウンロードし、Any4D で4D再構成を行う。
#
# 依存ライブラリ:
#   pip install physical-ai-av DracoPy opencv-python
# --------------------------------------------------------

import argparse
import math
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as tvf

# Any4D パス設定
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from any4d.utils.inference import (
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
from any4d.utils.misc import seed_everything
from any4d.utils.viz import script_add_rerun_args
from demo_inference import (
    init_inference_model,
    sample_inference,
    visualize_raw_custom_data_inference_output,
)

# --------------------------------------------------------
# 定数
# --------------------------------------------------------
DEFAULT_TARGET_FOV_DEG = 70.0  # アンディストーション後の FOV（度）
DEFAULT_TARGET_H = 336
DEFAULT_TARGET_W = 518  # 解像度マッピング: (518, 336) = 3:2


# --------------------------------------------------------
# 1. データダウンロード
# --------------------------------------------------------
def download_single_clip(clip_id: str, cameras: list[str], cache_dir: str):
    """
    physical_ai_av で1クリップだけ必要なフィーチャーをダウンロードする。

    Args:
        clip_id: クリップ UUID
        cameras: 使用するカメラ名リスト
        cache_dir: ローカルキャッシュディレクトリ

    Returns:
        PhysicalAIAVDatasetInterface インスタンス
    """
    try:
        from physical_ai_av import PhysicalAIAVDatasetInterface
    except ImportError:
        raise ImportError(
            "physical-ai-av パッケージが見つかりません。"
            " `pip install physical-ai-av` を実行してください。"
        )

    os.makedirs(cache_dir, exist_ok=True)
    # confirm_download_threshold_gb=inf でダウンロード確認プロンプトをスキップ
    # (データは chunk 単位で保存されているため複数クリップ分のサイズになる場合がある)
    dataset = PhysicalAIAVDatasetInterface(
        local_dir=cache_dir,
        confirm_download_threshold_gb=math.inf,
    )

    features = list(cameras) + [
        "lidar_top_360fov",
        "egomotion",
        "camera_intrinsics",
        "sensor_extrinsics",
    ]
    print(f"[download] clip_id={clip_id}, features={features}")
    dataset.download_clip_features(clip_id=clip_id, features=features)
    return dataset


# --------------------------------------------------------
# 2. クリップ自動選択
# --------------------------------------------------------
def auto_select_clip(
    dataset, filter_country: str | None = None, filter_hour: int | None = None
) -> str:
    """
    条件フィルタを適用して最初のクリップ ID を返す。

    clip_index: clip_is_valid / chunk / split
    sensor_presence: カメラ・LiDAR 存在フラグ
    追加メタデータ (country, hour_of_day 等) は download_metadata() で取得。

    Args:
        dataset: PhysicalAIAVDatasetInterface インスタンス
        filter_country: 国コードフィルタ (例: "US", "DE")
        filter_hour: 時間帯フィルタ (0–23)

    Returns:
        clip_id 文字列
    """
    sp = dataset.sensor_presence.copy()
    ci = dataset.clip_index

    # 有効なクリップのみ
    valid_ids = ci[ci["clip_is_valid"]].index
    sp = sp[sp.index.isin(valid_ids)]

    # LiDAR が存在するクリップのみ
    if "lidar_top_360fov" in sp.columns:
        sp = sp[sp["lidar_top_360fov"] == True]

    # country / hour フィルタ — 追加メタデータが必要
    if filter_country is not None or filter_hour is not None:
        try:
            dataset.download_metadata()
        except Exception as e:
            warnings.warn(f"追加メタデータのダウンロードに失敗しました: {e}")

        extra_meta = getattr(dataset, "metadata", {})
        for _key, meta_df in extra_meta.items():
            if filter_country is not None and "country" in meta_df.columns:
                valid = meta_df.loc[meta_df["country"] == filter_country].index
                sp = sp[sp.index.isin(valid)]
            if filter_hour is not None and "hour_of_day" in meta_df.columns:
                valid = meta_df.loc[meta_df["hour_of_day"] == filter_hour].index
                sp = sp[sp.index.isin(valid)]

        if len(sp) == 0:
            warnings.warn(
                "country/hour フィルタ後にクリップが見つかりません。フィルタをスキップします。"
            )
            # フィルタなしで再取得
            sp = dataset.sensor_presence.copy()
            sp = sp[sp.index.isin(valid_ids)]
            if "lidar_top_360fov" in sp.columns:
                sp = sp[sp["lidar_top_360fov"] == True]

    if len(sp) == 0:
        raise ValueError("条件に合うクリップが見つかりませんでした。フィルタ条件を緩めてください。")

    clip_id = sp.index[0]
    print(f"[auto_select] clip_id={clip_id}")
    return clip_id


# --------------------------------------------------------
# 3. 利用可能カメラ一覧取得
# --------------------------------------------------------
def get_available_cameras(dataset, clip_id: str) -> list[str]:
    """
    sensor_presence テーブルから当該クリップに存在するカメラ名を返す。

    Args:
        dataset: PhysicalAIAVDatasetInterface インスタンス
        clip_id: クリップ UUID

    Returns:
        カメラ名のリスト
    """
    df = dataset.sensor_presence
    if clip_id not in df.index:
        # インデックスがない場合はカラム名からカメラを推定
        camera_cols = [c for c in df.columns if c.startswith("camera_")]
        return camera_cols

    row = df.loc[clip_id]
    camera_cols = [c for c in df.columns if c.startswith("camera_") and row.get(c, False)]
    return camera_cols


# --------------------------------------------------------
# 4. F-theta アンディストーション写像計算
# --------------------------------------------------------
def compute_ftheta_undistort_map(
    intr_row,
    orig_h: int,
    orig_w: int,
    out_h: int,
    out_w: int,
    fov_deg: float = DEFAULT_TARGET_FOV_DEG,
):
    """
    F-theta 多項式モデル（fw_poly: θ → r）を使って
    cv2.remap 用の写像 (map_x, map_y) とピンホール内部行列 K_new を計算する。

    F-theta: r = f(θ) = a0*θ + a1*θ^3 + a2*θ^5 + ...
    ここで fw_poly_0~4 は各次数の係数。

    Args:
        intr_row: camera_intrinsics の1行 (cx, cy, fw_poly_0~4 等を含む)
        orig_h, orig_w: 元画像サイズ
        out_h, out_w: 出力（アンディストーション後）画像サイズ
        fov_deg: 出力ピンホール画像の対角 FOV（度）

    Returns:
        map_x, map_y: cv2.remap 用 float32 写像
        K_new: (3, 3) ピンホール内部行列 numpy array
    """
    # fw_poly 係数を取得
    poly = []
    for i in range(5):
        key = f"fw_poly_{i}"
        val = float(intr_row[key]) if key in intr_row.index else 0.0
        poly.append(val)

    # 光学中心
    cx_orig = float(intr_row.get("cx", orig_w / 2))
    cy_orig = float(intr_row.get("cy", orig_h / 2))

    # 新しいピンホール内部行列
    # fov_deg = 対角 FOV
    # new_focal = (out_h/2) / tan(fov_y/2)  where fov_y ≈ fov_deg * out_h / diag
    diag = math.sqrt(out_w**2 + out_h**2)
    fov_rad = math.radians(fov_deg)
    # 対角 focal を計算し、縦横に分解
    f_diag = (diag / 2.0) / math.tan(fov_rad / 2.0)
    new_fx = f_diag * out_w / diag
    new_fy = f_diag * out_h / diag
    new_cx = out_w / 2.0
    new_cy = out_h / 2.0

    K_new = np.array(
        [[new_fx, 0.0, new_cx], [0.0, new_fy, new_cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # 出力ピクセル (u, v) → カメラ光線方向 → F-theta 歪み座標
    u_coords = np.arange(out_w, dtype=np.float32)
    v_coords = np.arange(out_h, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)  # (out_h, out_w)

    # ピンホール座標系での方向ベクトル
    x_cam = (uu - new_cx) / new_fx
    y_cam = (vv - new_cy) / new_fy
    z_cam = np.ones_like(x_cam)
    norm_xy = np.sqrt(x_cam**2 + y_cam**2)

    # θ = arctan(r_cam / z_cam) = arctan(norm_xy)
    theta = np.arctan(norm_xy)  # (out_h, out_w)

    # F-theta 多項式: r_img = poly[0] + poly[1]*θ + poly[2]*θ^2 + poly[3]*θ^3 + poly[4]*θ^4
    # PhysAI-AV の fw_poly_i は連続次数の係数（bw_poly_i も同様に r^i の係数）
    r_img = (
        poly[0]
        + poly[1] * theta
        + poly[2] * theta**2
        + poly[3] * theta**3
        + poly[4] * theta**4
    )

    # 正規化して元画像のピクセル座標に変換
    # 光線の XY 方向が 0 のとき（光軸上）は特別扱い
    norm_xy_safe = np.where(norm_xy < 1e-8, 1.0, norm_xy)
    cos_phi = x_cam / norm_xy_safe
    sin_phi = y_cam / norm_xy_safe

    map_x = (cx_orig + r_img * cos_phi).astype(np.float32)
    map_y = (cy_orig + r_img * sin_phi).astype(np.float32)

    # 光軸上のピクセルは光学中心にマップ
    on_axis = norm_xy < 1e-8
    map_x[on_axis] = cx_orig
    map_y[on_axis] = cy_orig

    return map_x, map_y, K_new


# --------------------------------------------------------
# 5. 画像アンディストーション
# --------------------------------------------------------
def undistort_image(image_bgr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """
    cv2.remap でアンディストーションを適用する。

    Args:
        image_bgr: BGR 画像 (H, W, 3) uint8
        map_x, map_y: compute_ftheta_undistort_map の出力

    Returns:
        アンディストーション後の BGR 画像 (out_h, out_w, 3) uint8
    """
    return cv2.remap(
        image_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# --------------------------------------------------------
# 6. LiDAR デコード (DracoPy)
# --------------------------------------------------------
def decode_lidar_spin(draco_encoded_bytes: bytes) -> np.ndarray:
    """
    DracoPy で Draco 圧縮された点群をデコードして (N, 3) numpy array を返す。

    Args:
        draco_encoded_bytes: Draco 圧縮バイト列

    Returns:
        (N, 3) float32 numpy array (X, Y, Z in LiDAR frame)
    """
    try:
        import DracoPy
    except ImportError:
        raise ImportError(
            "DracoPy が見つかりません。`pip install DracoPy` を実行してください。"
        )

    mesh_or_pc = DracoPy.decode(draco_encoded_bytes)
    points = np.array(mesh_or_pc.points, dtype=np.float32)  # (N, 3)
    return points


# --------------------------------------------------------
# 7. カメラフレームに最近傍 LiDAR スピンを同期
# --------------------------------------------------------
# --------------------------------------------------------
# 8. LiDAR 点群をカメラ座標系に変換
# --------------------------------------------------------
def _row_to_transform(extr_row) -> np.ndarray:
    """
    sensor_extrinsics 行から 4×4 変換行列を生成する。
    変換: T_rig_sensor（センサー座標 → rig 座標）

    quaternion 順序: (qw, qx, qy, qz) または (qx, qy, qz, qw) に対応。
    カラム名 qx, qy, qz, qw が存在すると仮定する。
    """
    qx = float(extr_row["qx"])
    qy = float(extr_row["qy"])
    qz = float(extr_row["qz"])
    qw = float(extr_row["qw"])

    tx = float(extr_row["x"])
    ty = float(extr_row["y"])
    tz = float(extr_row["z"])

    # クォータニオン → 回転行列
    R = _quat_to_rot(qw, qx, qy, qz)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """クォータニオン (w, x, y, z) → 3×3 回転行列"""
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    wx = s * qw * qx
    wy = s * qw * qy
    wz = s * qw * qz
    xx = s * qx * qx
    xy = s * qx * qy
    xz = s * qx * qz
    yy = s * qy * qy
    yz = s * qy * qz
    zz = s * qz * qz
    R = np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )
    return R


def transform_lidar_to_camera(
    lidar_pts_rig: np.ndarray,
    lidar_extr_row,
    cam_extr_row,
) -> np.ndarray:
    """
    LiDAR 座標系 → rig 座標系 → カメラ座標系 の 2 段変換。
    T_cam_lidar = inv(T_rig_cam) @ T_rig_lidar

    Args:
        lidar_pts_rig: LiDAR 点群（rig 座標系）(N, 3) または LiDAR 座標系 (N, 3)
                       rig 座標系で渡す場合は lidar_extr_row=None にすること
        lidar_extr_row: LiDAR の sensor_extrinsics 行（T_rig_lidar を計算）
        cam_extr_row: カメラの sensor_extrinsics 行（T_rig_cam を計算）

    Returns:
        カメラ座標系での点群 (N, 3)
    """
    N = lidar_pts_rig.shape[0]

    if lidar_extr_row is not None:
        # LiDAR 座標系 → rig 座標系
        T_rig_lidar = _row_to_transform(lidar_extr_row)
        pts_h = np.hstack([lidar_pts_rig, np.ones((N, 1))])  # (N, 4)
        pts_rig = (T_rig_lidar @ pts_h.T).T[:, :3]  # (N, 3)
    else:
        pts_rig = lidar_pts_rig

    # rig 座標系 → カメラ座標系
    T_rig_cam = _row_to_transform(cam_extr_row)
    T_cam_rig = np.linalg.inv(T_rig_cam)

    pts_h = np.hstack([pts_rig, np.ones((N, 1))])  # (N, 4)
    pts_cam = (T_cam_rig @ pts_h.T).T[:, :3]  # (N, 3)
    return pts_cam.astype(np.float32)


# --------------------------------------------------------
# 9. スパース深度マップ生成
# --------------------------------------------------------
def project_points_to_depth_map(
    pts_cam: np.ndarray,
    K_new: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """
    カメラ座標系の3D点をピンホール投影でスパース深度マップに変換する。

    Args:
        pts_cam: カメラ座標系の点群 (N, 3)
        K_new: 3×3 ピンホール内部行列
        out_h, out_w: 出力深度マップサイズ

    Returns:
        (out_h, out_w) float32 array, 0=無効
    """
    depth_map = np.zeros((out_h, out_w), dtype=np.float32)

    # カメラ前方にある点のみ
    valid = pts_cam[:, 2] > 0.1
    pts = pts_cam[valid]
    if len(pts) == 0:
        return depth_map

    fx = K_new[0, 0]
    fy = K_new[1, 1]
    cx = K_new[0, 2]
    cy = K_new[1, 2]

    z = pts[:, 2]
    u = (pts[:, 0] / z * fx + cx).astype(np.int32)
    v = (pts[:, 1] / z * fy + cy).astype(np.int32)

    # 画像内の点のみ保持
    in_bounds = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]

    # 深度衝突がある場合は手前の点を優先
    # numpy による散布（後に上書きされる可能性があるため、手前の点を後から書き込む）
    order = np.argsort(z)[::-1]  # 遠い順に書き込み → 近い点が最後
    depth_map[v[order], u[order]] = z[order]

    return depth_map


# --------------------------------------------------------
# 10. cam2world 行列計算
# --------------------------------------------------------
def compute_cam2world(ego_state, cam_extr_row) -> torch.Tensor:
    """
    T_world_cam = T_world_vehicle @ T_vehicle_cam を計算する。

    ego_state は `Interpolator[EgomotionState](timestamp)` の戻り値。
    ego_state.pose は scipy.spatial.transform.RigidTransform。

    Args:
        ego_state: EgomotionState（補間済み）
        cam_extr_row: カメラの sensor_extrinsics 行

    Returns:
        (1, 4, 4) torch.float32 tensor（cam2world 変換行列）
    """
    pose = ego_state.pose  # scipy.spatial.transform.RigidTransform

    R = pose.rotation.as_matrix()   # (3,3) or (1,3,3)
    t = pose.translation             # (3,) or (1,3)

    # 補間後にバッチ次元が残ることがあるので squeeze
    R = np.array(R, dtype=np.float64)
    t = np.array(t, dtype=np.float64)
    if R.ndim == 3:
        R = R[0]
    if t.ndim == 2:
        t = t[0]

    T_world_veh = np.eye(4, dtype=np.float64)
    T_world_veh[:3, :3] = R
    T_world_veh[:3, 3] = t

    # センサー外部パラメータ: T_rig_cam（vehicle → camera）
    T_rig_cam = _row_to_transform(cam_extr_row)

    # T_world_cam = T_world_vehicle @ T_rig_cam
    T_world_cam = T_world_veh @ T_rig_cam

    return torch.from_numpy(T_world_cam).float().unsqueeze(0)  # (1, 4, 4)


# --------------------------------------------------------
# 11. DINOv2 正規化
# --------------------------------------------------------
_DINOV2_NORMALIZE = tvf.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def normalize_dinov2(image_rgb_uint8: np.ndarray) -> torch.Tensor:
    """
    DINOv2 正規化を適用する。

    Args:
        image_rgb_uint8: RGB 画像 (H, W, 3) uint8

    Returns:
        (1, 3, H, W) float32 tensor
    """
    img = torch.from_numpy(image_rgb_uint8).float() / 255.0  # (H, W, 3) [0,1]
    img = img.permute(2, 0, 1)  # (3, H, W)
    img = _DINOV2_NORMALIZE(img)
    return img.unsqueeze(0)  # (1, 3, H, W)


# --------------------------------------------------------
# 12. Any4D ビュー構築
# --------------------------------------------------------
def build_any4d_view(
    img_tensor: torch.Tensor,
    K_new: np.ndarray,
    depth_map: np.ndarray,
    cam2world_4x4: torch.Tensor,
    idx: int,
    instance: str = "",
) -> dict:
    """
    ALLOWED_VIEW_KEYS に準拠したビュー辞書を生成する。

    Args:
        img_tensor: (1, 3, H, W) float32 tensor（DINOv2 正規化済み）
        K_new: (3, 3) ピンホール内部行列 numpy array
        depth_map: (H, W) float32 スパース深度マップ（0=無効）
        cam2world_4x4: (1, 4, 4) float32 tensor
        idx: ビューインデックス
        instance: インスタンス識別文字列

    Returns:
        Any4D ビュー辞書
    """
    H, W = img_tensor.shape[-2], img_tensor.shape[-1]

    # intrinsics: (1, 3, 3)
    K_tensor = torch.from_numpy(K_new).float().unsqueeze(0)  # (1, 3, 3)

    # depth_z: (1, H, W, 1) — 無効点は 0
    depth_tensor = torch.from_numpy(depth_map).float()
    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

    # true_shape: (1, 2) [H, W]
    true_shape = torch.tensor([[H, W]], dtype=torch.long)

    # is_metric_scale: (1,) True（LiDAR 由来の実スケール深度）
    is_metric = torch.ones(1, dtype=torch.bool)

    view = {
        "img": img_tensor,                    # (1, 3, H, W)
        "data_norm_type": ["dinov2"],          # list[str]
        "intrinsics": K_tensor,               # (1, 3, 3)
        "depth_z": depth_tensor,              # (1, H, W, 1)
        "camera_poses": cam2world_4x4,        # (1, 4, 4)
        "is_metric_scale": is_metric,         # (1,)
        "true_shape": true_shape,             # (1, 2)
        "idx": idx,
        "instance": instance,
    }
    return view


# --------------------------------------------------------
# 13. メイン推論
# --------------------------------------------------------
def run_inference(views: list[dict], args) -> dict:
    """
    Any4D 推論を実行する。

    1. validate_input_views_for_inference(views)
    2. preprocess_input_views_for_inference(views)
    3. sample_inference(model, preprocessed_views, ...)

    Args:
        views: build_any4d_view で生成したビューのリスト
        args: argparse.Namespace

    Returns:
        推論結果辞書
    """
    # モデル設定
    config = {
        "path": os.path.join(args.config_dir, "train.yaml"),
        "config_overrides": [
            f"machine={args.machine}",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            "model/task=images_only",
        ],
        "checkpoint_path": args.checkpoint_path,
        "trained_with_amp": True,
        "data_norm_type": "dinov2",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] device={device}")

    # モデル初期化
    model = init_inference_model(config, args.checkpoint_path, device)

    # scene_flow 出力を有効化: model.scene_rep_type を実行時にパッチ
    # any4d_4v_combined.pth は scene_flow ヘッドを持つため、
    # scene_rep_type を変えるだけで scene_flow がロールアウトされる
    original_scene_rep_type = model.scene_rep_type
    model.scene_rep_type = "raydirs+depth+pose+scene_flow+confidence+mask"
    print(f"[inference] scene_rep_type: {original_scene_rep_type} → {model.scene_rep_type}")

    # ビュー検証
    print("[inference] validate_input_views_for_inference ...")
    validated_views = validate_input_views_for_inference(views)

    # ビュー前処理
    print("[inference] preprocess_input_views_for_inference ...")
    preprocessed_views = preprocess_input_views_for_inference(validated_views)

    # 推論
    print(f"[inference] sample_inference ({len(preprocessed_views)} views) ...")
    result = sample_inference(model, preprocessed_views, device, use_amp=True)

    return result


# --------------------------------------------------------
# 14. クリップ一覧表示
# --------------------------------------------------------
def list_clips(dataset, n: int = 20):
    """
    利用可能なクリップ一覧を表示する（上位 n 件）。
    data_collection メタデータから country / month / hour_of_day / platform_class を表示する。
    """
    ci = dataset.clip_index
    sp = dataset.sensor_presence

    # 追加メタデータを取得（data_collection に country/month/hour_of_day/platform_class が含まれる）
    meta_series: dict[str, object] = {}
    try:
        dataset.download_metadata()
        extra_meta = getattr(dataset, "metadata", {})
        for meta_df in extra_meta.values():
            for col in ["country", "month", "hour_of_day", "platform_class"]:
                if col in meta_df.columns and col not in meta_series:
                    meta_series[col] = meta_df[col]
    except Exception as e:
        print(f"  [警告] 追加メタデータ取得失敗: {e}")

    # platform_class ごとのカウントを集計して先頭に表示
    if "platform_class" in meta_series:
        platform_counts: dict = {}
        for clip_id in ci.index:
            p = meta_series["platform_class"].get(clip_id, "unknown")
            platform_counts[p] = platform_counts.get(p, 0) + 1
        print(f"\n=== プラットフォーム別クリップ数 ===")
        for p, cnt in sorted(platform_counts.items(), key=lambda x: -x[1]):
            print(f"  {p:<20s}: {cnt} clips")

    # ヘッダー行
    print(f"\n=== 利用可能なクリップ一覧（上位 {n} 件 / 全 {len(ci)} 件） ===")
    col_country   = 14
    col_month     = 5
    col_hour      = 5
    col_platform  = 14
    col_cameras   = 4
    col_lidar     = 5
    header = (
        f"  {'#':>3}  {'clip_id':<36}  {'split':<6}  "
        f"{'country':<{col_country}}  {'month':>{col_month}}  "
        f"{'hour':>{col_hour}}  {'platform':<{col_platform}}  "
        f"{'cams':>{col_cameras}}  {'lidar':<{col_lidar}}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    cam_cols = [c for c in sp.columns if c.startswith("camera_")]
    for i, clip_id in enumerate(ci.index[:n]):
        split = ci.at[clip_id, "split"] if "split" in ci.columns else "?"

        # sensor presence（boolean Seriesは .astype(int).sum() で正しくカウント）
        n_cams = int(sp.loc[clip_id, cam_cols].astype(int).sum()) if clip_id in sp.index else -1
        has_lidar = bool(sp.at[clip_id, "lidar_top_360fov"]) if (clip_id in sp.index and "lidar_top_360fov" in sp.columns) else False

        # 追加メタデータ
        country  = str(meta_series["country"].get(clip_id, "-"))       if "country"       in meta_series else "-"
        month    = str(meta_series["month"].get(clip_id, "-"))          if "month"         in meta_series else "-"
        hour     = str(meta_series["hour_of_day"].get(clip_id, "-"))   if "hour_of_day"   in meta_series else "-"
        platform = str(meta_series["platform_class"].get(clip_id, "-")) if "platform_class" in meta_series else "-"

        lidar_str = "yes" if has_lidar else "no"
        print(
            f"  {i:>3}  {clip_id:<36}  {split:<6}  "
            f"{country:<{col_country}}  {month:>{col_month}}  "
            f"{hour:>{col_hour}}  {platform:<{col_platform}}  "
            f"{n_cams:>{col_cameras}}  {lidar_str:<{col_lidar}}"
        )
    print()


# --------------------------------------------------------
# 15. フレーム選択（等間隔）
# --------------------------------------------------------
def select_frame_indices(total_frames: int, n_frames: int) -> list[int]:
    """
    total_frames フレームから n_frames 個を等間隔で選択する。
    """
    if n_frames >= total_frames:
        return list(range(total_frames))
    step = total_frames / n_frames
    return [int(i * step) for i in range(n_frames)]


def compute_static_mask(
    result: dict,
    num_views: int,
    H: int,
    W: int,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    全時系列フレームの scene_flow magnitude を集約し、static 画素マスクを返す。

    あるピクセルが「静的」とみなされるのは、全ての非参照フレームとの比較で
    scene_flow ノルム < threshold（メートル）の場合。

    Args:
        result: sample_inference の戻り値辞書
        num_views: 総ビュー数（reference 1 + temporal N-1）
        H, W: 画像の高さ・幅
        threshold: 動的とみなす scene_flow ノルムの閾値 [m]

    Returns:
        (H, W) bool tensor。True = 静的画素
    """
    dynamic_accum = torch.zeros(H, W, dtype=torch.bool)
    n_sf_views = 0
    for i in range(1, num_views):
        pred = result.get(f"pred{i+1}", {})
        if "scene_flow" not in pred:
            continue
        sf = pred["scene_flow"][0].cpu()  # (H, W, 3)
        magnitude = sf.norm(dim=-1)       # (H, W)
        dynamic_accum |= (magnitude > threshold)
        n_sf_views += 1

    if n_sf_views == 0:
        print("  [static_mask] scene_flow が見つかりません。全画素を static とみなします。")
        return torch.ones(H, W, dtype=torch.bool)

    print(f"  [static_mask] {n_sf_views} フレームの scene_flow を使用 "
          f"(threshold={threshold} m)")
    return ~dynamic_accum


def select_temporal_frame_indices(total_frames: int, n_frames: int) -> list[int]:
    """
    時系列推論用フレームインデックスを選択する。
    動画中央のフレームを参照フレームとして、そこから連続する n_frames 個を返す。
    リスト先頭が参照フレーム（view0）。

    例: total_frames=605, n_frames=4 → [302, 303, 304, 305]
    """
    ref = total_frames // 2
    indices = [min(ref + i, total_frames - 1) for i in range(n_frames)]
    return indices


# --------------------------------------------------------
# 16a. 点群レンダリング
# --------------------------------------------------------
def render_pointcloud_to_image(
    pts3d_cam0: np.ndarray,
    colors_rgb: np.ndarray,
    K: np.ndarray,
    T_cam0_cami: np.ndarray,
    out_h: int,
    out_w: int,
    mask: np.ndarray | None = None,
    point_radius: int = 1,
) -> np.ndarray:
    """
    点群 (cam0 座標系) をカメラ i の視点からレンダリングして (out_h, out_w, 3) uint8 を返す。

    Args:
        pts3d_cam0: cam0 座標系の点群 (H, W, 3) または (N, 3)
        colors_rgb: 対応する RGB 色 (H, W, 3) または (N, 3) uint8
        K: ピンホール内部行列 (3, 3)
        T_cam0_cami: cam_i → cam0 変換行列 (4, 4)。
                     逆行列で cam0 → cam_i 変換として使用する。
        out_h, out_w: 出力解像度
        mask: 有効点マスク (H, W) または (N,) bool
        point_radius: 膨張半径 (pixel)。0 で膨張なし。

    Returns:
        (out_h, out_w, 3) uint8 RGB レンダリング画像。背景は黒。
    """
    pts = pts3d_cam0.reshape(-1, 3).astype(np.float32)
    cols = colors_rgb.reshape(-1, 3).astype(np.uint8)

    if mask is not None:
        m = mask.reshape(-1).astype(bool)
        pts = pts[m]
        cols = cols[m]

    if len(pts) == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # T_cami_cam0 = inv(T_cam0_cami) : cam0 → cam_i
    T_np = np.array(T_cam0_cami, dtype=np.float64)
    T_cami_cam0 = np.linalg.inv(T_np)

    N = pts.shape[0]
    pts_h = np.hstack([pts.astype(np.float64), np.ones((N, 1))])  # (N, 4)
    pts_cami = (T_cami_cam0 @ pts_h.T).T[:, :3].astype(np.float32)  # (N, 3)

    # カメラ前方の点のみ
    valid = pts_cami[:, 2] > 0.01
    pts_cami = pts_cami[valid]
    cols = cols[valid]

    if len(pts_cami) == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # ピンホール投影
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = pts_cami[:, 2]
    u = pts_cami[:, 0] / z * fx + cx
    v = pts_cami[:, 1] / z * fy + cy

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < out_w) & (vi >= 0) & (vi < out_h)
    ui, vi, z, cols = ui[in_bounds], vi[in_bounds], z[in_bounds], cols[in_bounds]

    if len(ui) == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # 遠→近の順で描画 (painter's algorithm)
    order = np.argsort(z)[::-1]
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[vi[order], ui[order]] = cols[order]

    # 点を膨張させて穴を減らす
    if point_radius > 0:
        k = 2 * point_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(canvas, kernel)
        bg = np.all(canvas == 0, axis=-1)
        canvas[bg] = dilated[bg]

    return canvas


def save_pointcloud_render_mp4(
    result: dict,
    preprocessed_views: list,
    output_path: str,
    fps: float = 10.0,
    max_depth: float = 40.0,
    sf_threshold: float = 0.1,
    point_radius: int = 1,
    side_by_side: bool = True,
) -> None:
    """
    推論結果の点群を各フレームのカメラ視点からレンダリングして MP4 に保存する。

    reference frame (pred1) の点群 (cam0 座標系) を、各時刻フレームで予測された
    カメラ姿勢に基づいて投影し、1フレームずつ OpenCV で描画して動画にまとめる。

    Args:
        result: sample_inference の戻り値
        preprocessed_views: preprocess_input_views_for_inference 後のビューリスト
        output_path: 出力 MP4 パス (.mp4)
        fps: フレームレート
        max_depth: レンダリング対象の最大深度 [m]
        sf_threshold: 動的判定閾値 [m]
        point_radius: 点の描画膨張半径 (pixel)
        side_by_side: True の場合、元画像と点群レンダリングを横並びで出力
    """
    from any4d.utils.geometry import (
        quaternion_to_rotation_matrix,
        recover_pinhole_intrinsics_from_ray_directions,
    )
    from any4d.utils.image import rgb as to_rgb

    n_views = len(preprocessed_views)

    # ray_directions から inference 解像度の K を復元
    ray_dirs = result["pred1"]["ray_directions"][0].cpu()  # (H, W, 3)
    K_infer = recover_pinhole_intrinsics_from_ray_directions(ray_dirs).numpy()  # (3, 3)

    # reference frame の点群と色
    pts3d = result["pred1"]["pts3d"][0].cpu().numpy()       # (H, W, 3) in cam0
    H_inf, W_inf = pts3d.shape[:2]
    # to_rgb は [0, 1] float32 を返すため uint8 に変換
    colors_rgb = (to_rgb(preprocessed_views[0]["img"], norm_type="dinov2")[0] * 255).astype(np.uint8)  # (H, W, 3)

    # static + 深度マスク
    static_mask = compute_static_mask(result, n_views, H_inf, W_inf, threshold=sf_threshold)
    depth_z_ref = pts3d[..., 2]
    depth_mask = (depth_z_ref > 0.01) & (depth_z_ref < max_depth)
    combined_mask = static_mask.numpy() & depth_mask  # (H, W) bool

    frame_w = W_inf * 2 if side_by_side else W_inf
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, H_inf))

    print(f"[render_mp4] {n_views} フレームをレンダリング中 → {output_path}")
    for i in range(n_views):
        pred_key = f"pred{i + 1}"
        if "cam_quats" in result.get(pred_key, {}):
            cam_quats = result[pred_key]["cam_quats"][0].cpu()
            cam_trans = result[pred_key]["cam_trans"][0].cpu()
            cam_rot = quaternion_to_rotation_matrix(cam_quats)
            T_cam0_cami = torch.eye(4)
            T_cam0_cami[:3, :3] = cam_rot
            T_cam0_cami[:3, 3] = cam_trans
            T_cam0_cami_np = T_cam0_cami.numpy()
        else:
            # cam_quats が無い場合は identity（reference frame）
            T_cam0_cami_np = np.eye(4, dtype=np.float64)

        rendered = render_pointcloud_to_image(
            pts3d, colors_rgb, K_infer, T_cam0_cami_np,
            H_inf, W_inf, mask=combined_mask, point_radius=point_radius,
        )
        rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

        if side_by_side:
            orig_rgb = (to_rgb(preprocessed_views[i]["img"], norm_type="dinov2")[0] * 255).astype(np.uint8)
            orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
            if orig_bgr.shape[:2] != (H_inf, W_inf):
                orig_bgr = cv2.resize(orig_bgr, (W_inf, H_inf))
            frame_bgr = np.concatenate([orig_bgr, rendered_bgr], axis=1)
        else:
            frame_bgr = rendered_bgr

        writer.write(frame_bgr)
        print(f"  フレーム {i + 1}/{n_views} 完了")

    writer.release()
    print(f"[render_mp4] 保存完了: {output_path}")


# --------------------------------------------------------
# 16. メイン処理
# --------------------------------------------------------
def main():
    parser = get_parser()
    script_add_rerun_args(parser)
    args = parser.parse_args()

    seed_everything(0)

    # ---- データセット初期化（メタデータのみ）----
    try:
        from physical_ai_av import PhysicalAIAVDatasetInterface
    except ImportError:
        raise ImportError(
            "physical-ai-av パッケージが見つかりません。"
            " `pip install physical-ai-av` を実行してください。"
        )

    os.makedirs(args.cache_dir, exist_ok=True)
    dataset = PhysicalAIAVDatasetInterface(local_dir=args.cache_dir)

    # ---- クリップ一覧モード ----
    if args.list_clips:
        list_clips(dataset, n=20)
        return

    # ---- クリップ ID 決定 ----
    if args.clip_id is not None:
        clip_id = args.clip_id
    else:
        clip_id = auto_select_clip(
            dataset,
            filter_country=args.filter_country,
            filter_hour=args.filter_hour,
        )

    print(f"[main] clip_id = {clip_id}")

    # ---- 使用カメラ決定 ----
    # Approach A: 時系列推論モード
    # モデルは「同一カメラの N 時刻フレーム」で学習されているため、
    # デフォルトは1台のカメラの複数フレームを使う。
    if args.cameras:
        cameras = list(args.cameras)
    else:
        cameras = ["camera_front_wide_120fov"]
        print(f"[main] 時系列モード: カメラ1台を使用 ({cameras[0]})")

    if len(cameras) > 1:
        print(f"[main] 警告: 複数カメラが指定されています ({cameras})。"
              "モデルは時系列単一カメラ向けに学習されており、"
              "空間マルチカメラでは性能が低下する可能性があります。")

    num_cameras = len(cameras)

    # ---- フレーム数決定 ----
    # 学習設定に合わせて 4 フレーム（reference 1 + temporal 3）をデフォルトとする
    if args.num_frames is None:
        if num_cameras == 1:
            n_frames = 4  # 学習設定に合わせた値
        else:
            n_frames = max(1, min(8 // num_cameras, 4))
    else:
        n_frames = args.num_frames
    print(f"[main] cameras={cameras}, n_frames={n_frames}/camera")

    # ---- データダウンロード ----
    dataset = download_single_clip(clip_id, cameras, args.cache_dir)

    # ---- 出力ディレクトリ準備 ----
    os.makedirs(args.output_dir, exist_ok=True)
    undist_dir = os.path.join(args.output_dir, "undistorted")
    depth_dir = os.path.join(args.output_dir, "depth")
    os.makedirs(undist_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # ---- フィーチャーデータ読み込み ----
    # get_clip_feature() の戻り値:
    #   egomotion          → Interpolator[EgomotionState]
    #   camera_*           → SeekVideoReader (.timestamps / .decode_images_from_frame_indices)
    #   camera_intrinsics  → DataFrame (pd.read_parquet().loc[clip_id])
    #   sensor_extrinsics  → DataFrame (pd.read_parquet().loc[clip_id])
    #   lidar_top_360fov   → dict {key: DataFrame or BytesIO}

    # maybe_stream=True: ローカルキャッシュがない場合は HF Hub からストリーミング
    print("[main] egomotion をロード中...")
    ego_interpolator = dataset.get_clip_feature(clip_id, "egomotion", maybe_stream=True)
    print(f"  egomotion time_range: {ego_interpolator.time_range}")

    print("[main] camera_intrinsics をロード中...")
    intr_df = dataset.get_clip_feature(clip_id, "camera_intrinsics", maybe_stream=True)

    print("[main] sensor_extrinsics をロード中...")
    extr_df = dataset.get_clip_feature(clip_id, "sensor_extrinsics", maybe_stream=True)

    print("[main] lidar_top_360fov をロード中...")
    try:
        lidar_data = dataset.get_clip_feature(clip_id, "lidar_top_360fov", maybe_stream=True)
        print(f"  LiDAR データキー: {list(lidar_data.keys()) if isinstance(lidar_data, dict) else type(lidar_data)}")
    except Exception as e:
        print(f"  警告: LiDAR データ取得に失敗しました: {e}")
        lidar_data = None

    # ---- 解像度設定 ----
    out_h = args.target_h
    # アスペクト比に基づいて out_w を決定（14の倍数, RESOLUTION_MAPPINGS 参照）
    # デフォルト: (518, 336) = 3:2
    out_w = DEFAULT_TARGET_W
    print(f"[main] 出力解像度: {out_w}×{out_h}")

    # ---- 各カメラのフレーム処理 ----
    all_views = []
    view_idx_global = 0

    # LiDAR timestamps / frames をパース（一度だけ）
    lidar_timestamps, lidar_frames = _parse_lidar_data(lidar_data)

    for cam_name in cameras:
        print(f"\n[process] カメラ: {cam_name}")

        # ---- SeekVideoReader で動画を開く ----
        try:
            cam_reader = dataset.get_clip_feature(clip_id, cam_name, maybe_stream=True)
        except Exception as e:
            print(f"  警告: {cam_name} の動画取得に失敗しました: {e}。スキップします。")
            continue

        total_frames = len(cam_reader.timestamps)
        if num_cameras == 1:
            # 時系列モード: 動画中央から連続フレームを選択
            frame_indices = select_temporal_frame_indices(total_frames, n_frames)
        else:
            # 空間マルチカメラモード: 等間隔サンプリング
            frame_indices = select_frame_indices(total_frames, n_frames)
        print(f"  全{total_frames}フレーム中 {len(frame_indices)} フレームを選択 (indices={frame_indices})")

        # このカメラの intrinsics 行を取得
        intr_row = _get_sensor_row(intr_df, cam_name)
        if intr_row is None:
            print(f"  警告: {cam_name} の intrinsics が見つかりません。スキップします。")
            continue

        # このカメラの extrinsics 行を取得
        cam_extr_row = _get_sensor_row(extr_df, cam_name)
        if cam_extr_row is None:
            print(f"  警告: {cam_name} の extrinsics が見つかりません。スキップします。")
            continue

        # LiDAR の extrinsics
        lidar_extr_row = _get_sensor_row(extr_df, "lidar_top_360fov")

        # 元画像サイズを最初のフレームから取得
        try:
            first_frames = cam_reader.decode_images_from_frame_indices(
                np.array([0], dtype=np.int64)
            )
            orig_h, orig_w = first_frames[0].shape[:2]
        except Exception as e:
            print(f"  警告: {cam_name} の画像デコードに失敗: {e}。スキップします。")
            continue
        print(f"  元解像度: {orig_w}×{orig_h}")

        # F-theta アンディストーション写像を計算
        map_x, map_y, K_new = compute_ftheta_undistort_map(
            intr_row, orig_h, orig_w, out_h, out_w, fov_deg=DEFAULT_TARGET_FOV_DEG
        )
        print(f"  K_new(fx={K_new[0,0]:.1f}, fy={K_new[1,1]:.1f}, cx={K_new[0,2]:.1f}, cy={K_new[1,2]:.1f})")

        for fi, frame_idx in enumerate(frame_indices):
            # タイムスタンプ（egomotion と同一単位 — 通常ナノ秒）
            timestamp = int(cam_reader.timestamps[frame_idx])

            # ---- 画像デコード (SeekVideoReader → RGB) ----
            try:
                decoded = cam_reader.decode_images_from_frame_indices(
                    np.array([frame_idx], dtype=np.int64)
                )
                img_rgb = decoded[0]  # (H, W, 3) uint8 RGB
            except Exception as e:
                print(f"  フレーム {frame_idx} のデコードに失敗: {e}。スキップ。")
                continue

            # ---- アンディストーション ----
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_undist_bgr = undistort_image(img_bgr, map_x, map_y)
            img_undist_rgb = cv2.cvtColor(img_undist_bgr, cv2.COLOR_BGR2RGB)

            # アンディストーション結果を保存（デバッグ用）
            save_path = os.path.join(undist_dir, f"{cam_name}_frame{frame_idx:04d}.jpg")
            cv2.imwrite(save_path, img_undist_bgr)

            # ---- LiDAR スパース深度マップ生成 ----
            depth_map = np.zeros((out_h, out_w), dtype=np.float32)
            if lidar_timestamps is not None and lidar_frames is not None:
                try:
                    lidar_idx = int(np.argmin(np.abs(lidar_timestamps - timestamp)))
                    lidar_bytes = lidar_frames[lidar_idx]
                    if lidar_bytes is not None:
                        lidar_pts = decode_lidar_spin(lidar_bytes)  # (N, 3)
                        pts_cam = transform_lidar_to_camera(
                            lidar_pts, lidar_extr_row, cam_extr_row
                        )
                        depth_map = project_points_to_depth_map(pts_cam, K_new, out_h, out_w)
                except Exception as e:
                    print(f"  LiDAR 処理エラー (フレーム {frame_idx}): {e}")

            # 深度マップ保存（デバッグ用）
            depth_vis = depth_map / (depth_map.max() + 1e-6) * 255.0
            depth_save_path = os.path.join(depth_dir, f"{cam_name}_frame{frame_idx:04d}_depth.png")
            cv2.imwrite(depth_save_path, depth_vis.astype(np.uint8))

            # ---- cam2world 計算 (Interpolator[EgomotionState]) ----
            try:
                ego_state = ego_interpolator(timestamp)
                cam2world = compute_cam2world(ego_state, cam_extr_row)
            except Exception as e:
                print(f"  警告: egomotion 補間エラー (ts={timestamp}): {e}。単位行列を使用。")
                cam2world = torch.eye(4, dtype=torch.float32).unsqueeze(0)

            # ---- DINOv2 正規化 ----
            img_tensor = normalize_dinov2(img_undist_rgb)  # (1, 3, H, W)

            # ---- Any4D ビュー構築 ----
            view = build_any4d_view(
                img_tensor=img_tensor,
                K_new=K_new,
                depth_map=depth_map,
                cam2world_4x4=cam2world,
                idx=view_idx_global,
                instance=f"{cam_name}_frame{frame_idx:04d}",
            )
            all_views.append(view)
            view_idx_global += 1

    if len(all_views) == 0:
        print("[main] エラー: 有効なビューが1つも構築できませんでした。")
        return

    print(f"\n[main] 合計 {len(all_views)} ビューで推論を実行します。")

    # ---- カメラ姿勢をビュー0基準の相対座標に変換 ----
    # モデルは学習時に view0 = 参照フレームとして相対姿勢で訓練されている。
    # compute_cam2world が返す T_world_cam は絶対ワールド座標なので、
    # T_cam0_cami = T_world_cam0^{-1} @ T_world_cami に変換する。
    if "camera_poses" in all_views[0]:
        T_world_cam0 = all_views[0]["camera_poses"][0]   # (4, 4) cam0→world
        T_cam0_world = torch.inverse(T_world_cam0)        # (4, 4) world→cam0
        for view in all_views:
            T_world_cami = view["camera_poses"][0]        # (4, 4) cami→world
            T_cam0_cami = T_cam0_world @ T_world_cami     # (4, 4) cami→cam0
            view["camera_poses"] = T_cam0_cami.unsqueeze(0)  # (1, 4, 4)
        print("[main] カメラ姿勢をビュー0基準の相対座標に変換しました。")

    # ---- rerun 初期化 ----
    if args.viz:
        import rerun as rr
        rr.script_setup(args, "PhysAIAV_Any4D")
        rr.connect_grpc(f"rerun+http://127.0.0.1:{args.port}/proxy", flush_timeout_sec=None)
        rr.set_time("stable_time", timestamp=0)
        rr.log("pred", rr.ViewCoordinates.RDF, static=True)

    # ---- 推論実行 ----
    result = run_inference(all_views, args)

    print("[main] 推論完了。")

    # ---- 可視化 ----
    if args.viz:
        import rerun as rr
        # preprocessed views での可視化のため、再度前処理を実行
        validated = validate_input_views_for_inference(all_views)
        preprocessed = preprocess_input_views_for_inference(validated)

        # 可視化に必要なマスクを追加（推論には不要なため前処理後に付与）
        H_v = preprocessed[0]["img"].shape[-2]
        W_v = preprocessed[0]["img"].shape[-1]
        for v in preprocessed:
            v["non_ambiguous_mask"] = torch.ones(H_v, W_v, dtype=torch.bool)
            v["binary_mask"] = torch.ones(H_v, W_v, dtype=torch.bool)

        # args に必要な属性を追加
        if not hasattr(args, "use_scene_flow_mask_refined"):
            args.use_scene_flow_mask_refined = False

        # ---- static マスク計算 ----
        # 全時系列フレームの scene_flow magnitude を集約し、
        # 閾値以上の動きがある画素を動的とみなして除去する
        static_mask = compute_static_mask(result, len(preprocessed), H_v, W_v,
                                          threshold=args.sf_threshold)
        n_static = int(static_mask.sum().item())
        n_total = H_v * W_v
        print(f"[viz] static マスク: {n_static}/{n_total} 画素 "
              f"({100*n_static/n_total:.1f}%) が静的")

        # static マスクを reference view の non_ambiguous_mask に適用
        preprocessed[0]["non_ambiguous_mask"] = static_mask

        # デモと同様に reference(view0) + 各時刻フレームのペアで可視化
        # ref フレームのみ先に可視化（時刻 0）
        rr.set_time("stable_time", timestamp=0)
        ref_pair_views = [preprocessed[0], preprocessed[min(1, len(preprocessed)-1)]]
        ref_pair_pred = {
            "pred1": result["pred1"],
            "pred2": result[f"pred{min(2, len(preprocessed))}"],
        }
        visualize_raw_custom_data_inference_output(
            args, ref_pair_views, ref_pair_pred,
            img_norm_type="dinov2", use_scene_flow_type="allo_scene_flow",
        )

        # 各時刻フレームを reference との時間軸でアニメーション表示
        for i in range(1, len(preprocessed)):
            rr.set_time("stable_time", timestamp=i * 0.1)
            pair_views = [preprocessed[0], preprocessed[i]]
            pair_pred = {
                "pred1": result["pred1"],
                "pred2": result[f"pred{i+1}"],
            }
            visualize_raw_custom_data_inference_output(
                args, pair_views, pair_pred,
                img_norm_type="dinov2", use_scene_flow_type="allo_scene_flow",
            )

        rr.script_teardown(args)

    # ---- MP4 レンダリング保存 ----
    if args.save_mp4 is not None:
        mp4_path = (
            os.path.join(args.output_dir, "pointcloud_render.mp4")
            if args.save_mp4 == "auto"
            else args.save_mp4
        )
        validated_for_mp4 = validate_input_views_for_inference(all_views)
        preprocessed_for_mp4 = preprocess_input_views_for_inference(validated_for_mp4)
        H_v = preprocessed_for_mp4[0]["img"].shape[-2]
        W_v = preprocessed_for_mp4[0]["img"].shape[-1]
        for v in preprocessed_for_mp4:
            if "non_ambiguous_mask" not in v:
                v["non_ambiguous_mask"] = torch.ones(H_v, W_v, dtype=torch.bool)
            if "binary_mask" not in v:
                v["binary_mask"] = torch.ones(H_v, W_v, dtype=torch.bool)

        save_pointcloud_render_mp4(
            result=result,
            preprocessed_views=preprocessed_for_mp4,
            output_path=mp4_path,
            fps=args.mp4_fps,
            max_depth=args.max_depth,
            sf_threshold=args.sf_threshold,
            point_radius=1,
            side_by_side=not args.mp4_no_side_by_side,
        )

    print(f"[main] 出力ディレクトリ: {args.output_dir}")
    print("  - アンディストーション画像: undistorted/")
    print("  - スパース深度マップ: depth/")
    if args.save_mp4 is not None:
        print(f"  - 点群レンダリング MP4: {mp4_path}")


# --------------------------------------------------------
# ヘルパー関数
# --------------------------------------------------------
def _get_sensor_row(df, sensor_name: str):
    """
    DataFrame から指定センサー名の行を取得する。
    インデックスまたは 'sensor_name' カラムで検索する。
    """
    if df is None:
        return None
    if sensor_name in df.index:
        return df.loc[sensor_name]
    if "sensor_name" in df.columns:
        rows = df[df["sensor_name"] == sensor_name]
        if len(rows) > 0:
            return rows.iloc[0]
    # カラム名 'name' でも検索
    if "name" in df.columns:
        rows = df[df["name"] == sensor_name]
        if len(rows) > 0:
            return rows.iloc[0]
    return None


def _parse_lidar_data(lidar_data) -> tuple[np.ndarray | None, list | None]:
    """
    get_clip_feature('lidar_top_360fov') の戻り値 (dict) から
    タイムスタンプ配列とフレームごとの Draco バイト列リストを抽出する。

    Returns:
        (timestamps_ns, frames_bytes_list) or (None, None) if parsing fails
    """
    if lidar_data is None:
        return None, None

    if not isinstance(lidar_data, dict):
        print(f"  LiDAR データの型が予期しない形式です: {type(lidar_data)}")
        return None, None

    timestamps = None
    frames = None

    for key, val in lidar_data.items():
        # タイムスタンプ DataFrame を探す（PhysAI-AV 形式を含む）
        if hasattr(val, "columns"):
            for ts_col in ["reference_timestamp", "timestamp", "timestamps", "time"]:
                if ts_col in val.columns:
                    timestamps = val[ts_col].to_numpy()
                    break
            # Draco バイト列カラムを探す（PhysAI-AV: draco_encoded_pointcloud）
            for bytes_col in ["draco_encoded_pointcloud", "pointcloud", "draco_bytes"]:
                if bytes_col in val.columns:
                    frames = val[bytes_col].tolist()
                    break
        # Draco バイト列 / リストを探す（BytesIO の場合）
        elif hasattr(val, "read"):
            val.seek(0)
            raw = val.read()
            frames = [raw]

    if timestamps is None:
        print("  LiDAR タイムスタンプが見つかりませんでした。")
        print(f"  LiDAR data keys: { {k: type(v).__name__ for k, v in lidar_data.items()} }")
        return None, None

    if frames is None:
        print("  LiDAR Draco データが見つかりませんでした。")
        return None, None

    return timestamps, frames


# --------------------------------------------------------
# 引数パーサー
# --------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser(
        description="PhysicalAI-AV → Any4D 4D再構成スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # クリップ指定
    parser.add_argument(
        "--clip_id",
        type=str,
        default=None,
        help="クリップ UUID（省略時は条件フィルタで自動選択）",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="使用するカメラ名（スペース区切り複数指定可）。省略時は自動選択。",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="1カメラあたりの選択フレーム数（単一カメラ時は省略で 4、複数カメラ時は auto）",
    )

    # フィルタ
    parser.add_argument(
        "--filter_country",
        type=str,
        default=None,
        help="自動選択時の国コードフィルタ (例: US, DE)",
    )
    parser.add_argument(
        "--filter_hour",
        type=int,
        default=None,
        help="自動選択時の時間帯フィルタ (0-23)",
    )

    # ディレクトリ
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/physai_av_cache",
        help="ダウンロードキャッシュ先",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/physai_av",
        help="出力先ディレクトリ",
    )

    # 解像度
    parser.add_argument(
        "--target_h",
        type=int,
        default=DEFAULT_TARGET_H,
        help="アンディストーション後の高さ（pixels）",
    )

    # モデル設定
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/any4d_4v_combined.pth",
        help="Any4D チェックポイントパス",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Hydra 設定ディレクトリ",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default="local",
        help="Hydra machine 設定名",
    )

    # 可視化
    parser.add_argument(
        "--viz",
        action="store_true",
        help="rerun 可視化を有効化",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9877,
        help="rerun ポート",
    )
    parser.add_argument(
        "--sf_threshold",
        type=float,
        default=0.1,
        help="scene_flow の動的判定閾値 [m]。この値以上移動した画素を動的とみなし除去する。",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=40.0,
        help="可視化する最大深度 [m]。デフォルト40m。遠景を表示したい場合は増やす（例: 100.0）",
    )

    # ユーティリティ
    parser.add_argument(
        "--list_clips",
        action="store_true",
        help="利用可能なクリップ一覧を表示して終了",
    )

    # MP4 レンダリング保存
    parser.add_argument(
        "--save_mp4",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help=(
            "点群レンダリング MP4 の保存パス。"
            "パスなしで指定した場合は output_dir/pointcloud_render.mp4 に保存。"
        ),
    )
    parser.add_argument(
        "--mp4_fps",
        type=float,
        default=10.0,
        help="MP4 フレームレート",
    )
    parser.add_argument(
        "--mp4_no_side_by_side",
        action="store_true",
        help="点群レンダリングのみ出力（デフォルトは元画像と横並び）",
    )

    return parser


if __name__ == "__main__":
    main()
