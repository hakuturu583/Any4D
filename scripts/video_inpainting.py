"""
動画インペインティングモジュール

ProPainter (ICCV 2023) を使用して、点群が投影されない画素（動的物体除去後の穴）を
動画拡散モデルで自然に補完する。

使用方法:
    from video_inpainting import save_sliding_window_inpainted_mp4
    save_sliding_window_inpainted_mp4(...)

ProPainter セットアップ (初回のみ):
    git clone https://github.com/hakuturu583/ProPainter third_party/ProPainter
    # 初回実行時に自動的に重みがダウンロードされます
"""

from __future__ import annotations

import gc
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image


PROPAINTER_REPO = "https://github.com/hakuturu583/ProPainter"
# third_party/ProPainter を基準とするデフォルトパス (scripts/ の1つ上)
_DEFAULT_PROPAINTER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party",
    "ProPainter",
)


def _ensure_propainter(propainter_dir: str = _DEFAULT_PROPAINTER_DIR) -> None:
    """
    ProPainter が third_party/ に存在しなければ git clone し、sys.path に追加する。
    """
    if propainter_dir not in sys.path:
        if not os.path.isdir(propainter_dir):
            import subprocess

            os.makedirs(os.path.dirname(propainter_dir), exist_ok=True)
            print(f"[inpaint] ProPainter をクローン中: {PROPAINTER_REPO} → {propainter_dir}")
            subprocess.check_call(["git", "clone", PROPAINTER_REPO, propainter_dir])
        sys.path.insert(0, propainter_dir)
        print(f"[inpaint] sys.path に追加: {propainter_dir}")


def generate_inpaint_masks(
    validity_mask: np.ndarray,
    dilate_px: int = 8,
) -> tuple[Image.Image, Image.Image]:
    """
    点群の validity_mask からインペインティング用マスクを生成する。

    Args:
        validity_mask: (H, W) bool。True = 点が投影された画素、False = 穴（補完対象）
        dilate_px: マスクの膨張量 (px)。ProPainter のフロー補完用に少し大きめにする。

    Returns:
        (flow_mask_pil, mask_dilated_pil): PIL Image (mode='L', 255=マスク領域)
            - flow_mask_pil: dilate_px 膨張 (ProPainter の flow 補完用)
            - mask_dilated_pil: 5px 膨張 (inpaint 用)
    """
    hole = (~validity_mask).astype(np.uint8) * 255  # (H, W) uint8

    kernel_flow = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    )
    flow_mask = cv2.dilate(hole, kernel_flow)

    kernel_inpaint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_dilated = cv2.dilate(hole, kernel_inpaint)

    flow_mask_pil = Image.fromarray(flow_mask, mode="L")
    mask_dilated_pil = Image.fromarray(mask_dilated, mode="L")

    return flow_mask_pil, mask_dilated_pil


def _pad_to_multiple(img: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, tuple]:
    """
    画像を multiple の倍数サイズにパディングする。

    Returns:
        (padded, (pad_h, pad_w)): パディング後の画像とパディング量
    """
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    if img.ndim == 3:
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    else:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, (pad_h, pad_w)


def _project_pts_to_flow(
    pts3d_src: np.ndarray,
    K_dst: np.ndarray,
    T_world_src: np.ndarray,
    T_world_dst: np.ndarray,
    validity_src: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """
    pts3d_src (cam_src 座標系) を cam_dst に投影し、光学フロー (2, H, W) を返す。

    flow[0] = u_dst - u_src (水平方向)
    flow[1] = v_dst - v_src (垂直方向)

    無効画素 (validity=False, z<=0, 画像外) はフロー=0。
    """
    # T_dst_src = inv(T_world_dst) @ T_world_src
    T_dst_src = np.linalg.inv(T_world_dst) @ T_world_src  # (4, 4)
    R = T_dst_src[:3, :3].astype(np.float32)
    t = T_dst_src[:3, 3].astype(np.float32)

    # pts3d_src を dst カメラ座標系に変換
    pts = pts3d_src.reshape(-1, 3).astype(np.float32)
    pts_dst = (R @ pts.T).T + t  # (N, 3)

    # ピンホール投影
    fx, fy = float(K_dst[0, 0]), float(K_dst[1, 1])
    cx, cy = float(K_dst[0, 2]), float(K_dst[1, 2])
    z = pts_dst[:, 2].reshape(H, W)
    u_dst = (fx * pts_dst[:, 0] / (z.ravel() + 1e-8) + cx).reshape(H, W)
    v_dst = (fy * pts_dst[:, 1] / (z.ravel() + 1e-8) + cy).reshape(H, W)

    # ソース画素座標グリッド
    uu = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    vv = np.tile(np.arange(H, dtype=np.float32).reshape(H, 1), (1, W))

    flow_u = (u_dst - uu).astype(np.float32)
    flow_v = (v_dst - vv).astype(np.float32)

    # 無効領域をゼロに
    invalid = (~validity_src) | (z <= 0) | (u_dst < 0) | (u_dst >= W) | (v_dst < 0) | (v_dst >= H)
    flow_u[invalid] = 0.0
    flow_v[invalid] = 0.0

    return np.stack([flow_u, flow_v], axis=0)  # (2, H, W)


def compute_geometric_flows(
    pts3d_list: list[np.ndarray],
    K_list: list[np.ndarray],
    poses_list: list[np.ndarray],
    validity_list: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Any4D の pts3d・カメラ姿勢から幾何学的に正確な光学フローを計算する。

    RAFT のような画像ベース推定と異なり、3D 幾何から直接導出するため
    カメラ自己運動が大きい場合でも誤差が生じない。

    Args:
        pts3d_list:   N × (H, W, 3) float  各フレームの点群 (cam_i 座標系)
        K_list:       N × (3, 3)            カメラ内部行列
        poses_list:   N × (4, 4)            T_world_cam (cam→world)
        validity_list: N × (H, W) bool      有効画素マスク

    Returns:
        gt_flows_f: (1, N-1, 2, H, W) float32  前向きフロー (frame i → i+1)
        gt_flows_b: (1, N-1, 2, H, W) float32  後ろ向きフロー (frame i+1 → i)
    """
    N = len(pts3d_list)
    H, W = pts3d_list[0].shape[:2]
    flows_f, flows_b = [], []

    for i in range(N - 1):
        flows_f.append(_project_pts_to_flow(
            pts3d_list[i], K_list[i + 1],
            poses_list[i], poses_list[i + 1],
            validity_list[i], H, W,
        ))
        flows_b.append(_project_pts_to_flow(
            pts3d_list[i + 1], K_list[i],
            poses_list[i + 1], poses_list[i],
            validity_list[i + 1], H, W,
        ))

    gt_flows_f = torch.from_numpy(np.stack(flows_f)).unsqueeze(0)  # (1, N-1, 2, H, W)
    gt_flows_b = torch.from_numpy(np.stack(flows_b)).unsqueeze(0)
    return gt_flows_f, gt_flows_b


def run_propainter_inpaint(
    frames_rgb: list[np.ndarray],
    masks: list[tuple[Image.Image, Image.Image]],
    device: torch.device,
    propainter_weights_dir: str = "checkpoints/propainter",
    subvideo_length: int = 80,
    neighbor_length: int = 10,
    ref_stride: int = 10,
    fp16: bool = True,
    raft_iter: int = 20,
    geo_flows_f: torch.Tensor | None = None,
    geo_flows_b: torch.Tensor | None = None,
) -> list[np.ndarray]:
    """
    ProPainter Python API でインペインティングを実行する。

    Args:
        frames_rgb: 元画像フレームのリスト [(H, W, 3) uint8 RGB]
        masks: フレームごとの (flow_mask_pil, mask_dilated_pil) タプルのリスト
        device: 推論デバイス
        propainter_weights_dir: ProPainter 重みファイルのディレクトリ
        subvideo_length: サブビデオ分割長 (VRAM 不足時は 40 に下げる)
        neighbor_length: 近傍フレーム数 (デフォルト 10)
        ref_stride: 参照フレームストライド (デフォルト 10)
        fp16: fp16 推論を使用するか (RTX 4090 では True 推奨)
        raft_iter: RAFT の反復回数 (デフォルト 20)

    Returns:
        インペイント済みフレームのリスト [(H, W, 3) uint8 RGB]
    """
    _ensure_propainter()
    from model.modules.flow_comp_raft import RAFT_bi
    from model.propainter import InpaintGenerator
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    from core.utils import to_tensors

    pretrain_model_url = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
    os.makedirs(propainter_weights_dir, exist_ok=True)

    def _load_ckpt(filename: str) -> str:
        local_path = os.path.join(propainter_weights_dir, filename)
        if os.path.exists(local_path):
            return local_path
        from utils.download_util import load_file_from_url
        return load_file_from_url(
            url=os.path.join(pretrain_model_url, filename),
            model_dir=propainter_weights_dir,
            progress=True,
            file_name=filename,
        )

    print("[inpaint] ProPainter モデルをロード中...")

    # RAFT_bi ロード
    raft_ckpt = _load_ckpt("raft-things.pth")
    fix_raft = RAFT_bi(raft_ckpt, device)

    # RecurrentFlowCompleteNet ロード
    flow_ckpt = _load_ckpt("recurrent_flow_completion.pth")
    fix_flow_complete = RecurrentFlowCompleteNet(flow_ckpt)
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    # InpaintGenerator ロード
    inpaint_ckpt = _load_ckpt("ProPainter.pth")
    model = InpaintGenerator(model_path=inpaint_ckpt).to(device)
    model.eval()

    if fp16:
        # RAFT は corr.py 内で強制的に .float() するため fp16 化不可。float32 のまま使用。
        fix_flow_complete = fix_flow_complete.half()
        model = model.half()

    video_length = len(frames_rgb)
    if video_length < 2:
        print("[inpaint] ⚠ フレーム数が 1 のため ProPainter をスキップします（光学フロー計算には最低 2 フレーム必要）")
        return [f.copy() for f in frames_rgb]

    h_orig, w_orig = frames_rgb[0].shape[:2]

    # 8 の倍数にパディング
    pad_h = (8 - h_orig % 8) % 8
    pad_w = (8 - w_orig % 8) % 8
    h = h_orig + pad_h
    w = w_orig + pad_w

    print(f"[inpaint] フレーム数={video_length}, 解像度={w_orig}x{h_orig} → パディング後 {w}x{h}")

    # フレームを PIL → numpy でパディング
    frames_padded = []
    for f in frames_rgb:
        if pad_h > 0 or pad_w > 0:
            f = np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        frames_padded.append(Image.fromarray(f))

    # マスクをパディング
    flow_masks_padded = []
    masks_dilated_padded = []
    for fm, md in masks:
        fm_np = np.array(fm)
        md_np = np.array(md)
        if pad_h > 0 or pad_w > 0:
            fm_np = np.pad(fm_np, ((0, pad_h), (0, pad_w)), mode="edge")
            md_np = np.pad(md_np, ((0, pad_h), (0, pad_w)), mode="edge")
        flow_masks_padded.append(Image.fromarray(fm_np, mode="L"))
        masks_dilated_padded.append(Image.fromarray(md_np, mode="L"))

    # テンソル変換
    # to_tensors() は PIL Image のリスト全体を受け取り (T, C, H, W) を返す
    transform = to_tensors()
    frames_tensor = transform(frames_padded).unsqueeze(0)          # (1, T, 3, H, W) [0, 1]
    frames_tensor = frames_tensor * 2 - 1                          # [-1, 1] 正規化
    flow_masks_tensor = transform(flow_masks_padded).unsqueeze(0)  # (1, T, 1, H, W)
    masks_dilated_tensor = transform(masks_dilated_padded).unsqueeze(0)

    frames_tensor = frames_tensor.to(device)
    flow_masks_tensor = flow_masks_tensor.to(device)
    masks_dilated_tensor = masks_dilated_tensor.to(device)

    if fp16:
        frames_tensor = frames_tensor.half()
        flow_masks_tensor = flow_masks_tensor.half()
        masks_dilated_tensor = masks_dilated_tensor.half()

    # ---- Step 1: 光学フロー計算 ----
    if geo_flows_f is not None and geo_flows_b is not None:
        # 幾何フローを使用（RAFT スキップ）
        # geo_flows は元解像度 (h_orig, w_orig) で計算済み → パディング後サイズに合わせる
        print("[inpaint] 幾何フローを使用（RAFT スキップ）...")
        import torch.nn.functional as F
        gt_flows_f = geo_flows_f.to(device)
        gt_flows_b = geo_flows_b.to(device)
        if pad_h > 0 or pad_w > 0:
            gt_flows_f = F.pad(gt_flows_f, (0, pad_w, 0, pad_h))
            gt_flows_b = F.pad(gt_flows_b, (0, pad_w, 0, pad_h))
    else:
        print("[inpaint] RAFT 光学フロー計算中...")
        short_clip_len = max(subvideo_length // 2, 12)
        if w <= 640:
            short_clip_len = 80
        elif w <= 1280:
            short_clip_len = 40
        else:
            short_clip_len = 20

        gt_flows_f_list, gt_flows_b_list = [], []
        with torch.no_grad():
            if frames_tensor.size(1) > short_clip_len:
                for f_start in range(0, video_length, short_clip_len):
                    f_end = min(video_length, f_start + short_clip_len)
                    if f_start == 0:
                        flows_f, flows_b = fix_raft(frames_tensor[:, f_start:f_end].float(), iters=raft_iter)
                    else:
                        flows_f, flows_b = fix_raft(frames_tensor[:, f_start - 1:f_end].float(), iters=raft_iter)
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            else:
                gt_flows_f, gt_flows_b = fix_raft(frames_tensor.float(), iters=raft_iter)

    # RAFT / 幾何フローともに float32 → fp16 の場合はキャスト
    if fp16:
        gt_flows_f = gt_flows_f.half()
        gt_flows_b = gt_flows_b.half()

    # ---- Step 2: マスク領域のフロー補完 ----
    print("[inpaint] フロー補完中...")
    flow_length = video_length - 1
    masked_flows_f = gt_flows_f * (1 - flow_masks_tensor[:, :-1])
    masked_flows_b = gt_flows_b * (1 - flow_masks_tensor[:, 1:])

    pred_flows_f_list, pred_flows_b_list = [], []
    pad_len = 2

    with torch.no_grad():
        if flow_length > subvideo_length:
            for f_start in range(0, flow_length, subvideo_length):
                s_f = max(0, f_start - pad_len)
                e_f = min(flow_length, f_start + subvideo_length + pad_len)
                pad_len_s = max(0, f_start) - s_f
                pad_len_e = e_f - min(flow_length, f_start + subvideo_length)

                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (masked_flows_f[:, s_f:e_f], masked_flows_b[:, s_f:e_f]),
                    flow_masks_tensor[:, s_f:e_f + 1],
                )
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (masked_flows_f[:, s_f:e_f], masked_flows_b[:, s_f:e_f]),
                    pred_flows_bi_sub,
                    flow_masks_tensor[:, s_f:e_f + 1],
                )
                pred_flows_f_list.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                pred_flows_b_list.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
            pred_flows_f = torch.cat(pred_flows_f_list, dim=1)
            pred_flows_b = torch.cat(pred_flows_b_list, dim=1)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(
                (masked_flows_f, masked_flows_b),
                flow_masks_tensor,
            )
            pred_flows_bi = fix_flow_complete.combine_flow(
                (masked_flows_f, masked_flows_b),
                pred_flows_bi,
                flow_masks_tensor,
            )
            pred_flows_f, pred_flows_b = pred_flows_bi

    pred_flows_bi = (pred_flows_f, pred_flows_b)

    # ---- Step 3: 画像伝播 ----
    print("[inpaint] 画像伝播中...")
    masked_frames = frames_tensor * (1 - masks_dilated_tensor)
    prop_imgs_list, updated_masks_list = [], []

    with torch.no_grad():
        if video_length > subvideo_length:
            for f_start in range(0, video_length, subvideo_length):
                s_f = max(0, f_start - pad_len)
                e_f = min(video_length, f_start + subvideo_length + pad_len)
                pad_len_s = max(0, f_start) - s_f
                pad_len_e = e_f - min(video_length, f_start + subvideo_length)

                prop_sub, updated_sub = model.img_propagation(
                    masked_frames[:, s_f:e_f],
                    (pred_flows_f[:, s_f:max(0, e_f - 1)], pred_flows_b[:, s_f:max(0, e_f - 1)]),
                    masks_dilated_tensor[:, s_f:e_f],
                    "nearest",
                )
                prop_imgs_list.append(prop_sub[:, pad_len_s:e_f - s_f - pad_len_e].cpu())
                updated_masks_list.append(updated_sub[:, pad_len_s:e_f - s_f - pad_len_e].cpu())
            prop_imgs = torch.cat(prop_imgs_list, dim=1).to(device)
            updated_masks = torch.cat(updated_masks_list, dim=1).to(device)
        else:
            prop_imgs, updated_masks = model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated_tensor, "nearest"
            )

    # ---- Step 4: トランスフォーマーによるインペインティング ----
    print("[inpaint] トランスフォーマーインペインティング中...")
    updated_frames = frames_tensor * (1 - masks_dilated_tensor) + prop_imgs * masks_dilated_tensor
    updated_masks_frames = masks_dilated_tensor * (1 - updated_masks)

    comp_frames = [None] * video_length
    neighbor_stride = neighbor_length // 2

    def get_ref_index(mid_id, neighbor_ids, length, stride=10, ref_num=-1):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_id - stride * (ref_num // 2))
            end_idx = min(length, mid_id + stride * (ref_num // 2))
            for i in range(start_idx, end_idx, stride):
                if i not in neighbor_ids:
                    if len(ref_index) >= ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    with torch.no_grad():
        for f_idx in range(0, video_length, neighbor_stride):
            neighbor_ids = list(range(
                max(0, f_idx - neighbor_length // 2),
                min(video_length, f_idx + neighbor_length // 2 + 1),
            ))
            ref_ids = get_ref_index(f_idx, neighbor_ids, video_length, stride=ref_stride)
            l_t = len(neighbor_ids)

            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, ...]
            selected_masks = masks_dilated_tensor[:, neighbor_ids + ref_ids, ...]
            selected_update_masks = updated_masks_frames[:, neighbor_ids + ref_ids, ...]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], ...],
                pred_flows_bi[1][:, neighbor_ids[:-1], ...],
            )

            with torch.cuda.amp.autocast(enabled=fp16):
                pred_img = model(
                    selected_imgs,
                    selected_pred_flows_bi,
                    selected_masks,
                    selected_update_masks,
                    l_t,
                )

            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2  # [0, 1]
            pred_img = pred_img.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, 3)

            binary_masks = masks_dilated_tensor[0, neighbor_ids, ...].permute(0, 2, 3, 1).cpu().numpy()

            for i, idx in enumerate(neighbor_ids):
                img = pred_img[i] * binary_masks[i] + (
                    (frames_tensor[0, idx, ...] + 1) / 2
                ).permute(1, 2, 0).cpu().numpy() * (1 - binary_masks[i])
                img = np.clip(img * 255, 0, 255).astype(np.uint8)

                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (
                        comp_frames[idx].astype(np.float32) * 0.5
                        + img.astype(np.float32) * 0.5
                    ).astype(np.uint8)

    # パディングを除去して元サイズに戻す
    result = []
    for frame in comp_frames:
        if frame is None:
            result.append(np.zeros((h_orig, w_orig, 3), dtype=np.uint8))
        else:
            result.append(frame[:h_orig, :w_orig])

    return result


def save_sliding_window_inpainted_mp4(
    all_views_world: list[dict],
    model,
    device,
    output_path: str,
    window_size: int,
    window_stride: int,
    fps: float,
    max_depth: float,
    sf_percentile: float,
    sf_abs_threshold: float | None = None,
    sf_close_px: int = 0,
    sf_dilate_px: int = 0,
    side_by_side: bool = True,
    dilate_px: int = 8,
    point_radius: int = 2,
    subvideo_length: int = 80,
    propainter_weights_dir: str = "checkpoints/propainter",
    fp16: bool = True,
) -> None:
    """
    スライディングウィンドウで Any4D 推論 → ProPainter インペインティング → MP4 保存。

    フェーズ:
        1. スライディングウィンドウ推論 + validity_mask 収集
        2. Any4D モデルを CPU に退避して VRAM 解放
        3. ProPainter でインペインティング
        4. side_by_side (元画像 | 点群レンダリング | インペイント) で MP4 保存

    Args:
        all_views_world: 絶対ワールド姿勢の camera_poses を持つビューリスト
        model: _init_any4d_model で初期化済みのモデル
        device: 推論デバイス
        output_path: 出力 MP4 パス
        window_size: 1 ウィンドウのフレーム数
        window_stride: ウィンドウのスライド幅
        fps: 出力 MP4 のフレームレート
        max_depth: レンダリング対象の最大深度 [m]
        sf_percentile: 動的物体判定のパーセンタイル閾値
        side_by_side: True で (元画像 | 点群 | インペイント) 3列出力
        dilate_px: インペインティングマスクの膨張量 (px)
        subvideo_length: ProPainter サブビデオ分割長 (VRAM 不足時は 40)
        propainter_weights_dir: ProPainter 重みファイルのディレクトリ
        fp16: ProPainter を fp16 で実行するか
    """
    # physai_av_to_any4d.py のヘルパーを import
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from physai_av_to_any4d import (
        _infer_with_model,
        _make_relative_views,
        compute_static_mask,
        render_pointcloud_to_image,
    )
    from any4d.utils.geometry import recover_pinhole_intrinsics_from_ray_directions
    from any4d.utils.image import rgb as to_rgb

    total = len(all_views_world)
    window_starts = list(range(0, total - window_size + 1, window_stride))
    if not window_starts:
        print(
            f"[inpaint] ⚠ ウィンドウが作れません "
            f"(total={total} < window_size={window_size})。"
            f"--num_frames を増やすか --window_size を小さくしてください。"
        )
        return

    print(
        f"[inpaint] Phase 1: {len(window_starts)} windows で Any4D 推論開始 "
        f"(size={window_size}, stride={window_stride})"
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Phase 1: スライディングウィンドウ推論 + validity_mask 収集
    collected_frames = []    # 元画像 RGB (H, W, 3) uint8
    collected_rendered = []  # 点群レンダリング RGB (H, W, 3) uint8
    collected_validity = []  # validity_mask (H, W) bool
    collected_pts3d = []     # 点群 (H, W, 3) in cam_i 座標系
    collected_K = []         # カメラ内部行列 (3, 3)
    collected_poses = []     # T_world_cam (4, 4) 絶対姿勢

    for wi, start in enumerate(window_starts):
        window_views_world = all_views_world[start:start + window_size]
        window_views_rel = _make_relative_views(window_views_world)

        result, preprocessed = _infer_with_model(model, device, window_views_rel, verbose=False)

        H = preprocessed[0]["img"].shape[-2]
        W = preprocessed[0]["img"].shape[-1]
        static_mask = compute_static_mask(
            result, len(window_views_rel), H, W,
            sf_percentile=sf_percentile,
            sf_abs_threshold=sf_abs_threshold,
            sf_close_px=sf_close_px,
            sf_dilate_px=sf_dilate_px,
        )

        ray_dirs = result["pred1"]["ray_directions"][0].cpu()
        K_infer = recover_pinhole_intrinsics_from_ray_directions(ray_dirs).numpy()
        pts3d = result["pred1"]["pts3d"][0].cpu().numpy()  # (H, W, 3) in cam0
        colors_rgb = (to_rgb(preprocessed[0]["img"], norm_type="dinov2")[0] * 255).astype(
            np.uint8
        )

        # 幾何フロー計算用に絶対姿勢を収集（window_views_world[0] が当該フレームの cam0）
        T_world_cam = window_views_world[0]["camera_poses"][0].cpu().numpy()  # (4, 4)

        depth_mask = (pts3d[..., 2] > 0.01) & (pts3d[..., 2] < max_depth)
        combined_mask = static_mask.numpy() & depth_mask

        rendered, validity = render_pointcloud_to_image(
            pts3d,
            colors_rgb,
            K_infer,
            np.eye(4, dtype=np.float64),
            H,
            W,
            mask=combined_mask,
            point_radius=point_radius,
            return_validity_mask=True,
        )

        collected_frames.append(colors_rgb.copy())
        collected_rendered.append(rendered.copy())
        collected_validity.append(validity.copy())
        collected_pts3d.append(pts3d.copy())
        collected_K.append(K_infer.copy())
        collected_poses.append(T_world_cam.copy())

        print(
            f"  [inpaint] window {wi + 1}/{len(window_starts)}: "
            f"frames[{start}~{start + window_size - 1}] 完了 "
            f"(有効画素: {validity.sum()}/{H * W} = {100 * validity.mean():.1f}%)"
        )

    # Phase 2: Any4D モデルを解放して VRAM を確保
    print("[inpaint] Phase 2: Any4D モデルを CPU に退避して VRAM を解放...")
    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # Phase 2.5: 幾何フロー計算（RAFT の代替）
    geo_flows_f = None
    geo_flows_b = None
    if len(collected_pts3d) >= 2:
        print("[inpaint] Phase 2.5: 幾何フロー計算中...")
        try:
            geo_flows_f, geo_flows_b = compute_geometric_flows(
                collected_pts3d,
                collected_K,
                collected_poses,
                collected_validity,
            )
            print(
                f"  幾何フロー計算完了: shape={geo_flows_f.shape} "
                f"(フレーム間フロー {geo_flows_f.shape[1]} ペア)"
            )
        except Exception as e:
            print(f"  [警告] 幾何フロー計算に失敗しました ({e})。RAFT にフォールバックします。")
            geo_flows_f = None
            geo_flows_b = None
    else:
        print("[inpaint] Phase 2.5: フレーム数 < 2 のため幾何フロー計算をスキップ")

    # Phase 3: インペインティングマスク生成 + ProPainter 実行
    print("[inpaint] Phase 3: インペインティングマスク生成中...")
    inpaint_masks = [generate_inpaint_masks(v, dilate_px=dilate_px) for v in collected_validity]

    n_holes = sum(
        (np.array(md) > 0).sum()
        for _, md in inpaint_masks
    )
    print(f"  マスク生成完了: 合計穴画素数 = {n_holes:,}")

    print("[inpaint] Phase 3: ProPainter インペインティング開始...")
    # ProPainter に渡す合成画像を作成する。
    # - inpaint マスク外（validity=True かつ補完対象外）→ 点群レンダリング色
    # - inpaint マスク内（dilated mask 領域）         → 元カメラ色
    #
    # 単純に ~validity で埋めると点群端の暗い画素（点密度低下領域）が
    # マスク膨張部分に残り、ProPainter がその黒・藍色を周囲に滲ませる。
    # dilated mask 全体を元カメラ色で埋めることで ProPainter が
    # 明るく自然な初期値から補完できる。
    collected_composite = []
    for rendered, original, (_, mask_dilated_pil) in zip(
        collected_rendered, collected_frames, inpaint_masks
    ):
        mask_np = np.array(mask_dilated_pil) > 0  # (H, W) bool
        composite = rendered.copy()
        composite[mask_np] = original[mask_np]
        collected_composite.append(composite)

    inpainted_frames = run_propainter_inpaint(
        frames_rgb=collected_composite,
        masks=inpaint_masks,
        device=device,
        propainter_weights_dir=propainter_weights_dir,
        subvideo_length=subvideo_length,
        fp16=fp16,
        geo_flows_f=geo_flows_f,
        geo_flows_b=geo_flows_b,
    )

    # Phase 4: MP4 保存
    print(f"[inpaint] Phase 4: MP4 保存中... → {output_path}")
    writer = None

    for i, (orig_rgb, rendered_rgb, inpainted_rgb) in enumerate(
        zip(collected_frames, collected_rendered, inpainted_frames)
    ):
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)

        if side_by_side:
            # 3列: 元画像 | 点群レンダリング | インペイント
            H, W = orig_bgr.shape[:2]
            if rendered_bgr.shape[:2] != (H, W):
                rendered_bgr = cv2.resize(rendered_bgr, (W, H))
            if inpainted_bgr.shape[:2] != (H, W):
                inpainted_bgr = cv2.resize(inpainted_bgr, (W, H))
            frame_bgr = np.concatenate([orig_bgr, rendered_bgr, inpainted_bgr], axis=1)
        else:
            frame_bgr = inpainted_bgr

        if writer is None:
            fh, fw = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

        writer.write(frame_bgr)

    if writer:
        writer.release()

    print(f"[inpaint] 保存完了: {output_path}")
    print(f"  フレーム数: {len(collected_frames)}")
    if side_by_side:
        H, W = collected_frames[0].shape[:2]
        print(f"  解像度: {3 * W}x{H} (3列: 元画像 | 点群 | インペイント)")
    else:
        H, W = inpainted_frames[0].shape[:2]
        print(f"  解像度: {W}x{H}")
