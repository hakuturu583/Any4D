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

    # ---- Step 1: RAFT 光学フロー計算 ----
    print("[inpaint] RAFT 光学フロー計算中...")
    short_clip_len = max(subvideo_length // 2, 12)
    # フレーム幅に応じて調整
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

    # RAFT は常に float32 を出力する。fix_flow_complete が fp16 の場合は合わせてキャスト。
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
    collected_frames = []  # 元画像 RGB (H, W, 3) uint8
    collected_rendered = []  # 点群レンダリング RGB (H, W, 3) uint8
    collected_validity = []  # validity_mask (H, W) bool

    for wi, start in enumerate(window_starts):
        window_views_world = all_views_world[start:start + window_size]
        window_views_rel = _make_relative_views(window_views_world)

        result, preprocessed = _infer_with_model(model, device, window_views_rel, verbose=False)

        H = preprocessed[0]["img"].shape[-2]
        W = preprocessed[0]["img"].shape[-1]
        static_mask = compute_static_mask(
            result, len(window_views_rel), H, W, sf_percentile=sf_percentile
        )

        ray_dirs = result["pred1"]["ray_directions"][0].cpu()
        K_infer = recover_pinhole_intrinsics_from_ray_directions(ray_dirs).numpy()
        pts3d = result["pred1"]["pts3d"][0].cpu().numpy()  # (H, W, 3) in cam0
        colors_rgb = (to_rgb(preprocessed[0]["img"], norm_type="dinov2")[0] * 255).astype(
            np.uint8
        )

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

    # Phase 3: インペインティングマスク生成 + ProPainter 実行
    print("[inpaint] Phase 3: インペインティングマスク生成中...")
    inpaint_masks = [generate_inpaint_masks(v, dilate_px=dilate_px) for v in collected_validity]

    n_holes = sum(
        (np.array(md) > 0).sum()
        for _, md in inpaint_masks
    )
    print(f"  マスク生成完了: 合計穴画素数 = {n_holes:,}")

    print("[inpaint] Phase 3: ProPainter インペインティング開始...")
    inpainted_frames = run_propainter_inpaint(
        frames_rgb=collected_frames,
        masks=inpaint_masks,
        device=device,
        propainter_weights_dir=propainter_weights_dir,
        subvideo_length=subvideo_length,
        fp16=fp16,
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
