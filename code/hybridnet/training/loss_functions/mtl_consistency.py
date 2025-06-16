import torch
import torch.nn as nn
import torch.nn.functional as F


softmax_helper = lambda x: F.softmax(x, 1)
class MTLConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, seg_output, cls_output):
        """
        :param seg_output: segmentation output tensor (B, C=2, D, H, W)
        :param cls_output: classification output (B, C=2)
        """
        # 배치 사이즈 검증
        seg_batch_size = seg_output.shape[0]
        cls_batch_size = cls_output.shape[0]
        
        if seg_batch_size != cls_batch_size:
            raise ValueError(f"Batch size mismatch: seg_output batch size ({seg_batch_size}) != cls_output batch size ({cls_batch_size})")
        
        # --- Consistency Loss ---
        output_softmax = softmax_helper(seg_output)      # (B, 2, D, H, W)
        output_seg = output_softmax.argmax(dim=1)           # (B, D, H, W)
        seg_pred_mask = (output_seg == 1)
        seg_probs = output_softmax[:, 1]                    # (B, D, H, W)

        seg_pred_scores = []
        for b in range(seg_batch_size):
            pred_voxels = seg_probs[b][seg_pred_mask[b]]
            if pred_voxels.numel() > 0:
                mean_val = pred_voxels.mean()
                seg_pred_scores.append(mean_val)
            else:
                seg_pred_scores.append(torch.tensor(0.0, device=seg_output.device))

        seg_pred_scores = torch.stack(seg_pred_scores)  # (B,)
        cls_probs = F.softmax(cls_output, dim=1)[:, 1]   # (B,)
        
        # Shape 확인을 위한 디버그 출력
        # print(f"seg_pred_scores shape: {seg_pred_scores.shape}, cls_probs shape: {cls_probs.shape}")
        # print(f"seg_probs shape: {seg_probs.shape}, cls_output shape: {cls_output.shape}")
        
        consistency_loss = F.mse_loss(seg_pred_scores, cls_probs)
        #print(f"[Consistency Loss] MSE: {consistency_loss.item():.6f}")
        return consistency_loss
