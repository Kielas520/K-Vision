import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNSiLU(nn.Module):
    """标准的卷积块，支持调整 kernel_size 和 padding"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class DepthwiseConvBlock(nn.Module):
    """深度可分离卷积块 (Depthwise Separable Convolution) - 已增加残差连接支持"""
    def __init__(self, in_channels, out_channels, stride=1, use_res=False):
        super().__init__()
        # 只有在步长为1且输入输出通道相同时，才开启残差连接
        self.use_res = use_res and (stride == 1) and (in_channels == out_channels)
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.silu1 = nn.SiLU(inplace=True)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu2 = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.silu1(self.bn1(self.depthwise(x)))
        out = self.silu2(self.bn2(self.pointwise(out)))
        if self.use_res:
            return x + out
        return out

class StackedBlocks(nn.Module):
    """连续堆叠多个 Block，增加网络深度和特征提取容量"""
    def __init__(self, in_channels, out_channels, num_blocks, stride=1):
        super().__init__()
        layers = []
        # 第一层负责跨通道和下采样
        layers.append(DepthwiseConvBlock(in_channels, out_channels, stride=stride, use_res=False))
        # 后续层保持通道数和分辨率，并开启残差连接
        for _ in range(num_blocks - 1):
            layers.append(DepthwiseConvBlock(out_channels, out_channels, stride=1, use_res=True))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)

class SPPF(nn.Module):
    """空间金字塔池化 (快速版)，显著增加感受野"""
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        c_ = in_channels // 2  
        self.cv1 = ConvBNSiLU(in_channels, c_, kernel_size=1, padding=0)
        self.cv2 = ConvBNSiLU(c_ * 4, out_channels, kernel_size=1, padding=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class RMHead(nn.Module):
    """
    精简后的 Head：取消独立 Box 分支，合并分类与置信度
    """
    def __init__(self, in_channels=256, reg_max=16, num_classes=12):
        super().__init__()
        self.reg_max = reg_max
        self.num_classes = num_classes
        
        # 全卷积输出头 用于将语义转化成输出特征
        # 解耦头：分类和回归使用独立的卷积分支 
        self.cls_convs = nn.Sequential(
            ConvBNSiLU(in_channels, in_channels, kernel_size=3, padding=1),
            ConvBNSiLU(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.reg_convs = nn.Sequential(
            ConvBNSiLU(in_channels, in_channels, kernel_size=3, padding=1),
            ConvBNSiLU(in_channels, in_channels, kernel_size=3, padding=1)
        )
        
        # 分类分支同时承担正负样本判别任务
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        # 回归分支仅输出 8 个关键点的 DFL 分布
        self.pose_pred = nn.Conv2d(in_channels, 8 * reg_max, kernel_size=1)
        
        self._initialize_biases()

    def _initialize_biases(self):
        # 偏置初始化：让初始预测概率接近 0.01，防止训练初期负样本 Loss 爆炸
        prior_prob = 0.01
        bias_val = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        # 初始状态下所有类别概率均接近 prior_prob
        nn.init.constant_(self.cls_pred.bias, bias_val) # type: ignore

    def forward(self, x):
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        # 输出顺序：[batch, num_classes + 8*reg_max, h, w]
        return torch.cat([self.cls_pred(cls_feat), self.pose_pred(reg_feat)], dim=1)

class RMNeck(nn.Module):
    """标准的 FPN + PAN 结构，输出 P3, P4, P5 三个尺度"""
    def __init__(self, in_channels_list=[64, 128, 256], out_channels=256):
        super().__init__()
        c3, c4, c5 = in_channels_list
        
        # Top-down
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_f5 = ConvBNSiLU(c5, out_channels, 1, padding=0)
        self.conv_f4 = ConvBNSiLU(c4 + out_channels, out_channels, 1, padding=0)
        self.conv_f3 = ConvBNSiLU(c3 + out_channels, out_channels, 1, padding=0)
        
        # Bottom-up
        self.down_p3 = ConvBNSiLU(out_channels, out_channels, 3, stride=2, padding=1)
        self.conv_p4 = ConvBNSiLU(out_channels * 2, out_channels, 1, padding=0)
        self.down_p4 = ConvBNSiLU(out_channels, out_channels, 3, stride=2, padding=1)
        self.conv_p5 = ConvBNSiLU(out_channels + c5, out_channels, 1, padding=0)

    def forward(self, s3, s4, s5):
        # Top-down path
        f5 = self.conv_f5(s5)
        f4 = self.conv_f4(torch.cat([s4, self.up(f5)], 1))
        f3 = self.conv_f3(torch.cat([s3, self.up(f4)], 1))
        
        # Bottom-up path
        p3 = f3
        p4 = self.conv_p4(torch.cat([f4, self.down_p3(p3)], 1))
        p5 = self.conv_p5(torch.cat([f5, self.down_p4(p4)], 1))
        return p3, p4, p5 

class RMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBNSiLU(3, 16, stride=2)
        self.stage2 = StackedBlocks(16, 32, num_blocks=2, stride=2)   # Stride 4
        self.stage3 = StackedBlocks(32, 64, num_blocks=3, stride=2)   # Stage 3: 步长 8 (416 -> 52x52, 64通道)
        self.stage4 = StackedBlocks(64, 128, num_blocks=3, stride=2)  # Stage 4: 步长 16 (416 -> 26x26, 128通道)
        self.stage5 = StackedBlocks(128, 256, num_blocks=3, stride=2) # Stage 5: 步长 32 (416 -> 13x13, 256通道)
        
        self.sppf = SPPF(256, 256, k=5)

    def forward(self, x):
        x = self.stage1(x)
        feat_s2 = self.stage2(x)
        feat_s3 = self.stage3(feat_s2)
        feat_s4 = self.stage4(feat_s3)
        feat_s5 = self.stage5(feat_s4)
        feat_s5 = self.sppf(feat_s5)
        
        # 仅抛出 S3, S4, S5 参与 FPN 融合
        return feat_s3, feat_s4, feat_s5

class RMDetector(nn.Module):
    def __init__(self, reg_max=16, num_classes=12):
        super().__init__()
        self.backbone = RMBackbone()
        self.neck = RMNeck()
        self.head = RMHead(reg_max=reg_max, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        ps = self.neck(*feats)
        # 对三个尺度分别预测，返回列表
        return [self.head(p) for p in ps]

def keypoint_nms(pts, scores, dist_thresh=15.0):
    """
    基于关键点最小欧氏距离的 NMS
    pts: [N, 4, 2]
    scores: [N] (综合分类得分)
    dist_thresh: 判定为共享灯条/角点的最小像素距离阈值
    """
    if pts.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=pts.device)

    pts = pts.view(-1, 4, 2)
    # 按得分降序排列
    sorted_indices = torch.argsort(scores, descending=True)
    pts = pts[sorted_indices]
    
    keep = []
    suppressed = torch.zeros(len(pts), dtype=torch.bool, device=pts.device)

    for i in range(len(pts)):
        if suppressed[i]:
            continue
        
        keep.append(sorted_indices[i])
        
        if i == len(pts) - 1:
            break
            
        remaining_indices = torch.arange(i + 1, len(pts), device=pts.device)
        valid_mask = ~suppressed[remaining_indices]
        valid_indices = remaining_indices[valid_mask]
        
        if len(valid_indices) == 0:
            continue

        pts_i = pts[i]               
        pts_j = pts[valid_indices]   
        
        # 计算两两关键点距离 [M, 4, 4]
        diff = pts_i.unsqueeze(0).unsqueeze(2) - pts_j.unsqueeze(1) 
        dists = torch.norm(diff, dim=-1) 
        
        # 提取每对装甲板之间的最小关键点距离
        min_dists = dists.view(len(valid_indices), -1).min(dim=1)[0] 
        
        # 如果最小点距小于阈值，说明共享了灯条，触发抑制
        suppressed[valid_indices[min_dists < dist_thresh]] = True

    return torch.tensor(keep, dtype=torch.int64, device=pts.device)

def decode_tensor(tensor, is_pred=True, class_tensor=None, conf_threshold=0.5, kpt_dist_thresh=15.0, grid_size=(52, 52), reg_max=16, img_size=(416, 416), num_classes=12, negative_class_id=12):
    batch_size = tensor.shape[0]
    grid_w, grid_h = grid_size
    img_w, img_h = img_size
    
    if is_pred:
        # 预测模式提取分类与置信度
        cls_logits = tensor[:, :num_classes, :, :]
        cls_scores = torch.sigmoid(cls_logits)
        conf, classes = torch.max(cls_scores, dim=1)
    else:
        # GT 模式：直接读取首位置信度
        conf = tensor[:, 0, :, :]
        if class_tensor is not None:
            classes = class_tensor[:, 0, :, :].float()
        else:
            classes = torch.zeros_like(conf)
        
    batch_results = []
    
    for b in range(batch_size):
        if is_pred:
            # === 【核心修改】===
            # 不仅置信度要达标，并且类别绝对不能是负样本（12）！
            mask = (conf[b] >= conf_threshold) & (classes[b] != negative_class_id)
        else:
            # GT 模式不变
            mask = conf[b] >= conf_threshold
            
        if not mask.any():
            batch_results.append([])
            continue
            
        grid_y, grid_x = torch.nonzero(mask, as_tuple=True)
        scores = conf[b, grid_y, grid_x]
        item_classes = classes[b, grid_y, grid_x].float()
        
        if is_pred:
            # === 【修改开始】 ===
            expected_raw_channels = num_classes + 8 * reg_max
            expected_fused_channels = num_classes + 8
            
            if tensor.shape[1] == expected_raw_channels:
                # 【情况A：Python 原生推理】模型还没融合 DFL，自己做 Softmax
                pose_start = num_classes
                pose_end = num_classes + 8 * reg_max
                raw_pose_dist = tensor[b, pose_start:pose_end, grid_y, grid_x].T 
                raw_pose_dist = raw_pose_dist.view(-1, 8, reg_max)
                
                prob = F.softmax(raw_pose_dist, dim=-1)
                project = torch.arange(reg_max, dtype=torch.float32, device=tensor.device)
                decoded_pose_offset = (prob * project).sum(dim=-1) - (reg_max // 2)
                
            elif tensor.shape[1] == expected_fused_channels:
                # 【情况B：ONNX 引擎推理】模型在导出时已经融合了 DFL，直接取走坐标即可！
                pose_start = num_classes
                pose_end = num_classes + 8
                decoded_pose_offset = tensor[b, pose_start:pose_end, grid_y, grid_x].T
                
            else:
                raise ValueError(f"无法识别的模型输出维度: {tensor.shape[1]}")
        else:
            # GT 模式：自动兼容 9维 (1 conf + 8 pose) 或 13维 (1 conf + 4 box + 8 pose)
            if tensor.shape[1] == 9:
                decoded_pose_offset = tensor[b, 1:9, grid_y, grid_x].T
            else:
                decoded_pose_offset = tensor[b, 5:13, grid_y, grid_x].T
            
        decoded_pose = torch.zeros_like(decoded_pose_offset)
        
        for i in range(4): 
            px_norm = (decoded_pose_offset[:, i*2] + grid_x) / grid_w
            py_norm = (decoded_pose_offset[:, i*2 + 1] + grid_y) / grid_h
            
            decoded_pose[:, i*2] = px_norm * img_w
            decoded_pose[:, i*2 + 1] = py_norm * img_h
        
        keep_idx = keypoint_nms(decoded_pose, scores, dist_thresh=kpt_dist_thresh)
        
        dets = torch.cat([
            scores[keep_idx].unsqueeze(1), 
            item_classes[keep_idx].unsqueeze(1), 
            decoded_pose[keep_idx]
        ], dim=1)
        batch_results.append(dets.detach().cpu().numpy())
        
    return batch_results

if __name__ == "__main__":
    detector = RMDetector(reg_max=16, num_classes=12)
    dummy_input = torch.randn(1, 3, 416, 416)
    output = detector(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    # 分类(12) + 关键点(8 * 16 = 128) = 140
    print(f"输出形状 (多尺度列表): {[o.shape for o in output]}") 
    # 预期输出形状: [torch.Size([1, 140, 52, 52]), torch.Size([1, 140, 26, 26]), torch.Size([1, 140, 13, 13])]