import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from rich.console import Console
from rich.status import Status
from rich.panel import Panel

from src.training.src import RMDetector

# 初始化 rich 终端控制台
console = Console()

# ================= 新增：模型导出包装器 =================
class RMExportWrapper(nn.Module):
    """
    导出专用的包裹器。
    用于在导出 ONNX 时，提前将 DFL (分布焦点损失) 的连续坐标还原计算融合进计算图中，
    从而将网络输出的通道数从 141 (13分类 + 8*16特征) 直接降维到 21 (13分类 + 8坐标)，
    极大地减轻 CPU 和后续 C++ 推理代码的解析压力。
    """
    def __init__(self, base_model, num_classes=13, reg_max=16):
        super().__init__()
        self.model = base_model
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # 预定义积分矩阵 [0, 1, ..., 15]
        # 使用 register_buffer 确保它被作为常量导出到 ONNX 中
        project = torch.arange(reg_max, dtype=torch.float32)
        self.register_buffer('project', project.view(1, 1, reg_max, 1, 1))

    def forward(self, x):
        preds = self.model(x)
        fused_preds = []
        
        for pred in preds:
            b, c, h, w = pred.shape
            
            # 1. 拆分：分离出 分类得分 和 位姿分布
            cls_logits = pred[:, :self.num_classes, :, :]
            pose_logits = pred[:, self.num_classes:, :, :]
            
            # 2. Reshape: [B, 128, H, W] -> [B, 8, 16, H, W]
            pose_logits = pose_logits.view(b, 8, self.reg_max, h, w)
            
            # 3. Softmax: 在 16 这个维度上算概率
            pose_prob = F.softmax(pose_logits, dim=2)
            
            # 4. 积分计算坐标偏移量 (加权求和) -> 变为 [B, 8, H, W]
            pose_coords = (pose_prob * self.project).sum(dim=2) - (self.reg_max // 2)
            
            # 5. 重组：拼回 分类得分 和 真实的坐标偏移量
            fused_pred = torch.cat([cls_logits, pose_coords], dim=1)
            fused_preds.append(fused_pred)
            
        return fused_preds
# ========================================================

def export_onnx(model, dummy_input, output_path: Path, cfg):
    """导出 ONNX 格式并根据配置进行轻量化"""
    console.print(f"[*] 开始导出 ONNX 模型: [cyan]{output_path}[/cyan]")
    
    simplify = cfg['onnx'].get('simplify', True)
    opset_version = cfg['onnx'].get('opset', 18)
    
    with Status("[bold yellow]正在导出原生 ONNX 模型 (Legacy TorchScript 引擎)...", console=console):
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output_p3', 'output_p4', 'output_p5'], 
            dynamo=False
        )
    
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            with Status("[bold yellow]正在执行 ONNX 极致轻量化 (onnxsim)...", console=console):
                onnx_model = onnx.load(str(output_path))
                model_simp, check = onnx_simplify(onnx_model)
                
                if check:
                    onnx.save(model_simp, str(output_path))
                    console.print(f"[+] [bold green]ONNX 轻量化完成[/bold green]，文件已保存至: [cyan]{output_path}[/cyan]")
                else:
                    console.print("[-] [bold red]ONNX 轻量化校验失败，保留初始版本。[/bold red]")
        except ImportError:
            console.print("[-] [bold red]未检测到 onnx 或 onnxsim 库，跳过极致轻量化步骤。[/bold red]")
    else:
        console.print(f"[+] [bold green]ONNX 导出完成（按配置跳过轻量化）[/bold green]，文件已保存至: [cyan]{output_path}[/cyan]")

def export_torchscript(model, dummy_input, output_path: Path):
    """导出 TorchScript 格式"""
    console.print(f"[*] 开始导出 TorchScript 模型: [cyan]{output_path}[/cyan]")
    with Status("[bold yellow]正在跟踪生成 TorchScript 模型...", console=console):
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(output_path)) # type: ignore
    console.print(f"[+] [bold green]TorchScript 导出完成[/bold green]，文件已保存至: [cyan]{output_path}[/cyan]")

def main():
    config_file = Path("./config.yaml")
    if not config_file.exists():
        console.print(f"[bold red]错误：找不到配置文件 {config_file.absolute()}[/bold red]")
        return

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg_full = yaml.safe_load(f)
        
    if 'kielas_rm_export' not in cfg_full:
        console.print("[bold red]错误：配置文件中缺少 'kielas_rm_export' 模块。[/bold red]")
        return
        
    cfg = cfg_full['kielas_rm_export']
    
    weights_path = Path(cfg['weights'])
    output_dir = Path(cfg['output_dir'])
    formats = cfg.get('formats', [])
    input_size = cfg.get('input_size', [416, 416])
    
    reg_max = cfg.get('reg_max', 16)
    # 【修正】：类别数默认改为 13
    num_classes = cfg.get('num_classes', 13)

    if not weights_path.exists():
        console.print(f"[bold red]错误：权重文件不存在 {weights_path.absolute()}[/bold red]")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[*] [bold cyan]正在初始化模型并加载权重...[/bold cyan]")
    device = torch.device('cpu') 
    
    # 1. 先加载原生的训练模型
    base_model = RMDetector(reg_max=reg_max, num_classes=num_classes)
    base_model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # 2. 【修改点】：给它套上壳子，变成导出专用模型
    model = RMExportWrapper(base_model, num_classes=num_classes, reg_max=reg_max)
    model.eval() 
    
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0], device=device)
    model_name = weights_path.stem
    
    if "onnx" in formats:
        onnx_path = output_dir / f"{model_name}.onnx"
        export_onnx(model, dummy_input, onnx_path, cfg)
        
    if "torchscript" in formats:
        ts_path = output_dir / f"{model_name}.pt"
        export_torchscript(model, dummy_input, ts_path)

    console.print("\n[bold green]所有导出任务执行完毕！[/bold green]")
    console.print(Panel(f"模型导出目录: [cyan]{output_dir.absolute()}[/cyan]", title="任务完成"))

if __name__ == "__main__":
    with torch.no_grad():
        main()