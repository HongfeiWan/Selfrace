# 训练模块
import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
import json
from types import SimpleNamespace
from network import create_network

def check_gpu_info(print_info: bool = True, **kwargs):
    """
    检查GPU信息和CUDA支持情况

    Args:
        print_info: 是否打印函数内部的日志（默认True）。
        Print: 别名，兼容传入 Print=False 的调用方式。
    """
    # 兼容别名参数 Print=False 的用法
    if 'Print' in kwargs:
        try:
            print_info = bool(kwargs['Print'])
        except Exception:
            pass

    def log(*args, **kws):
        if print_info:
            print(*args, **kws)

    log("🔍 GPU 信息检测...")
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        log("✅ CUDA 可用")
        # 获取CUDA版本
        cuda_version = torch.version.cuda
        log(f"📋 CUDA 版本: {cuda_version}")
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        log(f"🎮 GPU 数量: {gpu_count}")
        # 获取当前GPU设备
        current_device = torch.cuda.current_device()
        log(f"🎯 当前GPU设备: {current_device}")
        # 获取GPU名称
        gpu_name = torch.cuda.get_device_name(current_device)
        log(f"🏷️  GPU名称: {gpu_name}")
        # 获取GPU内存信息
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
        gpu_memory_gb = gpu_memory / (1024**3)
        log(f"💾 GPU内存: {gpu_memory_gb:.2f} GB")
        # 检查分布式训练支持
        if dist.is_available():
            log("✅ PyTorch分布式训练支持可用")
            # 检查NCCL后端
            if dist.is_nccl_available():
                log("✅ NCCL后端可用")
            else:
                log("❌ NCCL后端不可用")
            # 检查GLOO后端
            if dist.is_gloo_available():
                log("✅ GLOO后端可用")
            else:
                log("❌ GLOO后端不可用")
        else:
            log("❌ PyTorch分布式训练支持不可用")
        # 显示所有GPU的详细信息
        log("\n📊 所有GPU详细信息:")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            log(f"  GPU {i}: {props.name}")
            log(f"    内存: {props.total_memory / (1024**3):.2f} GB")
            log(f"    计算能力: {props.major}.{props.minor}")
            log(f"    多处理器数量: {props.multi_processor_count}")
        # 返回CUDA rank列表
        cuda_ranks = list(range(gpu_count))
        return True, cuda_ranks
    else:
        log("❌ CUDA 不可用")
        log("📋 PyTorch版本:", torch.__version__)
        log("💡 请确保已正确安装CUDA和对应版本的PyTorch")
        return False, []

def gpu_train(cuda_ranks, config):
    """
    为每个CUDA设备分别创建模型与优化器，并将模型放入对应的cuda:rank设备。
    不使用分布式或DataParallel，直接一机多模型。
    Args:
        cuda_ranks: CUDA rank列表，例如 [0,1,2]
        config: 配置对象
    Returns:
        (models, optimizers, devices)
    """
    print(f"🚀 为各自CUDA设备创建独立模型 - CUDA Ranks: {cuda_ranks}")
    if not cuda_ranks:
        print("❌ 没有可用的CUDA设备")
        return None, None, None

    # 初始化容器
    models = []
    optimizers = []
    devices = []

    try:
        # 1) 设备列表
        devices = [torch.device(f"cuda:{rank}") for rank in cuda_ranks]

        # 2) 为每个设备创建模型与优化器
        print("🤖 为每块GPU创建模型与优化器...")
        for rank, device in zip(cuda_ranks, devices):
            torch.cuda.set_device(device)
            model = create_network(config=config, network_type="shared")
            model = model.to(device)
            models.append(model)

            optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
            optimizers.append(optimizer)

            print(f"  GPU {rank}: 模型与优化器创建完成")

        # 3) 打印显存
        print("💾 GPU内存使用情况:")
        for rank, device in zip(cuda_ranks, devices):
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            print(f"  GPU {rank}: {memory_allocated:.2f}GB (已分配) / {memory_reserved:.2f}GB (已保留)")

        # 4) 打印参数规模
        print("📊 模型与优化器统计:")
        for rank, model in zip(cuda_ranks, models):
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  GPU {rank}: 参数总数 {total_params:,}")

        return models, optimizers, devices

    except Exception as e:
        print(f"❌ GPU训练设置失败: {e}")
        cleanup_resources(models, optimizers)
        return None, None, None

def cleanup_resources(models, optimizers):
    """
    清理GPU资源
    """
    if models:
        for model in models:
            if hasattr(model, 'module'):
                del model.module
            del model
    if optimizers:
        for optimizer in optimizers:
            del optimizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 检查GPU信息
    cuda_available, cuda_ranks = check_gpu_info(Print=False)
    print(f"CUDA可用: {cuda_available}")
    print(f"CUDA rank列表: {cuda_ranks}")
    if cuda_available and cuda_ranks:
        # 读取配置文件
        config_path = 'configs/default_config.yaml'
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            config = json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))
            print("✅ 配置文件加载成功")
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            exit(1)
        # 创建分布式训练环境
        models, optimizers, devices = gpu_train(cuda_ranks, config)









