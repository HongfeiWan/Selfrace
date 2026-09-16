"""Manual forward/backward diagnostic for the independent actor/critic."""

from _bootstrap import PROJECT_ROOT, add_project_paths

add_project_paths()

import json
import traceback
from types import SimpleNamespace

import yaml
import torch

from network import create_network


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


print("🧪 测试独立 actor/critic 网络...")
try:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_path = PROJECT_ROOT / "configs" / "default_config.yaml"
    with open(config_path, "r", encoding="utf-8") as stream:
        config_dict = yaml.safe_load(stream)
    config = json.loads(json.dumps(config_dict), object_hook=lambda value: SimpleNamespace(**value))

    model = create_network(config=config, network_type="independent").to(device)
    print(f"🔧 device={device}, parameters={count_parameters(model):,}")

    batch_size = 2000
    max_agents = 150
    feature_width = model.policy_feature_encoder.total_input_dim
    features = torch.randn(batch_size, max_agents, feature_width, device=device)

    action_logits, values = model(features, mode="both")
    print(f"✅ both: logits={tuple(action_logits.shape)}, values={tuple(values.shape)}")
    print(f"✅ policy: {tuple(model(features, mode='policy').shape)}")
    print(f"✅ value: {tuple(model(features, mode='value').shape)}")

    (action_logits.sum() + values.sum()).backward()
    print("✅ backward")
except Exception as exc:
    print(f"❌ 测试失败: {exc}")
    traceback.print_exc()
