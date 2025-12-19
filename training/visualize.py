import os
import sys
import json
from typing import Dict, Any, List

import torch

# 将工程根目录加入 sys.path，方便直接 import simulator / training 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.ddppo import ddppo  # noqa: E402


import queue

class TrainingVisualizer:
    """
    使用 vispy 实时可视化 ddppo 训练时某个 batch 的路网与所有 agent 状态。

    - 地图：来自 simulator.road_network.left_boundaries / right_boundaries
    - 车辆：来自 simulator.agents_state（默认显示 batch 0 的所有 M）
    """

    def __init__(self) -> None:
        from vispy import app, scene

        self.app = app
        self.scene = scene

        # 创建画布与单个视图
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            size=(1000, 800),
            title="Selfrace DDPPO Training - Road & Agents",
            show=True,
        )

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"
        self.view.padding = 10

        # 地图点云
        self.road_markers = scene.Markers(parent=self.view.scene)
        # agent 点云
        self.agent_markers = scene.Markers(parent=self.view.scene)

        self._road_sent = False
        
        # 线程安全队列：用于从训练线程传递数据到 GUI 线程
        self.data_queue = queue.Queue(maxsize=1)
        
        # 定时器：在 GUI 线程中轮询队列并更新画布
        self.timer = app.Timer(interval=0.05, connect=self._on_timer, start=True)

    # ------------------------------------------------------------------ #
    # 与 ddppo 的集成接口
    # ------------------------------------------------------------------ #
    def on_sim_step(self, simulator) -> None:
        """
        由 VisualDDPPO 在每次 _capture_agent_state_snapshot 时调用（运行在训练线程）。
        用于实时更新每一步的 agent 状态。
        """
        self._extract_and_queue(simulator)

    def on_ppo_update(self, metrics: Dict[str, Any], simulator) -> None:
        """
        由 VisualDDPPO 在每次 _ppo_update_from_buffer 返回 metrics 时调用（运行在训练线程）。
        """
        # 这里也可以更新一次，确保 update 后的状态也能显示
        self._extract_and_queue(simulator)

    def _extract_and_queue(self, simulator):
        import numpy as np

        if simulator is None:
            return

        data_packet = {}

        # 1) 初始化地图数据（仅第一次）
        if not self._road_sent:
            try:
                # 左右边界点；根据 road_network 结构可能是 (N,2) 或 (B,N,2)
                left = simulator.road_network.left_boundaries
                right = simulator.road_network.right_boundaries
                if left.dim() == 3:
                    left = left[0]
                if right.dim() == 3:
                    right = right[0]
                road_pts = torch.cat([left.reshape(-1, 2), right.reshape(-1, 2)], dim=0)
                road_np = road_pts.detach().cpu().numpy()
                
                data_packet['road_np'] = road_np
                self._road_sent = True
            except Exception as e:
                print(f"[Visualizer] 准备路网数据失败: {e}")

        # 2) 更新 agents_state（默认显示 batch 0 的所有 M）
        try:
            agents_state = simulator.agents_state  # (B, M, S)
            if agents_state is not None:
                # B, M, S = agents_state.shape
                b = 0
                state_b = agents_state[b].detach().cpu()
                active_mask = state_b[:, 6] > 0.5
                if active_mask.any():
                    pos = state_b[active_mask, :2].numpy()
                else:
                    pos = state_b[:, :2].numpy()
                
                data_packet['agents_np'] = pos
        except Exception as e:
            print(f"[Visualizer] 准备 Agent 数据失败: {e}")

        # 将数据放入队列（丢弃旧帧以保持实时性）
        if data_packet:
            if self.data_queue.full():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    pass
            self.data_queue.put(data_packet)

    def _on_timer(self, event):
        """
        运行在 GUI 主线程，定期检查队列并更新画面。
        """
        try:
            data = self.data_queue.get_nowait()
        except queue.Empty:
            return

        # 更新路网
        if 'road_np' in data:
            road_np = data['road_np']
            self.road_markers.set_data(road_np, face_color=(0.2, 0.2, 0.2, 1.0), size=1.5)
            
            # 调整视口
            x_min, y_min = road_np.min(axis=0)
            x_max, y_max = road_np.max(axis=0)
            self.view.camera.rect = (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))

        # 更新 Agent
        if 'agents_np' in data:
            agents_np = data['agents_np']
            # 统一用红色小点表示车辆
            self.agent_markers.set_data(agents_np, face_color=(1.0, 0.0, 0.0, 1.0), size=4.0)


class VisualDDPPO(ddppo):
    """
    继承 ddppo，在 _ppo_update_from_buffer 里把 metrics 送给可视化器。
    其余训练流程保持与原版 ddppo 完全一致。
    """

    def __init__(self, config_path: str, visualizer: TrainingVisualizer) -> None:
        """
        可视化版本：
        - 不启动多进程，而是在当前进程中以单 GPU 方式直接调用 ddppo_worker。
        - 训练过程在后台线程中运行，metrics 通过 visualizer 实时展示。
        """
        self._visualizer = visualizer
        # 只做配置初始化，不启动多进程
        self._init_config_only(config_path)

        # 简化版：始终使用单 GPU（rank=0, world_size=1）
        cuda_available, cuda_ranks = self.check_gpu_info(print_info=True)
        if not cuda_available or not cuda_ranks:
            raise RuntimeError("没有可用的CUDA设备用于可视化训练")

        world_size = 1
        rank = 0
        master_addr = "127.0.0.1"
        master_port = self._find_free_port()
        store_port = self._find_free_port()

        # 直接在当前进程中以 rank=0 运行 worker（Windows 上本来就不启用分布式）
        self.ddppo_worker(rank, world_size, master_addr, master_port, store_port)

    def _capture_agent_state_snapshot(self, simulator):
        """
        重写此方法以在每个模拟步捕获状态时更新可视化。
        """
        # 调用父类方法获取 snapshot
        snapshot = super()._capture_agent_state_snapshot(simulator)
        
        # 更新可视化
        if getattr(self, "_visualizer", None) is not None:
            self._visualizer.on_sim_step(simulator)
            
        return snapshot

    def _ppo_update_from_buffer(
        self,
        states_buffer,
        rewards_buffer,
        dones_buffer,
        values_buffer,
        old_log_probs_buffer,
        actions_buffer,
        valid_mask,
        bootstrap_value,
        simulator,
        policy_net,
        value_net,
        optimizer_policy,
        optimizer_value,
        device,
        is_master,
        extra_metrics=None,
    ):
        # 调用原始实现
        metrics = super()._ppo_update_from_buffer(
            states_buffer,
            rewards_buffer,
            dones_buffer,
            values_buffer,
            old_log_probs_buffer,
            actions_buffer,
            valid_mask,
            bootstrap_value,
            simulator,
            policy_net,
            value_net,
            optimizer_policy,
            optimizer_value,
            device,
            is_master,
            extra_metrics,
        )

        # 仅在 master 进程、metrics 有效时更新可视化
        if is_master and metrics is not None and getattr(self, "_visualizer", None) is not None:
            try:
                # 将当前 simulator 一并传入，用于绘制路网与 agents_state
                self._visualizer.on_ppo_update(metrics, simulator)
            except Exception as e:
                print(f"[VisualDDPPO] 可视化更新失败: {e}")

        return metrics


def main():
    """
    入口：
    - 使用 vispy 打开一个训练可视化窗口；
    - 构造 VisualDDPPO，启动完整的 ddppo 训练（包括多进程）；
    - 在训练过程中，PPO 更新的 metrics 会实时绘制到窗口中。

    用法示例：
        python -m training.visualize
    """
    try:
        from vispy import app  # noqa: F401
    except Exception as e:  # pragma: no cover - 运行时环境问题
        print("导入 vispy 失败，请先安装依赖：pip install vispy")
        print(f"错误信息: {e}")
        return

    # 1. 创建可视化器（先启动 vispy 画布）
    visualizer = TrainingVisualizer()

    # 2. 配置路径（与 ddppo 主函数一致）
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "default_config.json")
    if not os.path.isfile(cfg_path):
        print(f"找不到配置文件: {cfg_path}")
        return

    # 3. 在后台线程/子进程中启动训练
    #    VisualDDPPO.__init__ 内部会按照 ddppo 的逻辑启动多进程训练并阻塞，
    #    因此这里推荐在单独线程中构造它，以免阻塞 vispy 事件循环。
    import threading

    def training_thread():
        print("[Visualizer] 启动 VisualDDPPO 训练进程...")
        VisualDDPPO(config_path=cfg_path, visualizer=visualizer)
        print("[Visualizer] 训练结束。")

    th = threading.Thread(target=training_thread, daemon=True)
    th.start()

    # 4. 进入 vispy 事件循环，直到窗口关闭
    from vispy import app

    app.run()


if __name__ == "__main__":
    main()


