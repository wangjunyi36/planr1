"""
记录训练过程中的 GPU 显存使用量和每个 epoch 训练时间，并写入 TensorBoard

注意：在 MIG (Multi-Instance GPU) 分区上，torch.cuda 的显存统计 API 会触发 NVML 相关
的 PyTorch 已知 bug。本回调会检测 MIG 并跳过显存记录，仅保留 epoch 时间。
"""
import time
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only


BYTES_PER_GB = 1024 ** 3


def _is_mig_device():
    """检测是否为 MIG 分区 GPU（NVML 在此上有兼容问题）"""
    if not torch.cuda.is_available():
        return False
    try:
        device_name = torch.cuda.get_device_name(0)
        return 'MIG' in device_name
    except Exception:
        return False


class GPUMetricsCallback(Callback):
    """记录 GPU 显存和 epoch 训练时间的 PyTorch Lightning 回调"""

    def __init__(self):
        super().__init__()
        self._epoch_start_time = 0.0
        self._skip_gpu_memory = _is_mig_device()

    def on_train_epoch_start(self, trainer, pl_module):
        """Epoch 开始时记录时间，非 MIG 时重置显存峰值"""
        self._epoch_start_time = time.perf_counter()
        if torch.cuda.is_available() and not self._skip_gpu_memory:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """Epoch 结束时计算并记录显存和时间到 TensorBoard"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_time = time.perf_counter() - self._epoch_start_time
        pl_module.log('train/epoch_time_sec', elapsed_time, on_step=False, on_epoch=True, sync_dist=True)

        if torch.cuda.is_available() and not self._skip_gpu_memory:
            max_memory_gb = torch.cuda.max_memory_allocated() / BYTES_PER_GB
            pl_module.log('train/gpu_memory_gb', max_memory_gb, on_step=False, on_epoch=True, sync_dist=True)
