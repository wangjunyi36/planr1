"""
记录每个 step 的「等数据时间」和「训练步时间」，用于判断瓶颈在数据还是 GPU。
- data_wait_sec：上一 step 结束到本 step 开始（主要是 DataLoader 取下一批的时间）
- step_sec：本 step 开始到结束（forward + backward）
"""
import time
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only


class DataStepTimingCallback(Callback):
    """按 step 记录数据等待时间与训练步时间，写入 TensorBoard"""

    def __init__(self):
        super().__init__()
        self._last_batch_end_time = None
        self._step_start_time = None
        self._data_wait_times = []
        self._step_times = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if self._last_batch_end_time is not None:
            data_wait = now - self._last_batch_end_time
            self._data_wait_times.append(data_wait)
        self._step_start_time = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        if self._step_start_time is not None:
            step_sec = now - self._step_start_time
            self._step_times.append(step_sec)
        self._last_batch_end_time = now

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if not self._data_wait_times or not self._step_times:
            return
        avg_data_wait = sum(self._data_wait_times) / len(self._data_wait_times)
        avg_step = sum(self._step_times) / len(self._step_times)
        total = avg_data_wait + avg_step
        pl_module.log('timing/avg_data_wait_sec', avg_data_wait, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('timing/avg_step_sec', avg_step, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('timing/data_wait_ratio', avg_data_wait / total if total > 0 else 0.0, on_step=False, on_epoch=True, sync_dist=True)
        self._data_wait_times = []
        self._step_times = []
