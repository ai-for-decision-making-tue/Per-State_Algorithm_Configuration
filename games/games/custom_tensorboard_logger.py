from tianshou.utils import TensorboardLogger
from typing import Any
from tianshou.utils.logger.base import VALID_LOG_VALS_TYPE
import torch


class CustomTensorboardLogger(TensorboardLogger):
    """
    Tensorboard logger adapted to write some extra stats to tensorboard.
    """

    def write(self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.shape:
                nr_unique_values = len(v.unique())
                bins = list(e for e in range(0, v.max().item())) if nr_unique_values <= 21 else 201
                self.writer.add_histogram(k, v, global_step=step, bins=bins)
            else:
                self.writer.add_scalar(k, v, global_step=step)
        if self.write_flush:  # issue 580
            self.writer.flush()  # issue #482

    def save_extra_data(
        self,
        epoch: int,
        data: dict[str, Any],
    ) -> None:
        for k, v in data.items():
            save_loc = f"test/{k}"
            if k.startswith("pmf"):
                # Assuming v contains the frequencies for the histogram bins
                bucket_counts = v
                bucket_limits = torch.arange(1, len(v) + 1).float()  # Bin edges

                sum_v = bucket_counts.sum().item()
                sum_sq = (bucket_counts**2).sum().item()

                self.writer.add_histogram_raw(
                    tag=save_loc,
                    min=0,
                    max=bucket_limits.max().item(),
                    num=sum_v,  # Total number of values
                    sum=sum_v,
                    sum_squares=sum_sq,
                    bucket_limits=bucket_limits.tolist(),
                    bucket_counts=bucket_counts.tolist(),
                    global_step=epoch,
                )
            else:
                self.write(save_loc, epoch, {save_loc: v})
