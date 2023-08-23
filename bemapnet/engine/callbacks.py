import os
import glob
import tqdm
import torch
import pickle
from clearml import Task
from loguru import logger
from typing import Callable, Optional
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from bemapnet.engine.executor import Callback, BaseExecutor, Trainer
from bemapnet.utils.misc import AvgMeter


__all__ = ["Callback", "MasterOnlyCallback", "CheckPointSaver", "CheckPointLoader", "CheckPointC2Loader",
           "ClearMLCallback", "EvalResultsSaver", "LamdaCallback", "ClipGrad", "ProgressBar",
           "LearningRateMonitor", "TextMonitor", "TensorBoardMonitor"]


class MasterOnlyCallback(Callback):
    enabled_rank = [0]


class CheckPointSaver(MasterOnlyCallback):
    def __init__(
        self,
        local_path,
        filename=r"checkpoint_epoch_{epoch}.pth",
        remote_path=None,
        save_interval: int = 1,
        num_keep_latest=None,
    ):
        self.local_path = local_path
        self.filename = filename
        self.remote_path = remote_path
        self.save_interval = save_interval
        self.num_keep_latest = num_keep_latest
        os.makedirs(local_path, exist_ok=True)

    def _make_checkpoint(self, trainer: Trainer):
        model_state = None
        if hasattr(trainer, "ema_model"):
            model = trainer.ema_model.ema
        else:
            model = trainer.model
        if model:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model_state = model.module.state_dict()
                model_state_cpu = type(model_state)()
                for key, val in model_state.items():
                    model_state_cpu[key] = val.cpu()
                model_state = model_state_cpu
            else:
                model_state = model.state_dict()

        optim_state = trainer.optimizer.state_dict() if trainer.optimizer else None

        callback_states = {}
        for cb in trainer.callbacks:
            if hasattr(cb, "state_dict"):
                cls_name = cb.__class__.__name__
                callback_states[cls_name] = cb.state_dict()

        ckpt = {
            "epoch": trainer.epoch,
            "it": trainer.global_step,
            "global_step": trainer.global_step,
            "model_state": model_state,
            "optimizer_state": optim_state,
            "callback": callback_states,
        }

        # save grad_scaler
        if hasattr(trainer, "grad_scaler"):
            ckpt["grad_scaler_state"] = trainer.grad_scaler.state_dict()

        return ckpt

    def after_epoch(self, trainer: Trainer, epoch: int, update_best_ckpt: bool = False):
        if (epoch + 1) % self.save_interval != 0:
            return
        filename = self.filename.format(epoch=epoch)
        save_path = os.path.join(self.local_path, filename)
        torch.save(self._make_checkpoint(trainer), save_path)
        if update_best_ckpt:
            torch.save(self._make_checkpoint(trainer), os.path.join(self.local_path, f"checkpoint_best.pth"))
        self._remove_out_of_date_ckpt()

    def _remove_out_of_date_ckpt(self):
        if not self.num_keep_latest:
            return

        ckpt_list = glob.glob(os.path.join(self.local_path, self.filename.format(epoch="*")))
        ckpt_list.sort(key=os.path.getmtime)
        if len(ckpt_list) > self.num_keep_latest:
            for cur_file_idx in range(0, len(ckpt_list) - self.num_keep_latest):
                os.remove(ckpt_list[cur_file_idx])


class CheckPointLoader(Callback):
    def __init__(
        self,
        path,
        weight_only=False,
    ):
        self.path = path
        self.weight_only = weight_only

    def load_checkpoint(self, trainer: Trainer):
        logger.info(f"Loading parameters from checkpoint {self.path}")
        with open(self.path, "rb") as f:
            checkpoint = torch.load(f, map_location=torch.device("cpu"))

        # TODO bulid model finetune callback
        model_state_dict = trainer.model.state_dict()
        checkpoint_state_dict = checkpoint["model_state"]
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    logger.info(
                        "'{}' has shape {} in the checkpoint but {} in the "
                        "model! Skipped.".format(k, shape_checkpoint, shape_model)
                    )
                    checkpoint_state_dict.pop(k)
        trainer.model.load_state_dict(checkpoint_state_dict, strict=False)

        if self.weight_only:
            return

        trainer.epoch = checkpoint.get("epoch", -1) + 1
        trainer.global_step = checkpoint.get("global_step", -1) + 1
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        # resume callback
        for cb in trainer.callbacks:
            if hasattr(cb, "state_dict"):
                cls_name = cb.__class__.__name__
                if cls_name in checkpoint["callback"]:
                    cb.load_state_dict(checkpoint["callback"][cls_name])
        # resume grad_scaler
        if hasattr(trainer, "grad_scaler") and "grad_scaler_state" in checkpoint:
            trainer.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])


class ClearMLCallback(MasterOnlyCallback):
    def __init__(self):
        super().__init__()
        self.task_id = None

    def after_init(self, executor: BaseExecutor):
        if self.task_id is None:
            self.task = Task.init(
                project_name="det3d",
                task_name=executor.exp.exp_name,
                auto_connect_frameworks={"pytorch": False},
                reuse_last_task_id=False,
                continue_last_task=False,
            )
        else:
            self.task = Task.get_task(task_id=self.task_id)
            self.task.add_tags(["resume"])
            logger.info(f"continue from clearml task {self.task_id}")
        self.task.connect(executor.exp)
        if hasattr(executor.exp, "get_pcdet_cfg"):
            self.task.connect(executor.exp.get_pcdet_cfg(), "pcdet_config")

    def state_dict(self):
        return {"task_id": self.task.task_id}

    def load_state_dict(self, state_dict):
        self.task_id = state_dict["task_id"]


class EvalResultsSaver(MasterOnlyCallback):
    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def after_eval(self, executor, det_annos: list):
        out_file = os.path.join(self.out_dir, "result.pkl")
        pickle.dump(det_annos, open(out_file, "wb"))


class LamdaCallback:
    def __init__(
        self,
        setup: Optional[Callable] = None,
        load_checkpoint: Optional[Callable] = None,
        after_init: Optional[Callable] = None,
        before_train: Optional[Callable] = None,
        before_epoch: Optional[Callable] = None,
        before_step: Optional[Callable] = None,
        before_backward: Optional[Callable] = None,
        before_optimize: Optional[Callable] = None,
        after_step: Optional[Callable] = None,
        after_epoch: Optional[Callable] = None,
        after_train: Optional[Callable] = None,
    ) -> None:
        for k, v in locals().items():
            if k == "self":
                continue
            if v is not None:
                setattr(self, k, v)


class ClipGrad(Callback):
    def __init__(self, max_norm: float):
        self.max_norm = max_norm

    def before_optimize(self, trainer):
        clip_grad_norm_(trainer.model.parameters(), self.max_norm)


class ProgressBar(MasterOnlyCallback):
    def __init__(self, logger=None) -> None:
        self.epoch_bar = None
        self.step_bar = None
        self.logger = logger

    def setup(self, trainer: Trainer):
        self.epoch_bar = tqdm.tqdm(initial=0, total=trainer.exp.max_epoch, desc="[Epoch]", dynamic_ncols=True)
        self.step_bar = tqdm.tqdm(initial=0, desc="[Step]", dynamic_ncols=True, leave=False)
        if self.logger:
            self.logger.remove(0)
            self.logger.add(lambda msg: self.step_bar.write(msg, end=""))

    def before_epoch(self, trainer: Trainer, epoch: int):
        self.epoch_bar.update(epoch - self.epoch_bar.n)
        self.step_bar.reset(len(trainer.train_dataloader))

    def after_step(self, trainer: Trainer, step, data_dict, *args, **kwargs):
        self.step_bar.update()

    def after_train(self, trainer: Trainer):
        if self.step_bar:
            self.step_bar.close()
        if self.epoch_bar:
            self.epoch_bar.close()


class LearningRateMonitor:
    def _get_learning_rate(self, optimizer):
        if hasattr(optimizer, "lr"):
            lr = float(optimizer.lr)
        else:
            lr = optimizer.param_groups[0]["lr"]
        return lr


class TextMonitor(MasterOnlyCallback, LearningRateMonitor):
    def __init__(self, interval=10):
        self.interval = interval
        self.avg_loss = AvgMeter()
        self.ext_dict = None

    def after_step(self, trainer: Trainer, step, data_dict, *args, **kwargs):
        self.avg_loss.update(kwargs["loss"])

        lr = self._get_learning_rate(trainer.optimizer)

        ext_info = ""
        if kwargs["extra"] is not None:
            if self.ext_dict is None:
                self.ext_dict = {k: AvgMeter() for k in kwargs["extra"]}
            for key, val in kwargs["extra"].items():
                self.ext_dict[key].update(val)
            ext_info = "".join([f" {k}={v.window_avg :.4f}" for k, v in self.ext_dict.items()])

        if step % self.interval != 0:
            return

        trainer.logger.info(
            f"e:{trainer.epoch}[{step}/{self.total_step}] lr={lr :.6f} loss={self.avg_loss.window_avg :.4f}{ext_info}"
        )

    def before_epoch(self, trainer: Trainer, epoch: int):
        lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.logger.info(f"e:{epoch} learning rate = {lr :.6f}")
        self.total_step = len(trainer.train_dataloader)


class TensorBoardMonitor(MasterOnlyCallback, LearningRateMonitor):
    def __init__(self, log_dir, interval=10):
        os.makedirs(log_dir, exist_ok=True)
        self.tb_log = SummaryWriter(log_dir=log_dir)
        self.interval = interval

    def after_step(self, trainer: Trainer, step, data_dict, *args, **kwargs):
        accumulated_iter = trainer.global_step
        if accumulated_iter % self.interval != 0:
            return
        lr = self._get_learning_rate(trainer.optimizer)
        self.tb_log.add_scalar("epoch", trainer.epoch, accumulated_iter)
        self.tb_log.add_scalar("train/loss", kwargs["loss"], accumulated_iter)
        self.tb_log.add_scalar("meta_data/learning_rate", lr, accumulated_iter)
        if kwargs["extra"] is not None:
            for key, val in kwargs["extra"].items():
                self.tb_log.add_scalar(f"train/{key}", val, accumulated_iter)
