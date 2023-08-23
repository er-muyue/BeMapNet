import os
import torch
from tqdm import tqdm
from typing import Sequence
from bemapnet.engine.experiment import BaseExp
from bemapnet.utils.misc import get_rank, synchronize


__all__ = ["Callback", "BaseExecutor", "Trainer", "BeMapNetEvaluator"]


class Callback:

    # callback enabled rank list
    # None means callback is always enabled
    enabled_rank = None

    def setup(self, executor):
        pass

    def load_checkpoint(self, executor):
        pass

    def after_init(self, executor):
        pass

    def before_train(self, executor):
        pass

    def before_epoch(self, executor, epoch: int):
        pass

    def before_step(self, executor, step, data_dict):
        pass

    def before_backward(self, executor):
        pass

    def before_optimize(self, executor):
        pass

    def after_step(self, executor, step, data_dict, *args, **kwargs):
        pass

    def after_epoch(self, executor, epoch: int, update_best_ckpt: bool = False):
        pass

    def after_train(self, executor):
        pass


class BaseExecutor:
    def __init__(self, exp: BaseExp, callbacks: Sequence["Callback"], logger=None) -> None:
        self.exp = exp
        self.logger = logger
        self.callbacks = callbacks
        self._invoke_callback("setup")

        self.epoch = 0
        self.global_step = 0
        self._invoke_callback("load_checkpoint")
        self._invoke_callback("after_init")

    @property
    def train_dataloader(self):
        return self.exp.train_dataloader

    @property
    def val_dataloader(self):
        return self.exp.val_dataloader

    @property
    def model(self):
        return self.exp.model

    @model.setter
    def model(self, value):
        self.exp.model = value

    @property
    def optimizer(self):
        return self.exp.optimizer

    @property
    def lr_scheduler(self):
        return self.exp.lr_scheduler

    def _invoke_callback(self, callback_name, *args, **kwargs):
        for cb in self.callbacks:
            if cb.enabled_rank is None or self.global_rank in cb.enabled_rank:
                func = getattr(cb, callback_name, None)
                if func:
                    func(self, *args, **kwargs)

    @property
    def global_rank(self):
        return get_rank()


class Trainer(BaseExecutor):
    def __init__(
        self, exp: BaseExp, callbacks: Sequence["Callback"], logger=None, use_amp=False, evaluator=None
    ) -> None:
        super(Trainer, self).__init__(exp, callbacks, logger)
        self.use_amp = use_amp
        self.evaluator = evaluator
        if self.use_amp:
            self.grad_scaler = torch.cuda.amp.GradScaler()

    def train(self):
        self.train_iter = iter(self.train_dataloader)
        self._invoke_callback("before_train")
        self.model.cuda()
        self.model.train()
        self.optimizer_to(self.optimizer, next(self.model.parameters()).device)
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.exp.max_epoch):
            self.epoch = epoch
            self.model.train()
            self.train_epoch(epoch)
        self._invoke_callback("after_train")

    def train_epoch(self, epoch):
        self._invoke_callback("before_epoch", epoch)
        sampler = self.train_dataloader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for step in range(len(self.train_dataloader)):
            try:
                data = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_dataloader)
                data = next(self.train_iter)
            self.train_step(data, step)
        if self.evaluator is not None:
            self.evaluator.eval()
        self._invoke_callback("after_epoch", epoch, update_best_ckpt=False)

    def train_step(self, data, step):
        self._invoke_callback("before_step", step, data)
        self.lr_scheduler.step(self.global_step)
        self.model.train()
        self.optimizer.zero_grad()
        if not self.use_amp:
            ret = self.exp.training_step(data)
        else:
            with torch.cuda.amp.autocast():
                ret = self.exp.training_step(data)
        if isinstance(ret, torch.Tensor):
            loss = ret
            ext_dict = None
        elif isinstance(ret, tuple):
            loss, ext_dict = ret
            ext_dict = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in ext_dict.items()}
        else:
            raise TypeError
        self._invoke_callback("before_backward")
        if not self.use_amp:
            loss.backward()
            self._invoke_callback("before_optimize")
            self.optimizer.step()
        else:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)  # NOTE: grads are unscaled before "before_optimize" callbacks
            self._invoke_callback("before_optimize")
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        self._invoke_callback("after_step", step, data, loss=loss.detach(), extra=ext_dict)
        self.global_step += 1

    # refer to: https://github.com/pytorch/pytorch/issues/8741
    @staticmethod
    def optimizer_to(optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)


class BeMapNetEvaluator(BaseExecutor):
    def __init__(self, exp: BaseExp, callbacks: Sequence["Callback"], logger=None) -> None:
        super(BeMapNetEvaluator, self).__init__(exp, callbacks, logger)

    def eval(self, ckpt_name=None):

        exp = self.exp
        val_iter = iter(self.val_dataloader)

        self._invoke_callback("before_eval")

        if ckpt_name is not None:
            if get_rank() == 0:
                self.logger.info("Eval with best checkpoint!")
            path = os.path.join(exp.output_dir, 'dump_model', ckpt_name)
            checkpoint = torch.load(open(path, "rb"), map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint["model_state"], strict=False)

        self.model.cuda()
        self.model.eval()

        for step in tqdm(range(len(self.val_dataloader))):
            batch_data = next(val_iter)
            with torch.no_grad():
                exp.test_step(batch_data)
            self._invoke_callback("after_step", step, {})

        synchronize()

        if get_rank() == 0:
            self.logger.info("Done with inference, start evaluation later!")
            gt_dir = exp.exp_config.map_conf['anno_root']
            dt_dir = exp.evaluation_save_dir
            val_txts = exp.exp_config.VAL_TXT

            for val_txt in val_txts:
                ap_table = "".join(os.popen(f"python3 tools/evaluation/eval.py {gt_dir} {dt_dir} {val_txt}").readlines())
                self.logger.info(" AP-Performance with HDMapNetAPI: \n" + val_txt + "\n" + ap_table)

        self._invoke_callback("after_eval")
