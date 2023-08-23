import os
import sys
import argparse
import datetime
import warnings
import subprocess
from bemapnet.engine.executor import Trainer, BeMapNetEvaluator
from bemapnet.engine.environ import ShareFSUUIDNameServer, RlaunchReplicaEnv
from bemapnet.engine.callbacks import CheckPointLoader, CheckPointSaver, ClearMLCallback, ProgressBar
from bemapnet.engine.callbacks import TensorBoardMonitor, TextMonitor, ClipGrad
from bemapnet.utils.env import collect_env_info, get_root_dir
from bemapnet.utils.misc import setup_logger, sanitize_filename, PyDecorator, all_gather_object


__all__ = ["BaseCli", "BeMapNetCli"]


class BaseCli:
    """Command line tools for any exp."""

    def __init__(self, Exp):
        """Make sure the order of initialization is: build_args --> build_env --> build_exp,
        since experiments depend on the environment and the environment depends on args.

        Args:
            Exp : experiment description class
        """
        self.ExpCls = Exp
        self.args = self._get_parser(Exp).parse_args()
        self.env = RlaunchReplicaEnv(self.args.sync_bn, self.args.devices, self.args.find_unused_parameters)

    @property
    def exp(self):
        if not hasattr(self, "_exp"):
            exp = self.ExpCls(
                **{x if y is not None else "none": y for (x, y) in vars(self.args).items()},
                total_devices=self.env.world_size(),
            )
            self.exp_updated_cfg_msg = exp.update_attr(self.args.exp_options)
            self._exp = exp
        return self._exp

    def _get_parser(self, Exp):
        parser = argparse.ArgumentParser()
        parser = Exp.add_argparse_args(parser)
        parser = self.add_argparse_args(parser)
        return parser

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser):
        parser.add_argument("--eval", dest="eval", action="store_true", help="conduct evaluation only")
        parser.add_argument("-te", "--train_and_eval", dest="train_and_eval", action="store_true", help="train+eval")
        parser.add_argument("--find_unused_parameters", dest="find_unused_parameters", action="store_true")
        parser.add_argument("-d", "--devices", default="0-7", type=str, help="device for training")
        parser.add_argument("--ckpt", type=str, default=None, help="checkpoint to start from or be evaluated")
        parser.add_argument("--pretrained_model", type=str, default=None, help="pretrained_model used by training")
        parser.add_argument("--sync_bn", type=int, default=0, help="0-> disable sync_bn, 1-> whole world")
        clearml_parser = parser.add_mutually_exclusive_group(required=False)
        clearml_parser.add_argument("--clearml", dest="clearml", action="store_true", help="enabel clearml for train")
        clearml_parser.add_argument("--no-clearml", dest="clearml", action="store_false", help="disable clearml")
        parser.set_defaults(clearml=True)
        return parser

    def _get_exp_output_dir(self):
        exp_dir = os.path.join(os.path.join(get_root_dir(), "outputs"), sanitize_filename(self.exp.exp_name))
        os.makedirs(exp_dir, exist_ok=True)
        output_dir = None
        if self.args.ckpt:
            output_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.args.ckpt)))
        elif self.env.global_rank() == 0:
            output_dir = os.path.join(exp_dir, datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
            os.makedirs(output_dir, exist_ok=True)
            # make a symlink "latest"
            symlink, symlink_tmp = os.path.join(exp_dir, "latest"), os.path.join(exp_dir, "latest_tmp")
            if os.path.exists(symlink_tmp):
                os.remove(symlink_tmp)
            os.symlink(os.path.relpath(output_dir, exp_dir), symlink_tmp)
            os.rename(symlink_tmp, symlink)
        output_dir = all_gather_object(output_dir)[0]
        return output_dir

    def get_evaluator(self, callbacks=None):
        exp = self.exp
        if self.args.ckpt is None:
            warnings.warn("No checkpoint is specified for evaluation")
        if exp.eval_executor_class is None:
            sys.exit("No evaluator is specified for evaluation")

        output_dir = self._get_exp_output_dir()
        logger = setup_logger(output_dir, distributed_rank=self.env.global_rank(), filename="eval.log")
        self._set_basic_log_message(logger)
        if callbacks is None:
            callbacks = [self.env, CheckPointLoader(self.args.ckpt)]
        evaluator = exp.eval_executor_class(exp=exp, callbacks=callbacks, logger=logger)
        return evaluator

    def _set_basic_log_message(self, logger):
        logger.opt(ansi=True).info("<yellow>Cli arguments:</yellow>\n<blue>{}</blue>".format(self.args))
        logger.info(f"exp_name: {self.exp.exp_name}")
        logger.opt(ansi=True).info(
            "<yellow>Used experiment configs</yellow>:\n<blue>{}</blue>".format(self.exp.get_cfg_as_str())
        )
        if self.exp_updated_cfg_msg:
            logger.opt(ansi=True).info(
                "<yellow>List of override configs</yellow>:\n<blue>{}</blue>".format(self.exp_updated_cfg_msg)
            )
        logger.opt(ansi=True).info("<yellow>Environment info:</yellow>\n<blue>{}</blue>".format(collect_env_info()))

    def get_trainer(self, callbacks=None, evaluator=None):
        args = self.args
        exp = self.exp
        if evaluator is not None:
            output_dir = self.exp.output_dir
        else:
            output_dir = self._get_exp_output_dir()

        logger = setup_logger(output_dir, distributed_rank=self.env.global_rank(), filename="train.log")
        self._set_basic_log_message(logger)

        if callbacks is None:
            callbacks = [
                self.env,
                ProgressBar(logger=logger),
                TextMonitor(interval=exp.print_interval),
                TensorBoardMonitor(os.path.join(output_dir, "tensorboard"), interval=exp.print_interval),
                CheckPointSaver(
                    local_path=os.path.join(output_dir, "dump_model"),
                    remote_path=exp.ckpt_oss_save_dir,
                    save_interval=exp.dump_interval,
                    num_keep_latest=exp.num_keep_latest_ckpt,
                ),
            ]
        if "grad_clip_value" in exp.__dict__:
            callbacks.append(ClipGrad(exp.grad_clip_value))
        if args.clearml:
            callbacks.append(ClearMLCallback())
        if args.ckpt:
            callbacks.append(CheckPointLoader(args.ckpt))
        if args.pretrained_model:
            callbacks.append(CheckPointLoader(args.pretrained_model, weight_only=True))
        callbacks.extend(exp.callbacks)

        trainer = Trainer(exp=exp, callbacks=callbacks, logger=logger, evaluator=evaluator)
        return trainer

    def executor(self):
        if self.args.eval:
            self.get_evaluator().eval()
        elif self.args.train_and_eval:
            evaluator = self.get_evaluator(callbacks=[])
            self.get_trainer(evaluator=evaluator).train()
        else:
            self.get_trainer().train()

    def dispatch(self, executor_func):
        is_master = self.env.global_rank() == 0
        with ShareFSUUIDNameServer(is_master) as ns:
            self.env.set_master_uri(ns)
            self.env.setup_nccl()
            if self.env.local_rank() == 0:
                command = sys.argv.copy()
                command[0] = os.path.abspath(command[0])
                command = [sys.executable] + command
                for local_rank in range(1, self.env.nr_gpus):
                    env_copy = os.environ.copy()
                    env_copy["LOCAL_RANK"] = f"{local_rank}"
                    subprocess.Popen(command, env=env_copy)
            self.env.init_dist()
        executor_func()

    def run(self):
        self.dispatch(self.executor)


class BeMapNetCli(BaseCli):
    @PyDecorator.overrides(BaseCli)
    def get_evaluator(self, callbacks=None):
        exp = self.exp

        output_dir = self._get_exp_output_dir()
        self.exp.output_dir = output_dir
        logger = setup_logger(output_dir, distributed_rank=self.env.global_rank(), filename="eval.log")
        self._set_basic_log_message(logger)
        if callbacks is None:
            callbacks = [
                self.env,
                CheckPointLoader(self.args.ckpt),
            ]

        evaluator = BeMapNetEvaluator(exp=exp, callbacks=callbacks, logger=logger)
        return evaluator
