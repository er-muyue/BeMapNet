import os
import time
import uuid
import torch
import subprocess
import numpy as np
from torch import nn
from loguru import logger
import torch.distributed as dist
from bemapnet.utils.env import get_root_dir
from bemapnet.utils.misc import parse_devices
from bemapnet.engine.callbacks import Callback


__all__ = ["ShareFSUUIDNameServer", "RlaunchReplicaEnv"]
output_root_dir = os.path.join(get_root_dir(), "outputs")


class ShareFSUUIDNameServer:
    def __init__(self, is_master):
        self.exp_id = self._get_exp_id()
        self.is_master = is_master
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def _get_exp_id(self):
        if "DET3D_EXPID" not in os.environ:
            if int(os.environ.get("RLAUNCH_REPLICA_TOTAL", 1)) == 1:
                return str(uuid.uuid4())
            msg = """cannot find DET3D_EXPID in environ please use following
            command DET3D_EXPID=$(cat /proc/sys/kernel/random/uuid) rlaunch ...
            """
            logger.error(msg)
            raise RuntimeError
        return str(os.environ["DET3D_EXPID"])

    @property
    def filepath(self):
        return os.path.join(output_root_dir, f"master_ip_{self.exp_id}.txt")

    def __enter__(self):
        if self.is_master:
            self.set_master()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.is_master:
            os.remove(self.filepath)

    def set_master(self):
        assert not os.path.exists(self.filepath)
        hostname = "Host"
        with open(self.filepath, "w") as f:
            f.write(hostname)

    def get_master(self):
        while True:
            if os.path.exists(self.filepath):
                with open(self.filepath, "r") as f:
                    return f.read()
            else:
                time.sleep(5)


class _DDPEnv(Callback):
    def __init__(self, sync_bn=0, devices=None, find_unused_parameters=False):
        if devices:
            devices = parse_devices(devices)
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
        self.nr_gpus = torch.cuda.device_count()
        self.sync_bn = sync_bn
        self.find_unused_parameters = find_unused_parameters

    @staticmethod
    def setup_nccl():
        ifname = filter(lambda x: x not in ("lo",), os.listdir("/sys/class/net/"))
        os.environ["NCCL_SOCKET_IFNAME"] = ",".join(ifname)
        os.environ["NCCL_IB_DISABLE"] = "1"

        # os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
        os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
            "cd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
            "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
            "| grep v >/dev/null && echo $i ; done; > /dev/null"
        )
        os.environ["NCCL_IB_GID_INDEX"] = "3"
        os.environ["NCCL_IB_TC"] = "106"

    def after_init(self, trainer):
        trainer.model.cuda()
        if int(self.sync_bn) > 1:
            ranks = np.arange(self.world_size()).reshape(-1, self.sync_bn)
            process_groups = [torch.distributed.new_group(list(pids)) for pids in ranks]
            trainer.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                trainer.model, process_groups[self.global_rank() // self.sync_bn]
            )
        elif int(self.sync_bn) == 1:
            trainer.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trainer.model)
        trainer.model = nn.parallel.DistributedDataParallel(
            trainer.model, device_ids=[self.local_rank()], find_unused_parameters=self.find_unused_parameters
        )

    def cleanup(self):
        dist.destroy_process_group()

    def init_dist(self):
        torch.cuda.set_device(self.local_rank())
        dist.init_process_group(
            backend="nccl",
            init_method=self._master_uri,
            rank=self.global_rank(),
            world_size=self.world_size(),
        )
        dist.barrier()


class RlaunchReplicaEnv(_DDPEnv):
    def __init__(self, sync_bn=0, devices=None, find_unused_parameters=False):
        super().__init__(sync_bn, devices, find_unused_parameters)

    def set_master_uri(self, ns):
        self._master_uri = f"tcp://{self.master_address(ns)}:{self.master_port()}"
        logger.info(self._master_uri)

    @staticmethod
    def is_brainpp_mm_env():
        return int(os.environ.get("RLAUNCH_REPLICA_TOTAL", 1)) > 1

    def master_address(self, ns) -> str:
        if self.node_rank() == 0:
            root_node = "localhost"
        else:
            root_node = ns.get_master()
        os.environ["MASTER_ADDR"] = root_node
        return root_node

    def master_port(self) -> int:
        port = os.environ.get("MASTER_PORT", 12345)
        os.environ["MASTER_PORT"] = str(port)
        return int(port)

    def world_size(self) -> int:
        return int(os.environ.get("RLAUNCH_REPLICA_TOTAL", 1)) * int(self.nr_gpus)

    def global_rank(self) -> int:
        return int(self.nr_gpus) * self.node_rank() + self.local_rank()

    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    def node_rank(self) -> int:
        return int(os.environ.get("RLAUNCH_REPLICA", 0))
