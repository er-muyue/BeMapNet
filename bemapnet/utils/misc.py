import os
import re
import torch
import torchvision
import unicodedata
from sys import stderr
from torch import Tensor
from loguru import logger
from argparse import Action
from collections import deque
from typing import Optional, List
from torch import distributed as dist


__all__ = [
    "PyDecorator", "NestedTensor", "AvgMeter", "DictAction", "sanitize_filename", "parse_devices", 
    "_max_by_axis", "nested_tensor_from_tensor_list", "_onnx_nested_tensor_from_tensor_list", 
    "get_param_groups", "setup_logger", "get_rank", "get_world_size", "synchronize", "reduce_sum", 
    "reduce_mean", "all_gather_object", "is_distributed", "is_available"
]


class PyDecorator:
    @staticmethod
    def overrides(interface_class):
        def overrider(method):
            assert method.__name__ in dir(interface_class), "{} function not in {}".format(
                method.__name__, interface_class
            )
            return method
        return overrider


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    

class AvgMeter(object):
    def __init__(self, window_size=50):
        self.window_size = window_size
        self._value_deque = deque(maxlen=window_size)
        self._total_value = 0.0
        self._wdsum_value = 0.0
        self._count_deque = deque(maxlen=window_size)
        self._total_count = 0.0
        self._wdsum_count = 0.0

    def reset(self):
        self._value_deque.clear()
        self._total_value = 0.0
        self._wdsum_value = 0.0
        self._count_deque.clear()
        self._total_count = 0.0
        self._wdsum_count = 0.0

    def update(self, value, n=1):
        if len(self._value_deque) >= self.window_size:
            self._wdsum_value -= self._value_deque.popleft()
            self._wdsum_count -= self._count_deque.popleft()
        self._value_deque.append(value * n)
        self._total_value += value * n
        self._wdsum_value += value * n
        self._count_deque.append(n)
        self._total_count += n
        self._wdsum_count += n

    @property
    def avg(self):
        return self.global_avg

    @property
    def global_avg(self):
        return self._total_value / max(self._total_count, 1e-5)

    @property
    def window_avg(self):
        return self._wdsum_value / max(self._wdsum_count, 1e-5)


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.
        All elements inside '()' or '[]' are treated as iterable values.
        Args:
            val (str): Value string.
        Returns:
            list | tuple: The expanded list or tuple from the string.
        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.
            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (char == ",") and (pre.count("(") == pre.count(")")) and (pre.count("[") == pre.count("]")):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


def sanitize_filename(value, allow_unicode=False):
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def parse_devices(gpu_ids):
    if "-" in gpu_ids:
        gpus = gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        parsed_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))
        return parsed_ids
    else:
        return gpu_ids


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def get_param_groups(model, optimizer_setup):
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model.named_parameters():
        if match_name_keywords(n, optimizer_setup["freeze_names"]):
            p.requires_grad = False

    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not match_name_keywords(n, optimizer_setup["backb_names"])
                and not match_name_keywords(n, optimizer_setup["extra_names"])
                and p.requires_grad
            ],
            "lr": optimizer_setup["base_lr"],
            "wd": optimizer_setup["wd"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, optimizer_setup["backb_names"]) and p.requires_grad
            ],
            "lr": optimizer_setup["backb_lr"],
            "wd": optimizer_setup["wd"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, optimizer_setup["extra_names"]) and p.requires_grad
            ],
            "lr": optimizer_setup["extra_lr"],
            "wd": optimizer_setup["wd"],
        },
    ]

    return param_groups


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): loaction to save log file
        distributed_rank(int): device rank when multi-gpu environment
        mode(str): log file write mode, `append` or `override`. default is `a`.
    Return:
        logger instance.
    """
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    format = f"[Rank #{distributed_rank}] | " + "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
    if distributed_rank > 0:
        logger.remove()
        logger.add(
            stderr,
            format=format,
            level="WARNING",
        )
    logger.add(
        save_file,
        format=format,
        filter="",
        level="INFO" if distributed_rank == 0 else "WARNING",
        enqueue=True,
    )

    return logger


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def synchronize():
    """Helper function to synchronize (barrier) among all processes when using distributed training"""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    current_world_size = dist.get_world_size()
    if current_world_size == 1:
        return
    dist.barrier()


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_mean(tensor):
    return reduce_sum(tensor) / float(get_world_size())


def all_gather_object(obj):
    world_size = get_world_size()
    if world_size < 2:
        return [obj]
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, obj)
    return output


def is_distributed() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def is_available() -> bool:
    return dist.is_available()
