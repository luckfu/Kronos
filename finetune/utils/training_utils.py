import os
import random
import datetime
import numpy as np
import torch
import torch.distributed as dist


def setup_ddp():
    """
    Initializes the distributed data parallel environment.

    This function relies on environment variables set by `torchrun` or a similar
    launcher. It initializes the process group and sets the CUDA device for the
    current process.

    Returns:
        tuple: A tuple containing (rank, world_size, local_rank).
    """
    if os.getenv("KRONOS_DEVICE", "").lower() in {"xla", "tpu"}:
        try:
            import torch_xla.core.xla_model as xm
            try:
                import torch_xla.runtime as xr
            except ImportError:
                xr = None
        except ImportError as exc:
            raise RuntimeError("TPU requested but torch_xla is not installed") from exc

        # PJRT moved topology helpers from xla_model to torch_xla.runtime.
        def runtime_value(name, legacy_name, default):
            if xr is not None and hasattr(xr, name):
                return int(getattr(xr, name)())
            if hasattr(xm, legacy_name):
                return int(getattr(xm, legacy_name)())
            return int(default)

        # A one-core smoke deliberately forces a logical world size of one.
        # The eight-core Kaggle launcher instead uses one worker thread per
        # device.  Environment ordinals are process-wide in that mode, so the
        # thread-local PJRT runtime API must be authoritative.
        single_process = os.getenv(
            "KRONOS_XLA_SINGLE_PROCESS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        thread_per_device = os.getenv(
            "KRONOS_XLA_THREAD_PER_DEVICE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if single_process:
            rank, world_size, local_rank = 0, 1, 0
        elif thread_per_device:
            rank = runtime_value("global_ordinal", "get_ordinal", 0)
            default_cores = int(os.getenv("KRONOS_TPU_CORES", "8"))
            world_size = runtime_value("addressable_device_count", "xrt_world_size", default_cores)
            if world_size <= 1 and default_cores > 1:
                world_size = default_cores
            local_rank = runtime_value("local_ordinal", "get_local_ordinal", rank)
        else:
            rank = int(os.environ["ORDINAL"]) if "ORDINAL" in os.environ else runtime_value(
                "global_ordinal", "get_ordinal", 0
            )
            world_size = (
                int(os.environ["WORLD_SIZE"])
                if "WORLD_SIZE" in os.environ
                else runtime_value("world_size", "xrt_world_size", 1)
            )
            if "LOCAL_ORDINAL" in os.environ:
                local_rank = int(os.environ["LOCAL_ORDINAL"])
            elif "LOCAL_RANK" in os.environ:
                local_rank = int(os.environ["LOCAL_RANK"])
            else:
                local_rank = runtime_value("local_ordinal", "get_local_ordinal", rank)
        print(
            f"[XLA Setup] Global Rank: {rank}/{world_size}, "
            f"Local Rank: {local_rank}"
        )
        return rank, world_size, local_rank

    if "WORLD_SIZE" not in os.environ:
        # Apple MPS and CPU training use a normal single-process loop.
        print("[DDP Setup] WORLD_SIZE is not set; using single-process training.")
        return 0, 1, 0

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    backend = os.environ.get("DIST_BACKEND", "nccl")
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"
    dist.init_process_group(backend=backend)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    print(
        f"[DDP Setup] Global Rank: {rank}/{world_size}, "
        f"Local Rank: {local_rank}, backend={backend}"
    )
    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0):
    """
    Sets the random seed for reproducibility across all relevant libraries.

    Args:
        seed (int): The base seed value.
        rank (int): The process rank, used to ensure different processes have
                    different seeds, which can be important for data loading.
    """
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        # The two lines below can impact performance, so they are often
        # reserved for final experiments where reproducibility is critical.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    """
    Calculates the number of trainable parameters in a PyTorch model and returns
    it as a human-readable string.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        str: A string representing the model size (e.g., "175.0B", "7.1M", "50.5K").
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"  # Billions
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"  # Millions
    else:
        return f"{total_params / 1e3:.1f}K"  # Thousands


def reduce_tensor(tensor: torch.Tensor, world_size: int, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduces a tensor's value across all processes in a distributed setup.

    Args:
        tensor (torch.Tensor): The tensor to be reduced.
        world_size (int): The total number of processes.
        op (dist.ReduceOp, optional): The reduction operation (SUM, AVG, etc.).
                                      Defaults to dist.ReduceOp.SUM.

    Returns:
        torch.Tensor: The reduced tensor, which will be identical on all processes.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    # Note: `dist.ReduceOp.AVG` is available in newer torch versions.
    # For compatibility, manual division is sometimes used after a SUM.
    if op == dist.ReduceOp.AVG:
        rt /= world_size
    return rt


def format_time(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable H:M:S string.

    Args:
        seconds (float): The total seconds.

    Returns:
        str: The formatted time string (e.g., "0:15:32").
    """
    return str(datetime.timedelta(seconds=int(seconds)))



