"""Small, optional acceleration regions; never capture a mutable simulator step."""

import importlib.util
import os
import warnings

import torch

from tensor_budget import is_cuda_oom


class CompiledStage:
    def __init__(self, function, enabled=False, mode='default', dynamic=True):
        self.function, self.enabled = function, enabled
        self.mode, self.dynamic = mode, dynamic
        self.compiled = None
        self.failed = False

    def __call__(self, *args):
        supported = (args and args[0].is_cuda and os.name != 'nt'
                     and importlib.util.find_spec('triton') is not None)
        if not self.enabled or self.failed or not supported:
            return self.function(*args)
        try:
            if self.compiled is None:
                self.compiled = torch.compile(self.function, fullgraph=True, dynamic=self.dynamic, mode=self.mode)
            return self.compiled(*args)
        except Exception as exc:
            if is_cuda_oom(exc):
                raise
            self.failed = True
            self.compiled = None
            warnings.warn(f'compiled stage disabled: {exc}', RuntimeWarning)
            return self.function(*args)


class SingleCudaGraph:
    """At most ONE private graph pool, for one explicitly bounded input bucket.

    Outputs are copied out so a later replay cannot overwrite a saved result.
    Parameters referenced by the callable must be updated in place, not replaced.
    Never used for PPO backward, random sampling, reset, or dynamic compaction.
    """
    def __init__(self, function, enabled=False, capacity=1024, max_private_mb=128,
                 estimated_bytes_per_row=65536):
        self.function, self.enabled = function, bool(enabled)
        self.capacity = max(1, int(capacity))
        self.max_bytes = max(1, int(max_private_mb * 1024**2))
        self.estimated_bytes = self.capacity * int(estimated_bytes_per_row)
        self.graph = self.inputs = self.outputs = self.signature = None
        self.disabled_reason = None
        self.private_reserved_bytes = 0

    def clear(self):
        self.graph = self.outputs = self.inputs = self.signature = None
        self.private_reserved_bytes = 0

    def _invoke(self, *args):
        # Capturing cached fp16 copies would freeze pre-update weights. Casts
        # must be part of the graph so every replay sees optimizer updates.
        with torch.autocast('cuda', enabled=torch.is_autocast_enabled('cuda'), cache_enabled=False):
            return self.function(*args)

    def __call__(self, *args):
        # CUDAGraph's generator bookkeeping is updated outside the caller's
        # inference guard by PyTorch/Inductor. Its long-lived tensors must be
        # ordinary no-grad tensors, not inference tensors.
        if torch.is_inference_mode_enabled():
            with torch.inference_mode(False), torch.no_grad():
                return self._call(*args)
        return self._call(*args)

    def _call(self, *args):
        eligible = (self.enabled and self.disabled_reason is None and args
                    and args[0].is_cuda and not torch.is_grad_enabled()
                    and args[0].shape[0] == self.capacity)
        if not eligible:
            return self.function(*args)
        signature = (tuple((tuple(x.shape), x.dtype, x.device) for x in args),
                     torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
        if self.signature is not None and signature != self.signature:
            # Do not cache/re-capture every shape, precision or device.
            return self.function(*args)
        capture_in_progress = False
        try:
            if self.graph is None:
                free, _ = torch.cuda.mem_get_info(args[0].device)
                if self.estimated_bytes > self.max_bytes or free < 2 * self.max_bytes:
                    self.disabled_reason = 'capture memory budget'
                    return self.function(*args)
                before = torch.cuda.memory_reserved(args[0].device)
                self.inputs = tuple(x.clone() for x in args)
                stream = torch.cuda.Stream(device=args[0].device)
                stream.wait_stream(torch.cuda.current_stream(args[0].device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.outputs = self._invoke(*self.inputs)
                torch.cuda.current_stream(args[0].device).wait_stream(stream)
                # Complete compilation/autotuning outside capture.
                torch.cuda.synchronize(args[0].device)
                self.graph = torch.cuda.CUDAGraph()
                capture_in_progress = True
                with torch.cuda.graph(self.graph, stream=stream):
                    self.outputs = self._invoke(*self.inputs)
                capture_in_progress = False
                self.signature = signature
                self.private_reserved_bytes = max(0, torch.cuda.memory_reserved(args[0].device) - before)
                if self.private_reserved_bytes > self.max_bytes:
                    self.clear()
                    self.disabled_reason = 'measured private-pool budget'
                    return self.function(*args)
            for target, source in zip(self.inputs, args):
                target.copy_(source)
            self.graph.replay()
            if isinstance(self.outputs, tuple):
                return tuple(value.clone() for value in self.outputs)
            return self.outputs.clone()
        except Exception as exc:
            self.clear()
            self.disabled_reason = str(exc)
            if capture_in_progress:
                # A failed CUDA capture can invalidate generator/stream state.
                # Do not silently continue sampling a policy in that process.
                raise RuntimeError('CUDA Graph capture failed; restart with cuda_graph.enabled=false') from exc
            warnings.warn(f'CUDA Graph disabled; pure eager/compiled fallback: {exc}', RuntimeWarning)
        # Safe to retry: the callable is pure, with no RNG/optimizer/world writes.
        return self.function(*args)
