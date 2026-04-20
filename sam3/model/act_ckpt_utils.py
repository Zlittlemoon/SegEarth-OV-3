# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import inspect
import os
from functools import wraps
from typing import Callable, TypeVar, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.utils._pytree import tree_map_only

# Type variables for better type hinting
T = TypeVar("T")
Module = TypeVar("Module", bound=nn.Module)


def activation_ckpt_wrapper(module: Union[nn.Module, Callable]) -> Callable:
    """
    Wraps a given module to enable or disable activation checkpointing.

    Activation checkpointing (gradient checkpointing) trades compute for memory by
    recomputing intermediate activations during the backward pass instead of storing
    them in memory during the forward pass.

    When activation checkpointing is enabled, the wrapper expects only keyword arguments,
    and it maps these to positional arguments based on the module's signature.

    Args:
        module: The module or function to wrap with activation checkpointing

    Returns:
        A wrapped callable that supports activation checkpointing

    Usage:
        The returned wrapper function can be called with the same arguments as the
        original module, with an additional `act_ckpt_enable` keyword argument to control
        activation checkpointing and optional `use_reentrant` parameter.

    Example:
        ```python
        wrapped_module = activation_ckpt_wrapper(my_module)
        output = wrapped_module(x=input_tensor, y=another_tensor, act_ckpt_enable=True)
        ```
    """

    @wraps(module)
    def act_ckpt_wrapper(
        *args, act_ckpt_enable: bool = True, use_reentrant: bool = False, **kwargs
    ):
        if act_ckpt_enable:
            # Runtime overrides for checkpoint stability.
            # Default to reentrant mode to avoid non-reentrant recompute mismatch
            # errors on some SDPA/MHA paths under AMP+DDP.
            env_use_reentrant = os.getenv("SAM3_ACT_CKPT_USE_REENTRANT", "1").strip()
            if env_use_reentrant in {"1", "true", "True"}:
                use_reentrant = True
            elif env_use_reentrant in {"0", "false", "False"}:
                use_reentrant = False

            if len(args) > 0:
                raise ValueError(
                    "This wrapper expects keyword arguments only when `act_ckpt_enable=True`"
                )
            # Get the signature of the target function/module
            callable_fn = module.forward if isinstance(module, nn.Module) else module
            sig = inspect.signature(callable_fn)
            # Create a mapping of parameter names to their default values
            param_defaults = {
                name: param.default for name, param in sig.parameters.items()
            }
            args = []
            for p_name in param_defaults.keys():
                if p_name in kwargs:
                    args.append(kwargs.pop(p_name))
                elif param_defaults[p_name] is not inspect.Parameter.empty:
                    # Set arg to default value if it's not in kwargs. Useful for primitive types or args that default to None
                    args.append(param_defaults[p_name])
                elif (
                    sig.parameters[p_name].kind is not inspect.Parameter.VAR_KEYWORD
                ):  # Skip **kwargs parameter
                    raise ValueError(f"Missing positional argument: {p_name}")

            # Always pass positional tensors to checkpoint API, and forward any
            # leftover kwargs through a closure. This avoids reentrant checkpoint
            # rejecting unexpected keyword arguments.
            def _run_module(*inner_args):
                return module(*inner_args, **kwargs)

            # Reentrant checkpoint requires at least one input tensor with
            # requires_grad=True; otherwise outputs can become non-differentiable
            # even when module parameters are trainable.
            if use_reentrant:
                has_grad_input = any(
                    isinstance(x, torch.Tensor) and x.requires_grad for x in args
                )
                if not has_grad_input:
                    return _run_module(*args)

            # PyTorch non-reentrant checkpoint may hit `early_stop` assertions or
            # recompute mismatch errors in some nested checkpoint/autocast/DDP paths.
            # Keep a compatibility path for non-reentrant, but prefer reentrant above.
            if not use_reentrant and hasattr(checkpoint, "set_checkpoint_early_stop"):
                with checkpoint.set_checkpoint_early_stop(False):
                    if "determinism_check" in inspect.signature(checkpoint.checkpoint).parameters:
                        ret = checkpoint.checkpoint(
                            _run_module,
                            *args,
                            use_reentrant=use_reentrant,
                            determinism_check="none",
                        )
                    else:
                        ret = checkpoint.checkpoint(
                            _run_module, *args, use_reentrant=use_reentrant
                        )
            else:
                ret = checkpoint.checkpoint(
                    _run_module, *args, use_reentrant=use_reentrant
                )
        else:
            ret = module(*args, **kwargs)

        return ret

    return act_ckpt_wrapper


def clone_output_wrapper(f: Callable[..., T]) -> Callable[..., T]:
    """
    Clone the CUDA output tensors of a function to avoid in-place operations.

    This wrapper is useful when working with torch.compile to prevent errors
    related to in-place operations on tensors.

    Args:
        f: The function whose CUDA tensor outputs should be cloned

    Returns:
        A wrapped function that clones any CUDA tensor outputs
    """

    @wraps(f)
    def wrapped(*args, **kwargs):
        outputs = f(*args, **kwargs)
        return tree_map_only(
            torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
        )

    return wrapped
