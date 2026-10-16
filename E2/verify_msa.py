"""Numerical checks independent of MNIST accuracy; mandatory before E2 training."""
from __future__ import annotations

import torch
from torch import nn

from E2.models import build_model, copy_custom_to_pytorch, tokenization_metadata


class VerificationError(RuntimeError):
    def __init__(self, report: dict):
        self.report = report
        failed = [case for case in report["cases"] if not case["passed"]]
        super().__init__(f"MSA verification failed: {failed}")


def pytorch_name(name: str) -> str:
    if ".attention." not in name:
        return name
    for old, new in (("qkv.weight", "attention.in_proj_weight"),
                     ("qkv.bias", "attention.in_proj_bias"),
                     ("out_proj.weight", "attention.out_proj.weight"),
                     ("out_proj.bias", "attention.out_proj.bias")):
        if name.endswith(".attention." + old):
            return name[:-len(old)] + new
    return name


def verify_pair(tokenizer: str, model_kwargs: dict, device: torch.device,
                dtype: torch.dtype, seed: int) -> dict:
    """Check eval/inference and deterministic train mode; stochastic masks need not match."""
    atol, rtol = (1e-10, 1e-8) if dtype == torch.float64 else (2e-5, 2e-4)
    case = {"tokenization": tokenizer, "device": str(device), "dtype": str(dtype),
            "atol": atol, "rtol": rtol, "passed": False, "max_abs_errors": {}}

    def compare(name, actual, expected):
        if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
            raise AssertionError(f"{name}: non-finite tensor")
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)
        case["max_abs_errors"][name] = float((actual - expected).abs().max().detach().cpu())

    torch.manual_seed(seed)
    a = build_model(f"custom_{tokenizer}", **model_kwargs).to(device=device, dtype=dtype)
    b = build_model(f"pytorch_{tokenizer}", **model_kwargs).to(device=device, dtype=dtype)
    copy_custom_to_pytorch(a, b)
    case["tokens"] = tokenization_metadata(a)
    a.eval()
    b.eval()
    # Check the attention operator directly, including its input/parameter gradients.
    attention_a, attention_b = a.blocks[0].attention, b.blocks[0].attention
    tokens = a.tokenizer.num_tokens + 1
    x = torch.randn(2, tokens, model_kwargs.get("embed_dim", 64), device=device,
                    dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    out_a, out_b = attention_a(x), attention_b(y)
    compare("attention_output", out_a, out_b)
    upstream = torch.randn_like(out_a)
    out_a.backward(upstream)
    out_b.backward(upstream)
    compare("attention_input_gradient", x.grad, y.grad)
    for name, left, right in (
        ("qkv_weight", attention_a.qkv.weight, attention_b.attention.in_proj_weight),
        ("qkv_bias", attention_a.qkv.bias, attention_b.attention.in_proj_bias),
        ("out_weight", attention_a.out_proj.weight, attention_b.attention.out_proj.weight),
        ("out_bias", attention_a.out_proj.bias, attention_b.attention.out_proj.bias),
    ):
        compare("attention_gradient_" + name, left.grad, right.grad)

    images = torch.randn(2, 1, 28, 28, device=device, dtype=dtype)
    # Real inference path: PyTorch can choose a different fast path under no_grad.
    with torch.no_grad():
        compare("classifier_inference_logits", a(images), b(images))

    # Compare the complete classifier's training graph without stochastic dropout masks.
    for model in (a, b):
        model.train()
        model.zero_grad(set_to_none=True)
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            elif isinstance(module, nn.MultiheadAttention):
                module.dropout = 0.0
    x = images.detach().clone().requires_grad_(True)
    y = images.detach().clone().requires_grad_(True)
    out_a, out_b = a(x), b(y)
    compare("classifier_train_logits", out_a, out_b)
    labels = torch.tensor([1, 7], device=device)
    loss_a, loss_b = nn.functional.cross_entropy(out_a, labels), nn.functional.cross_entropy(out_b, labels)
    compare("classifier_loss", loss_a, loss_b)
    loss_a.backward()
    loss_b.backward()
    compare("classifier_input_gradient", x.grad, y.grad)
    parameters_b = dict(b.named_parameters())
    gradient_error = 0.0
    gradient_count = 0
    for name, parameter_a in a.named_parameters():
        parameter_b = parameters_b[pytorch_name(name)]
        if parameter_a.grad is None or parameter_b.grad is None:
            raise AssertionError(f"Missing gradient: {name}")
        torch.testing.assert_close(parameter_a.grad, parameter_b.grad, atol=atol, rtol=rtol, msg=name)
        gradient_error = max(gradient_error, float((parameter_a.grad - parameter_b.grad).abs().max().cpu()))
        gradient_count += 1
    case["max_abs_errors"]["all_parameter_gradients"] = gradient_error
    case["parameter_tensors_checked"] = gradient_count

    # In float64 also verify a real optimizer update. Near-zero gradients can amplify
    # float32 rounding through Adam's epsilon; float32 checks target outputs/gradients.
    if dtype == torch.float64:
        for model in (a, b):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
            optimizer.step()
        parameters_b = dict(b.named_parameters())
        for name, parameter_a in a.named_parameters():
            torch.testing.assert_close(parameter_a, parameters_b[pytorch_name(name)],
                                       atol=atol, rtol=rtol, msg="Adam update: " + name)
        with torch.no_grad():
            compare("classifier_logits_after_adam_step", a(images), b(images))
    case["passed"] = True
    return case


def verify_experiments(tokenizers: list[str], model_kwargs: dict, device: torch.device,
                       seed: int = 42) -> dict:
    """CPU float64 reference plus float32 on the actual training device, for EVERY pair."""
    tokenizers = list(dict.fromkeys(tokenizers))
    if not tokenizers:
        raise ValueError("Select at least one tokenizer for verification.")
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    report = {"schema_version": 1, "seed": seed, "model_kwargs": model_kwargs,
              "torch_version": str(torch.__version__), "cases": [], "passed": False,
              "scope": "Same weights; eval and train with dropout disabled; outputs, loss, input and parameter gradients.",
              "limitation": "Stochastic training metrics need not be identical; this is not a proof for every possible input."}
    # Preflight must not perturb the RNG used by real experiments.
    with torch.random.fork_rng(devices=devices):
        for tokenizer in tokenizers:
            for target, dtype in ((torch.device("cpu"), torch.float64), (device, torch.float32)):
                try:
                    case = verify_pair(tokenizer, model_kwargs, target, dtype, seed)
                except (AssertionError, ValueError, RuntimeError) as error:
                    case = {"tokenization": tokenizer, "device": str(target), "dtype": str(dtype),
                            "passed": False, "error": f"{type(error).__name__}: {error}"}
                report["cases"].append(case)
                print(f"MSA check | {tokenizer} | {target} | {dtype} | "
                      f"{'PASS' if case['passed'] else 'FAIL'}", flush=True)
    report["passed"] = all(case["passed"] for case in report["cases"])
    if not report["passed"]:
        raise VerificationError(report)
    return report
