"""A1-specific training; shared E1-E3 training code is deliberately independent."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from A1.models import set_a1_train_mode


class NonFiniteTrainingError(RuntimeError):
    pass


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device):
    if device.type != "cuda":
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }


def run_epoch(model, loader, device, optimizer=None, show_progress=False):
    training = optimizer is not None
    if training:
        set_a1_train_mode(model)
    else:
        model.eval()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    count = correct = batches = 0
    loss_sum = 0.0
    parameters = [p for p in model.parameters() if p.requires_grad]
    phase = "train" if training else "validation"
    for step, (inputs, targets) in enumerate(tqdm(loader, desc=phase, leave=False, disable=not show_progress), 1):
        inputs, targets = inputs.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss).item():
                raise NonFiniteTrainingError(f"{phase}: loss is not finite at batch {step}")
            if training:
                loss.backward()
                try:
                    # Check the aggregate norm, without changing finite gradients.
                    torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
                except RuntimeError as exc:
                    raise NonFiniteTrainingError(f"train: gradient norm is not finite at batch {step}") from exc
                optimizer.step()
        size = targets.numel()
        count += size
        correct += int((logits.argmax(1) == targets).sum().item())
        loss_sum += float(loss.detach().item()) * size
        batches += 1
    synchronize(device)
    seconds = time.perf_counter() - start
    if not count:
        raise ValueError(f"Empty {phase} loader")
    return {"loss": loss_sum / count, "accuracy": correct / count,
            "seconds": seconds, "samples": count, "batches": batches,
            "images_per_second": count / seconds, **memory_snapshot(device)}


@dataclass
class A1TrainResult:
    history: list
    best_epoch: int
    best_val_accuracy: float
    parameter_count: int
    training_time_seconds: float
    phase_measurements: list


def fit_classifier(model, train_loader, val_loader, epochs, learning_rate,
                   weight_decay, device, show_progress=False, on_epoch=None, on_phase=None):
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    history, phases = [], []
    best_accuracy, best_epoch, best_state = -1.0, 0, None
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        training = run_epoch(model, train_loader, device, optimizer, show_progress)
        if on_phase is not None:
            on_phase(epoch, "train", training)
        validation = run_epoch(model, val_loader, device, show_progress=show_progress)
        if on_phase is not None:
            on_phase(epoch, "validation", validation)
        row = {"epoch": epoch, "train_loss": training["loss"], "train_accuracy": training["accuracy"],
               "val_loss": validation["loss"], "val_accuracy": validation["accuracy"]}
        history.append(row)
        phases.append({"epoch": epoch, "train": training, "validation": validation})
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch {epoch}/{epochs} | train {training['loss']:.4f}, {training['accuracy']:.4f} | "
              f"val {validation['loss']:.4f}, {validation['accuracy']:.4f} | "
              f"train {training['seconds']:.1f}s, val {validation['seconds']:.1f}s", flush=True)
        if on_epoch is not None:
            on_epoch(row, phases[-1])
    model.load_state_dict(best_state)
    synchronize(device)
    elapsed = time.perf_counter() - start
    return A1TrainResult(history, best_epoch, best_accuracy, sum(p.numel() for p in parameters), elapsed, phases)
