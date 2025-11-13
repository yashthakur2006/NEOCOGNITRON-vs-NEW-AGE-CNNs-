import argparse
import os
from typing import List, Dict

import torch

from .utils import set_seed, get_device
from .models import NeoCognitronLike, TinyCNN, count_parameters
from .data import get_mnist_loaders, ShiftedMNIST
from .training import train_one_epoch, evaluate_accuracy, invariance_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neocognitron-like vs Tiny CNN on MNIST (shift invariance experiment)."
    )

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for AdamW."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "cuda"],
        help="Force device (cpu/cuda). If None, choose automatically.",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="List of max_shift values (pixels) for shifted evaluations.",
    )
    parser.add_argument(
        "--invariance-samples",
        type=int,
        default=2000,
        help="Number of test samples for invariance score.",
    )
    parser.add_argument(
        "--invariance-transforms",
        type=int,
        default=5,
        help="Number of random shifts per image for invariance score.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints.",
    )

    return parser.parse_args()


def train_and_evaluate_model(
    model_name: str,
    model: torch.nn.Module,
    device: torch.device,
    train_loader,
    test_loader,
    test_dataset,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """
    Train a single model and evaluate clean + shifted + invariance metrics.
    Returns a dict of results.
    """
    print("=" * 80)
    print(f"Training model: {model_name}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print("=" * 80)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        evaluate_accuracy(
            model=model,
            loader=test_loader,
            device=device,
            desc=f"{model_name} clean test",
        )

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Checkpoint] Saved {model_name} to {ckpt_path}")

    # Final evaluations
    results: Dict[str, float] = {}

    # Clean accuracy
    clean_acc = evaluate_accuracy(
        model=model,
        loader=test_loader,
        device=device,
        desc=f"{model_name} clean test (final)",
    )
    results["clean_acc"] = clean_acc

    # Shifted accuracy + invariance scores
    for max_shift in args.shifts:
        shifted_ds = ShiftedMNIST(test_dataset, max_shift=max_shift)
        shifted_loader = torch.utils.data.DataLoader(
            shifted_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        acc_shifted = evaluate_accuracy(
            model=model,
            loader=shifted_loader,
            device=device,
            desc=f"{model_name} shifted +/-{max_shift}",
        )
        inv_score = invariance_score(
            model=model,
            base_dataset=test_dataset,
            device=device,
            max_shift=max_shift,
            num_transforms=args.invariance_transforms,
            num_samples=args.invariance_samples,
        )

        results[f"shifted_acc_k{max_shift}"] = acc_shifted
        results[f"invariance_k{max_shift}"] = inv_score

    return results


def main() -> None:
    args = parse_args()

    # Seed + device
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[Setup] Using device: {device}")

    # Data
    train_loader, test_loader, test_dataset = get_mnist_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Models to compare
    experiments: List[torch.nn.Module] = [
        NeoCognitronLike(num_classes=10),
        TinyCNN(num_classes=10),
    ]
    names = ["neocognitron_like", "tiny_cnn"]

    all_results: Dict[str, Dict[str, float]] = {}

    for name, model in zip(names, experiments):
        results = train_and_evaluate_model(
            model_name=name,
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            test_dataset=test_dataset,
            args=args,
        )
        all_results[name] = results

    print("\n" + "#" * 80)
    print("Summary of results:")
    print("#" * 80)
    for name, res in all_results.items():
        print(f"\nModel: {name}")
        print(f"  Clean accuracy: {res['clean_acc'] * 100:.2f}%")
        for k in args.shifts:
            acc_k = res.get(f"shifted_acc_k{k}", None)
            inv_k = res.get(f"invariance_k{k}", None)
            if acc_k is not None:
                print(f"  Shifted +/-{k}: acc={acc_k*100:.2f}%, invariance={inv_k:.4f}")


if __name__ == "__main__":
    main()
