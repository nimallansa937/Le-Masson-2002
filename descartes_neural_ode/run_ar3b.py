"""
A-R3b: GRU-ODE with Auxiliary Biological Loss
================================================

Main experiment script. Sweeps the alpha parameter controlling the
trade-off between spike prediction loss and biological encoding loss:

  alpha=1.0  -> spike-only (A-R3 baseline reproduction)
  alpha=0.7  -> spike-dominated
  alpha=0.5  -> equal weight
  alpha=0.3  -> bio-dominated
  alpha=0.0  -> bio-only (extreme control)

Each condition trains for 2 hours (configurable) using the same
progressive curriculum as DESCARTES A-R3, then runs the full evaluation
suite: spike correlation, Ridge R-squared, CKA, MI, gate analysis.

Results answer the central question: can explicit biological supervision
achieve BOTH correct spike prediction AND genuine biological encoding?
If yes, biology is compatible with function but requires supervision.
If no, there is a fundamental trade-off (the zombie IS optimal).

Usage:
  # Full sweep (5 conditions x 2h = 10h)
  python run_ar3b.py --data_dir /root/rung3_data --device cuda

  # Quick validation (6min, single alpha)
  python run_ar3b.py --data_dir /root/rung3_data --max_hours 0.1 --alpha 0.5

  # Local CPU test with short budget
  python run_ar3b.py --data_dir C:/rung3_temp --device cpu --max_hours 0.05
"""
import argparse
import json
import sys
import os
import time
import torch
import numpy as np
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from data.bio_data_loader import load_bio_aligned_data
from architectures.gru_ode_bio import GRUODEBio
from training.bio_loss import CombinedBioLoss
from training.train_bio import train_gru_ode_bio
from evaluation.bio_evaluation import (
    evaluate_bio_recovery, print_evaluation_summary
)


BANNER = r"""
 ___  ____  ____   ___    __   ____  ____  ____  ____
(  _\( ___)(  _ \ / __)  /__\ (  _ \(_  _)( ___)/ ___)
 )(_) )__)  \__ \( (__  /(  )\ )   /  )(   )__) \__ \
(____/(____)(___/ \___)(__)(__/(_)\_) (__) (____)(___/

  A-R3b: GRU-ODE with Auxiliary Biological Loss
  ─────────────────────────────────────────────
  Can explicit bio supervision achieve both
  spike prediction AND biological encoding?
"""


def main():
    parser = argparse.ArgumentParser(
        description='A-R3b: GRU-ODE with Auxiliary Biological Loss',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full sweep:    python run_ar3b.py --data_dir /root/rung3_data --device cuda
  Quick test:    python run_ar3b.py --data_dir /root/rung3_data --max_hours 0.1 --alpha 0.5
  Custom alphas: python run_ar3b.py --alpha 1.0 0.5 0.0 --max_hours 1.0
        """
    )
    parser.add_argument('--data_dir', type=str, default='/root/rung3_data',
                        help='Directory containing trial_gaba*.h5 files')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--max_hours', type=float, default=2.0,
                        help='Training budget per alpha value (hours)')
    parser.add_argument('--output_dir', type=str, default='./ar3b_results',
                        help='Output directory for results and checkpoints')
    parser.add_argument('--alpha', type=float, nargs='+',
                        default=[1.0, 0.7, 0.5, 0.3, 0.0],
                        help='Alpha values to sweep (1.0=spike-only, 0.0=bio-only)')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='GRU-ODE latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='GRU-ODE hidden dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate')
    parser.add_argument('--lr_patience', type=int, default=20,
                        help='Epochs before LR reduction')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum learning rate')
    parser.add_argument('--early_stop_patience', type=int, default=40,
                        help='Epochs before early stopping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (same across all alpha for fair comparison)')
    args = parser.parse_args()

    print(BANNER)
    print(f"  Device:          {args.device}")
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Alpha sweep:     {args.alpha}")
    print(f"  Latent dim:      {args.latent_dim}")
    print(f"  Budget/alpha:    {args.max_hours}h")
    print(f"  Total est. time: {args.max_hours * len(args.alpha):.1f}h")
    print(f"  Seed:            {args.seed}")
    print()

    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("  WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # Load data
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  LOADING DATA")
    print("="*70)

    # Load data with per-trial bio alignment (the critical fix)
    # Each training window gets bio targets from its OWN source trial
    # at its OWN temporal offset — not from a shared canonical trial
    train_loader, val_loader, bio_dims, bio_gt, data_info = \
        load_bio_aligned_data(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=0,
        )

    total_bio_vars = sum(bio_dims.values())
    print(f"\n  Bio GT (legacy): {len(bio_gt)} variable groups")

    # Save experiment config
    config = {
        'alpha_sweep': args.alpha,
        'latent_dim': args.latent_dim,
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lr_patience': args.lr_patience,
        'min_lr': args.min_lr,
        'early_stop_patience': args.early_stop_patience,
        'max_hours_per_alpha': args.max_hours,
        'seed': args.seed,
        'device': args.device,
        'bio_dims': bio_dims,
        'train_windows': data_info['n_train'],
        'val_windows': data_info['n_val'],
        'input_dim': data_info['input_dim'],
        'output_dim': data_info['output_dim'],
    }
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ─────────────────────────────────────────────────────────────────
    # Alpha sweep
    # ─────────────────────────────────────────────────────────────────
    all_results = {}
    sweep_start = time.time()

    for alpha_idx, alpha in enumerate(args.alpha):
        print(f"\n{'='*70}")
        print(f"  ALPHA = {alpha:.2f} ({alpha_idx+1}/{len(args.alpha)})")
        condition_name = (
            'Spike-only baseline (A-R3 reproduction)' if alpha >= 1.0
            else 'Bio-only extreme (no spike objective)' if alpha <= 0.0
            else f'Mixed: spike={alpha:.0%}, bio={1-alpha:.0%}'
        )
        print(f"  Condition: {condition_name}")
        print(f"{'='*70}")

        # === Create fresh model ===
        # For alpha=1.0 (spike-only), still create bio heads for
        # post-hoc evaluation — but loss ignores them during training
        model = GRUODEBio(
            input_dim=data_info['input_dim'],
            output_dim=data_info['output_dim'],
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            bio_dims=bio_dims,  # Always include for consistent evaluation
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}")

        # === Create loss function ===
        loss_fn = CombinedBioLoss(
            alpha=alpha,
            category_weights={
                'tc_gating': 1.0,    # Fast gating variables
                'nrt_state': 1.0,    # Mixed timescale state
                'synaptic': 1.0,     # Slow synaptic variables
            },
            temporal_smooth_ms={
                'tc_gating': 0,      # No smoothing for fast gating
                'nrt_state': 5,      # Light smoothing for mixed
                'synaptic': 20,      # Heavy smoothing for slow synaptic
            },
        )

        # === Train ===
        alpha_dir = output_dir / f'alpha_{alpha:.2f}'
        alpha_dir.mkdir(parents=True, exist_ok=True)

        train_results = train_gru_ode_bio(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            alpha=alpha,
            device=args.device,
            total_budget_hours=args.max_hours,
            output_dir=alpha_dir,
            lr=args.lr,
            lr_patience=args.lr_patience,
            min_lr=args.min_lr,
            early_stop_patience=args.early_stop_patience,
            seed=args.seed,
        )

        # === Evaluate ===
        # bio_gt is legacy fallback — the val_loader now provides properly
        # aligned per-window bio targets (the critical alignment fix)
        print("\n  Running evaluation suite...")
        eval_results = evaluate_bio_recovery(
            model, val_loader,
            bio_ground_truth=bio_gt,  # Legacy fallback only
            device=args.device,
        )

        # Print summary
        print_evaluation_summary(eval_results, alpha)

        # Store results
        all_results[f'alpha_{alpha:.2f}'] = {
            'train': train_results,
            'eval': {
                'spike_corr': eval_results['spike_corr'],
                'ridge': eval_results['ridge'],
                'cka': eval_results['cka'],
                'mi': eval_results.get('mi', {}),
                'gates': eval_results.get('gates', {}),
                'pearson': eval_results.get('pearson', {}),
            }
        }

        # Save intermediate results (in case of crash)
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # Save per-alpha detailed results
        with open(alpha_dir / 'eval_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)

    # ─────────────────────────────────────────────────────────────────
    # Final comparison table
    # ─────────────────────────────────────────────────────────────────
    total_time_h = (time.time() - sweep_start) / 3600

    print(f"\n\n{'='*90}")
    print("  A-R3b FINAL COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"  {'Alpha':<8} {'Spike':>8} {'Ridge R2':>10} {'Decode>0.25':>13} "
          f"{'Decode>0.50':>13} {'CKA':>8} {'Dedicated':>10}")
    print(f"  {'-'*80}")

    for alpha_key in sorted(all_results.keys()):
        r = all_results[alpha_key]
        e = r['eval']
        alpha_val = float(alpha_key.split('_')[1])
        ded = e.get('gates', {}).get('dedicated_dims', '?')
        total_vars = e['ridge'].get('total', 160)

        print(f"  {alpha_val:<8.2f} {e['spike_corr']:>8.4f} "
              f"{e['ridge']['mean_r2']:>10.4f} "
              f"{e['ridge']['decodable_025']:>9}/{total_vars:<3}  "
              f"{e['ridge']['decodable_050']:>9}/{total_vars:<3}  "
              f"{e['cka']:>8.4f} {str(ded):>6}/32")

    print(f"{'='*90}")
    print(f"\n  Total experiment time: {total_time_h:.2f}h")

    # ─────────────────────────────────────────────────────────────────
    # Interpretation
    # ─────────────────────────────────────────────────────────────────
    if len(all_results) >= 3:
        # Check which scenario we're in
        spike_only = all_results.get('alpha_1.00', {}).get('eval', {})
        mid_alpha = None
        for k in ['alpha_0.50', 'alpha_0.30', 'alpha_0.70']:
            if k in all_results:
                mid_alpha = all_results[k]['eval']
                break

        if spike_only and mid_alpha:
            spike_only_r2 = spike_only.get('ridge', {}).get('mean_r2', 0)
            mid_r2 = mid_alpha.get('ridge', {}).get('mean_r2', 0)
            spike_only_corr = spike_only.get('spike_corr', 0)
            mid_corr = mid_alpha.get('spike_corr', 0)

            print(f"\n  INTERPRETATION:")
            if mid_r2 > 0.25 and mid_corr > 0.3:
                print(f"  -> Scenario A: Bio loss achieves BOTH! (R2={mid_r2:.3f}, corr={mid_corr:.3f})")
                print(f"     Biology is compatible with function but requires supervision.")
            elif mid_r2 > 0.1 and mid_corr < spike_only_corr * 0.5:
                print(f"  -> Scenario B: Fundamental trade-off. Bio comes at spike cost.")
                print(f"     The zombie IS the optimal solution for spike prediction.")
            elif mid_r2 > 0.25 and mid_corr >= spike_only_corr * 0.9:
                print(f"  -> Scenario C: Bio loss is SUFFICIENT! Bio helps spikes!")
                print(f"     Biology is the natural encoding gradient descent couldn't find.")
            else:
                print(f"  -> Results don't clearly match expected scenarios.")
                print(f"     Bio R2={mid_r2:.3f}, spike={mid_corr:.3f} vs baseline spike={spike_only_corr:.3f}")

    # Save final results
    final_output = {
        'results': all_results,
        'config': config,
        'total_time_hours': total_time_h,
    }
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_output, f, indent=2, default=str)

    print(f"\n  All results saved to {output_dir}/")
    print(f"  Key files:")
    print(f"    final_results.json  — Complete results with config")
    print(f"    all_results.json    — Per-alpha results")
    for alpha in args.alpha:
        print(f"    alpha_{alpha:.2f}/          — Checkpoints + training log")


if __name__ == '__main__':
    main()
