"""
Run all Le Masson 2002 replication experiments.

Usage:
  python run_all.py              # Quick test (10s simulations)
  python run_all.py --full       # Full run (60s simulations)
  python run_all.py --exp 1      # Run only experiment 1
  python run_all.py --exp 4      # Run only bifurcation diagram
"""

import sys
import os
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_quick_test():
    """Quick smoke test: single short simulation to verify circuit works."""
    print("=" * 60)
    print("QUICK SMOKE TEST")
    print("=" * 60)

    from circuit.thalamic_circuit import ThalamicCircuit
    from analysis.oscillation import is_oscillating, oscillation_power
    from analysis.plotting import plot_voltage_traces

    # Test 1: No inhibition (relay mode)
    print("\nTest 1: No inhibition (GABA = 0 nS)")
    circuit = ThalamicCircuit(gaba_gmax_total=0, retinal_rate=42.0,
                              gamma_order=1.5, seed=42)
    sim = circuit.simulate(2.0, record_dt=0.0005)
    osc = is_oscillating(sim['tc_spike_times'], sim['duration_s'])
    print(f"  TC spikes: {len(sim['tc_spike_times'])}")
    print(f"  nRt spikes: {len(sim['nrt_spike_times'])}")
    print(f"  Oscillating: {osc} (expected: False)")
    plot_voltage_traces(sim, duration_s=2.0, title='Relay Mode (GABA=0)',
                        save_name='test_relay_mode.png')

    # Test 2: High inhibition (oscillation mode)
    print("\nTest 2: High inhibition (GABA = 50 nS)")
    circuit = ThalamicCircuit(gaba_gmax_total=50, retinal_rate=42.0,
                              gamma_order=1.5, seed=42)
    sim = circuit.simulate(2.0, record_dt=0.0005)
    osc = is_oscillating(sim['tc_spike_times'], sim['duration_s'])
    power = oscillation_power(sim['V_tc'], sim['t'])
    print(f"  TC spikes: {len(sim['tc_spike_times'])}")
    print(f"  nRt spikes: {len(sim['nrt_spike_times'])}")
    print(f"  Oscillating: {osc} (expected: True)")
    print(f"  Oscillation power: {power:.2e}")
    plot_voltage_traces(sim, duration_s=2.0, title='Oscillation Mode (GABA=50)',
                        save_name='test_oscillation_mode.png')

    # Test 3: TC neuron properties
    print("\nTest 3: TC neuron properties")
    from models.tc_neuron import TCNeuron
    tc = TCNeuron()
    r_in = tc.input_resistance_MOhm()
    print(f"  Input resistance: {r_in:.1f} MOhm (target: 68.1 +/- 3.1)")

    print("\nSmoke test complete. Figures saved to figures/")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Le Masson 2002 Replication Experiments')
    parser.add_argument('--full', action='store_true',
                        help='Run full-duration simulations (60s each)')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 4],
                        help='Run only a specific experiment')
    parser.add_argument('--test', action='store_true',
                        help='Run quick smoke test only')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    duration = 60.0 if args.full else 10.0

    start_time = time.time()

    if args.test:
        run_quick_test()
        return

    if args.exp is None or args.exp == 0:
        # Run smoke test first
        run_quick_test()

    # Run experiments
    if args.exp is None or args.exp == 1:
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Spindle Wave Generation")
        print("=" * 60)
        from experiments.exp1_spindle import run_experiment as run_exp1
        run_exp1(duration_s=duration, seed=args.seed)

    if args.exp is None or args.exp == 2:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Input-Output Correlation")
        print("=" * 60)
        from experiments.exp2_correlation import run_experiment as run_exp2
        run_exp2(duration_s=duration, seed=args.seed)

    if args.exp is None or args.exp == 3:
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Noradrenaline Modulation")
        print("=" * 60)
        from experiments.exp3_noradrenaline import run_experiment as run_exp3
        run_exp3(duration_s=duration, seed=args.seed)

    if args.exp is None or args.exp == 4:
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: BIFURCATION DIAGRAM")
        print("=" * 60)
        from experiments.exp4_bifurcation import run_experiment as run_exp4
        results, threshold = run_exp4(duration_s=duration, seed=args.seed)

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f} seconds")
    print("All figures saved to le_masson_replication/figures/")


if __name__ == '__main__':
    main()
