"""
Optimization-Laboratory Master Runner.
Executes the full automated test suite (77 tests across 6 modules)
and executes end-to-end experiment pipelines across all 5 independent projects:
1. EnergyOpt
2. PredictMaint
3. NonlinearLab
4. SparseModel
5. ResourceOpt
"""

import sys
import unittest
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

# Project experiment entry points
from EnergyOpt.experiments.run_experiments import run_energy_experiments
from PredictMaint.experiments.run_experiments import run_maintenance_experiments
from NonlinearLab.experiments.run_experiments import run_nonlinear_experiments
from SparseModel.experiments.run_experiments import run_sparse_experiments
from ResourceOpt.experiments.run_experiments import run_resource_experiments


def run_complete_test_suite() -> bool:
    """Discovers and runs unit tests across shared_optimization and all 5 projects."""
    print("\n" + "=" * 80)
    print(" OPTIMIZATION-LABORATORY: RUNNING COMPLETE AUTOMATED UNIT TEST SUITE")
    print("=" * 80)
    
    test_folders = [
        ("Shared Optimization Core & Algorithms", LAB_ROOT / "shared_optimization" / "tests"),
        ("Project 1: EnergyOpt", LAB_ROOT / "EnergyOpt" / "tests"),
        ("Project 2: PredictMaint", LAB_ROOT / "PredictMaint" / "tests"),
        ("Project 3: NonlinearLab", LAB_ROOT / "NonlinearLab" / "tests"),
        ("Project 4: SparseModel", LAB_ROOT / "SparseModel" / "tests"),
        ("Project 5: ResourceOpt", LAB_ROOT / "ResourceOpt" / "tests"),
    ]
    
    total_tests = 0
    all_passed = True
    
    loader = unittest.TestLoader()
    for name, path in test_folders:
        print(f"\n--- Testing: {name} ---")
        suite = loader.discover(str(path))
        num_tests = suite.countTestCases()
        total_tests += num_tests
        runner = unittest.TextTestRunner(verbosity=1)
        res = runner.run(suite)
        if not res.wasSuccessful():
            all_passed = False
            
    print("\n" + "=" * 80)
    print(f" TEST SUITE SUMMARY: {total_tests} TOTAL TESTS RUN | ALL PASSED: {all_passed}")
    print("=" * 80)
    return all_passed


def run_all_projects() -> None:
    """Executes all 5 end-to-end problem experiments."""
    t_start = time.perf_counter()
    print("\n" + "=" * 80)
    print(" OPTIMIZATION-LABORATORY: RUNNING ALL 5 REAL-WORLD EXPERIMENTS")
    print("=" * 80)
    
    run_energy_experiments()
    run_maintenance_experiments()
    run_nonlinear_experiments()
    run_sparse_experiments()
    run_resource_experiments()
    
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f" ALL 5 EXPERIMENTS COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print(" All generated plots, numerical histories, and reports saved to respective results/ directories.")
    print("=" * 80)


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("--all", "-a"):
            run_all_projects()
            return
        elif arg in ("--test", "-t"):
            success = run_complete_test_suite()
            sys.exit(0 if success else 1)
            
    print("\n" + "=" * 75)
    print("         Optimization-Laboratory: Master Laboratory Runner")
    print("=" * 75)
    print(" [1] Run Complete Automated Test Suite (77 unit tests)")
    print(" [2] Run All 5 Real-World Optimization Experiments")
    print(" [3] Run Both (Full Verification & Complete Experiments)")
    print(" [0] Exit")
    print("=" * 75)
    
    try:
        choice = input("\nEnter choice [0-3]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return
        
    if choice == "1":
        run_complete_test_suite()
    elif choice == "2":
        run_all_projects()
    elif choice == "3":
        ok = run_complete_test_suite()
        if ok:
            run_all_projects()
        else:
            print("Tests failed; halting.")
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()
