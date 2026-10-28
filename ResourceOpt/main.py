"""
ResourceOpt Entry Point: Terminal Application.
Usage:
    python main.py             # Interactive CLI
    python main.py --all       # Executes complete experiment suite
    python main.py --test      # Runs ResourceOpt unit tests
"""

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LAB_ROOT = PROJECT_DIR.parent
for p in [str(PROJECT_DIR), str(LAB_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ResourceOpt.experiments.run_experiments import run_resource_experiments


def run_tests() -> bool:
    print("\n" + "=" * 75)
    print(" RUNNING RESOURCEOPT UNIT TEST SUITE")
    print("=" * 75)
    loader = unittest.TestLoader()
    suite = loader.discover(str(PROJECT_DIR / "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("--all", "-a"):
            run_resource_experiments()
            return
        elif arg in ("--test", "-t"):
            success = run_tests()
            sys.exit(0 if success else 1)
            
    print("\n" + "=" * 70)
    print("  ResourceOpt: Constrained Production & Resource Allocation")
    print("=" * 70)
    print(" [1] Run Constrained KKT & LICQ Benchmark Experiments")
    print(" [2] Run Automated Unit Tests")
    print(" [0] Exit")
    print("=" * 70)
    
    try:
        choice = input("\nEnter choice [0-2]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting ResourceOpt.")
        return
        
    if choice == "1":
        run_resource_experiments()
    elif choice == "2":
        run_tests()
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()
