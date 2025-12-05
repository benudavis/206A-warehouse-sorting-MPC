#!/usr/bin/env python3
"""
Master test runner for all kinematics tests.

Runs FK and IK test suites and provides summary.
"""

import subprocess
import sys
from pathlib import Path


def run_test_file(test_file):
    """Run a test file and return (name, passed)."""
    test_name = test_file.stem.replace("test_", "").replace("_", " ").title()
    
    print(f"\n{'='*70}")
    print(f"Running: {test_name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=False
    )
    
    passed = (result.returncode == 0)
    return test_name, passed


if __name__ == "__main__":
    print("\n" + "="*70)
    print("KINEMATICS TEST SUITE - MASTER RUNNER")
    print("="*70)
    
    tests_dir = Path(__file__).parent
    test_files = sorted(tests_dir.glob("test_*.py"))
    
    if not test_files:
        print("No test files found!")
        sys.exit(1)
    
    results = []
    for test_file in test_files:
        test_name, passed = run_test_file(test_file)
        results.append((test_name, passed))
    
    # Final summary
    print("\n" + "="*70)
    print("OVERALL TEST SUMMARY")
    print("="*70)
    print()
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print()
    print(f"Results: {passed_count}/{total_count} test suites passed")
    print("="*70)
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Kinematics implementation is ready.")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED. Review and fix before using in production.")
        sys.exit(1)

