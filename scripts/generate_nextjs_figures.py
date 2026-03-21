#!/usr/bin/env python
"""
Unified pipeline: generate subplot PNGs → compose in Next.js → screenshot.

This is the NEW top-level entry point that:
  1. Generates individual subplot PNGs via matplotlib (generate_subplots.py)
  2. Sets up public/subplots symlinks for Next.js
  3. Starts the Next.js dev server (if not already running)
  4. Screenshots each figure via Playwright (screenshot_figures.mjs)

Usage:
    python scripts/generate_nextjs_figures.py --series dpmm --figures 2 3 4 5 6 7 8 9 10
    python scripts/generate_nextjs_figures.py --series dpmm --figures all
    python scripts/generate_nextjs_figures.py --series dpmm --skip-subplots  # reuse existing

Steps can be run individually:
    # Step 1: Generate subplots only
    python -m benchmarks.figure_generators.generate_subplots --series dpmm --figures 1 2 3 4 5 6 7 8 9 10

    # Step 2: Setup symlinks
    bash model-arch-viewer/scripts/setup_subplots_link.sh

    # Step 3: Start Next.js (in another terminal)
    cd model-arch-viewer && npm run dev

    # Step 4: Screenshot
    cd model-arch-viewer && node screenshot_figures.mjs --series dpmm --figures 1 2 3 4 5 6 7 8 9 10
"""

import argparse
import os
import sys
import subprocess
import time
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = ROOT / "model-arch-viewer"
PORT = 3099


def check_server_running():
    """Check if Next.js dev server is running on the expected port."""
    import socket
    try:
        with socket.create_connection(("localhost", PORT), timeout=2):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def run_subplot_generation(series, figures):
    """Step 1: Generate individual subplot PNGs."""
    print(f"\n{'='*60}")
    print(f"  Step 1: Generating subplot PNGs ({series})")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m",
        "benchmarks.figure_generators.generate_subplots",
        "--series", series,
        "--figures",
    ] + [str(f) for f in figures]

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  ERROR: Subplot generation failed (exit {result.returncode})")
        return False
    return True


def setup_symlinks():
    """Step 2: Create public/subplots symlinks."""
    print(f"\n{'='*60}")
    print(f"  Step 2: Setting up symlinks")
    print(f"{'='*60}")

    script = VIEWER_DIR / "scripts" / "setup_subplots_link.sh"
    result = subprocess.run(["bash", str(script)], cwd=str(ROOT))
    return result.returncode == 0


def start_dev_server():
    """Step 3: Start Next.js dev server if not running."""
    if check_server_running():
        print(f"  Next.js dev server already running on port {PORT}")
        return None

    print(f"\n{'='*60}")
    print(f"  Step 3: Starting Next.js dev server (port {PORT})")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["PORT"] = str(PORT)
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(VIEWER_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    # Wait for server to be ready
    for _ in range(60):
        time.sleep(1)
        if check_server_running():
            print(f"  Dev server ready on port {PORT}")
            return proc
        if proc.poll() is not None:
            print("  ERROR: Dev server exited unexpectedly")
            return None

    print("  ERROR: Dev server did not start within 60s")
    proc.terminate()
    return None


def run_screenshots(series, figures):
    """Step 4: Screenshot figures via Playwright."""
    print(f"\n{'='*60}")
    print(f"  Step 4: Screenshotting figures ({series})")
    print(f"{'='*60}")

    cmd = [
        "node", "screenshot_figures.mjs",
        "--series", series,
        "--figures",
    ] + [str(f) for f in figures]

    result = subprocess.run(cmd, cwd=str(VIEWER_DIR))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: subplots → Next.js → screenshot")
    parser.add_argument("--series", required=True, choices=["dpmm", "topic"])
    parser.add_argument("--figures", nargs="+", default=["all"],
                        help="Which figures: 2 3 4 5 6, or 'all'")
    parser.add_argument("--skip-subplots", action="store_true",
                        help="Reuse existing subplot PNGs, skip regeneration")
    parser.add_argument("--skip-screenshot", action="store_true",
                        help="Generate subplots only, skip Next.js screenshot")
    args = parser.parse_args()

    all_figs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    figures = all_figs if "all" in args.figures else [int(f) for f in args.figures]

    dev_proc = None
    try:
        # Step 1
        if not args.skip_subplots:
            ok = run_subplot_generation(args.series, figures)
            if not ok:
                sys.exit(1)
        else:
            print("  Skipping subplot generation (--skip-subplots)")

        if args.skip_screenshot:
            print("  Skipping screenshot (--skip-screenshot)")
            return

        # Step 2
        setup_symlinks()

        # Step 3
        dev_proc = start_dev_server()
        if dev_proc is None and not check_server_running():
            print("  Cannot proceed without dev server. Start it manually:")
            print(f"    cd {VIEWER_DIR} && PORT={PORT} npm run dev")
            sys.exit(1)

        # Step 4
        ok = run_screenshots(args.series, figures)
        if not ok:
            sys.exit(1)

    finally:
        # Cleanup dev server if we started it
        if dev_proc is not None:
            print("\n  Stopping dev server...")
            dev_proc.terminate()
            try:
                dev_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dev_proc.kill()

    out_dir = ROOT / "benchmarks" / "paper_figures" / args.series
    print(f"\n{'='*60}")
    print(f"  All done. Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
