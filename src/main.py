"""
Main entry point for the application.
"""

import argparse
import os

def main():
    """
    Main function to parse arguments and run the application.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run the main application.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')

    parser.add_argument('--net', type=str, default=BASE_DIR + r"/networks/4x4.net.xml")
    parser.add_argument('--route', type=str, default=BASE_DIR + r'/routes/4x4c2c1.rou.xml')
    parser.add_argument('--noise-added', type=str, default="True")
    parser.add_argument("--intersection-id", type=str, default="10")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--noised-edge", type=str, default="CR30_LR_8")
    parser.add_argument("--simulation-time", type=int, default=1200)
    parser.add_argument("--run-per-alpha", type=int, default=5)
    parser.add_argument("--delta-time", type=int, default=3)
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--distance-threshold", type=int, default=200)
    parser.add_argument("--omega", type=float, default=0.0)
    parser.add_argument("--cutoff", type=int, default=2)


    args = parser.parse_args()

    