"""
Main entry for Actuated application
"""

import argparse
from multiprocessing import Pool, cpu_count
import os

import torch
import traci
import numpy as np
import pandas as pd
from tqdm import tqdm



def main():
    """
    Main function to parse arguments and run the application.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run the main application.")

    parser.add_argument('--net', type=str, default=BASE_DIR + r"/../../networks/4x4.net.xml")
    parser.add_argument('--route', type=str, default=BASE_DIR + r'/../../routes/4x4c2c1.rou.xml')
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--noised-edge", type=str, default="1,5")
    parser.add_argument("--noise-added", type=float, default=0.1)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--simulation-time", type=int, default=300)
    parser.add_argument("--run-per-alpha", type=int, default=3)
    parser.add_argument("--delta-time", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="")
    