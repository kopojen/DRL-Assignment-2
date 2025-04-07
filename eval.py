# custom_taxi_env.py
import gym
import numpy as np
import time
import random


from xml.etree import ElementTree as ET
import importlib.util
import requests
import argparse
import torch
import random
import sys
import importlib
import env
from Ntuple import NTupleApproximator
from student_agent import init_model
import gc


if __name__ == "__main__":
    init_model()
    env.eval_score()