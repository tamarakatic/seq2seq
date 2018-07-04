import os

current_file = os.path.dirname(os.path.abspath(__file__))

ROOT_PATH = os.path.abspath(os.path.join(current_file, os.path.pardir))
DATA_DIR = os.path.join(ROOT_PATH, 'data/')
MODEL_PATH = os.path.join(ROOT_PATH, 'models/weights.npz')
