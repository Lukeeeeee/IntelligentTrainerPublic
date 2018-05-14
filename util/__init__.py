import sys
import os

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
sys.path.append(CURRENT_PATH + '/../')

from util.classCreator.classCreator import *
