
import numpy as np
import os

from config import *
from util import *
from murty import *
from BTP import *

if __name__ == '__main__':
	
	for key in config:
	    print(key, ':', config[key])
	 
	print('\n')
	
	tracker = MHHT_Murty(config) if config['Solver'] == 'Murty' else MHHT_BTP(config)

	tracker.track()

