
import numpy as np
import os

from config import *
from util import *

def main():
	
	for key in config:
	    print(key, ':', config[key])
	 
	print('\n')
	
	solver = config['Solver']

	if solver == 'BTP':
		tracker = MHHT_BTP(config)
	elif solver == 'Murty':
		tracker = MHHT_Murty(config)
	elif solver == 'MSC':
		tracker = MHHT_MSC(config)

	tracker.track()

if __name__ == '__main__':
	
	main()

