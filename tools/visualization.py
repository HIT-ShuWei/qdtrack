import sys
sys.getdefaultencoding()
import pickle
import numpy as np


np.set_printoptions(threshold=np.inf)   # invoid show incompletely

# .pkl file path
path = '../qdtrack/test1.pkl'

# read the .pkl file
file = open(path, 'rb')
inf = pickle.load(file, encoding='iso-8859-1')

# print(inf)

inf = str(inf)
obj_path = 'res.txt'
ft = open(obj_path, "w")
ft.write(inf)
print('Finished!')