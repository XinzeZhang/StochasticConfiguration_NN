import scipy.io
import numpy as np

# data
data = scipy.io.loadmat("Demo_Iris.mat")
print(data)
for i in data:
	if '__' not in i and 'readme' not in i:
		np.savetxt(("Iris/"+i+".csv"),data[i],delimiter=' ',fmt='%s')
