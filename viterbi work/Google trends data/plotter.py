"""
plots data from csv files
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt

data = pd.read_csv(sys.argv[1])
print data.info()
data.plot()

plt.show()