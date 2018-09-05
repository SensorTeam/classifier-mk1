import pickle
import matplotlib.pyplot as plt   
ax = pickle.load(open("plot.pickle", "rb"))
plt.scatter(0,-1, c='black', marker='*')
plt.show()