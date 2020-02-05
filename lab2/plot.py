

import matplotlib.pyplot as plt

np = [1,2,4,8,16,32]
dec_test_error = [40.495, 75.1802, 133.195 , 131.56, 11.4418, 4.97487 ]

plt.figure(figsize=(8,4))
plt.plot(np, dec_test_error)
plt.xlabel("number of processes")
plt.ylabel("GFlops")
plt.title("Performance changing with number of processes")
plt.legend()
plt.show()