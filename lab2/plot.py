

import matplotlib.pyplot as plt

np = [1,2,4,8,16,32]
dec_test_error = [6.01022, 11.8307, 20.4593, 22.1712, 6.56264, 1.21355 ]

plt.figure(figsize=(8,4))
plt.plot(np, dec_test_error)
plt.xlabel("number of processes")
plt.ylabel("GFlops")
plt.title("Performance changing with number of processes")
plt.legend()
plt.show()