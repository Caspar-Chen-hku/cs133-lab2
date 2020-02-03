

import matplotlib.pyplot as plt

np = [1,2,4,8,16,32]
dec_test_error = [26.8176, 51.8944, 93.6048, 74.7562, 9.16179, 1.36207 ]

plt.figure(figsize=(8,4))
plt.plot(np, dec_test_error)
plt.xlabel("number of processes")
plt.ylabel("GFlops")
plt.title("Performance changing with number of processes")
plt.legend()
plt.show()