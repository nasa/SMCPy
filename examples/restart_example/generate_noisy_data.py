from model import model
import numpy as np


if __name__ == '__main__':
    # instance model / set up ground truth / add noise
    a = 2
    b = 3.5
    x = np.arange(50)
    m = model(x)
    std_dev = 0.6
    y_true = m.evaluate(a, b) 
    y_noisy = y_true + np.random.normal(0, std_dev, y_true.shape)

    np.savetxt('noisy_data.txt', y_noisy)
