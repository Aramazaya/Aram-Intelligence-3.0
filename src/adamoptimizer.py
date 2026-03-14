import numpy as np

class AdamOptimizer:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Args:
            alpha (float): step size (learning rate)
            beta1 (float): Exponential decay rate for first moment, in [0, 1)
            beta2 (float): Exponential decay rate for second moment, in [0, 1)
            epsilon (float): Small constant for numerical stability
        """
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0 # initialize timestep
        self.m = None  # initialize first moment vector
        self.v = None  # initialize second raw moment vector

    def step(self, theta, gt):
        # Initialize m and v
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)

        self.t += 1

        # get gradients

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gt

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gt * gt)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1.0 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        # Update parameters
        theta_updated = theta - self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return theta_updated
