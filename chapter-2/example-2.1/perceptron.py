class BiPerceptron:
    def __init__(self, w1=0, w2=0, theta=0) -> None:
        self.w1 = w1
        self.w2 = w2
        self.theta = theta

    def perceive(self,x1,x2):
        signal = self.w1*x1 + self.w2*x2
        if signal <= self.theta:
            return 0
        else:
            return 1

import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("perceptron")

class Perceptron:
    def __init__(self, W:np.ndarray, b:float) -> None:
        self.W = W
        self.b = b
    def perceive(self, X:np.ndarray):
        if X.shape != self.W.shape:
            log.warning(f"The shape of input X is not the same as weight W")
            return 0
        signal = np.dot(self.W, X) + self.b
        if signal <= 0:
            return 0
        else:
            return 1
        
if __name__ == "__main__":
    bp = BiPerceptron(1,1,0)
    print("BiPercetron:")
    print(bp.perceive(1,1))
    print(bp.perceive(1,0))
    print(bp.perceive(0,1))
    print(bp.perceive(0,0))

    p = Perceptron(np.array([1,1]), 0)
    print("Perceptron")
    print(p.perceive(np.array([1,1])))
    print(p.perceive(np.array([1,0])))
    print(p.perceive(np.array([0,1])))
    print(p.perceive(np.array([0,0])))
