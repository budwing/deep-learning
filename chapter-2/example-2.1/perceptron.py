import matplotlib.pyplot as plt
import numpy as np
import logging

log = logging.getLogger("perceptron")
class BiPerceptron:
    def __init__(self, w1=0, w2=0, theta=0) -> None:
        self.w1 = w1
        self.w2 = w2
        self.theta = theta
    
    def __call__(self, x1, x2):
        return self.perceive(x1, x2)
    
    def perceive(self, x1, x2):
        signal = self.w1*x1 + self.w2*x2
        if signal <= self.theta:
            return 0
        else:
            return 1
    
    def plot(self, logic_op:str=""):
        x1 = np.arange(-2,2,0.1)
        x2 = (self.theta - self.w1*x1)/self.w2
        plt.plot(x1, x2, color="black", linestyle="--")
        v = [0,1]
        for op1 in v:
            for op2 in v:
                r = self.perceive(op1,op2)
                plt.scatter([op1], [op2], color="black", s=50, marker="o" if r==1 else "x")
                plt.annotate(f"({op1}, {op2})", (op1, op2))
        plt.axhline(0, color="black")
        plt.axvline(0, color="black")
        plt.title(f"perceptron for logic {logic_op} operation")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.show()

class Perceptron:
    log = logging.getLogger("perceptron")
    def __init__(self, W:np.ndarray, b:float) -> None:
        self.W = W
        self.b = b

    def __call__(self, X:np.ndarray):
        return self.perceive(X)
    
    def perceive(self, X:np.ndarray):
        if X.shape != self.W.shape:
            log.warning(f"The shape of input X is not the same as weight W")
            return 0
        signal = np.dot(self.W, X) + self.b
        if signal <= 0:
            return 0
        else:
            return 1
        
def plot_xor(xor):
    v = [0,1]
    for op1 in v:
        for op2 in v:
            r = xor(op1,op2)
            plt.scatter([op1], [op2], color="black", s=50, marker="o" if r==1 else "x")
            plt.annotate(f"({op1}, {op2})", (op1, op2))
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.title("perceptron for logic XOR operation")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
   
if __name__ == "__main__":
    logic_and = BiPerceptron(0.3,0.3,0.5)
    print("logic and:")
    print(f"1 and 1 = {logic_and(1,1)}")
    print(f"1 and 0 = {logic_and(1,0)}")
    print(f"0 and 1 = {logic_and(0,1)}")
    print(f"0 and 0 = {logic_and(0,0)}")
    # logic_and.plot("AND")

    logic_or = BiPerceptron(0.3,0.3,0.1)
    print("logic or:")
    print(f"1 or 1 = {logic_or(1,1)}")
    print(f"1 or 0 = {logic_or(1,0)}")
    print(f"0 or 1 = {logic_or(0,1)}")
    print(f"0 or 0 = {logic_or(0,0)}")
    # logic_or.plot("OR")

    logic_not_and = BiPerceptron(-0.3,-0.3,-0.5)
    print("logic not and:")
    print(f"not (1 and 1) = {logic_not_and(1,1)}")
    print(f"not (1 and 0) = {logic_not_and(1,0)}")
    print(f"not (0 and 1) = {logic_not_and(0,1)}")
    print(f"not (0 and 0) = {logic_not_and(0,0)}")

    logic_xor = lambda x1, x2: logic_and(logic_or(x1, x2), logic_not_and(x1, x2))
    print("logic xor:")
    print(f"1 xor 1 = {logic_xor(1,1)}")
    print(f"1 xor 0 = {logic_xor(1,0)}")
    print(f"0 xor 1 = {logic_xor(0,1)}")
    print(f"0 xor 0 = {logic_xor(0,0)}")
    plot_xor(logic_xor)

    p = Perceptron(np.array([0.6,0.6]), -1)
    print("general perceptron for logic or")
    print(f"1 or 1 = {p(np.array([1,1]))}")
    print(f"1 or 0 = {p(np.array([1,0]))}")
    print(f"0 or 1 = {p(np.array([0,1]))}")
    print(f"0 or 0 = {p(np.array([0,0]))}")
