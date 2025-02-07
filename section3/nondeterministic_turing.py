import os
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def binary_to_decimal(binary_str):
    return int(binary_str, 2)

def decimal_to_binary(decimal):
    return bin(decimal)[2:]

def probabilistic_turing_machine_multiply(bin1, bin2):
    num1 = binary_to_decimal(bin1)
    num2 = binary_to_decimal(bin2)
    result = num1 * num2
    result_bin = decimal_to_binary(result)
    
    tape = list(bin1) + ["#"] + list(bin2) + ["$"] + ["B"] * 10
    head_position = 0
    state = "start"
    step_count = 0
    
    if state == "start":
        step_count += len(bin1)
        state = "compute"
    
    if state == "compute":
        step_count += random.randint(len(bin1), len(bin1) + len(bin2))
        state = "write_result"
    
    if state == "write_result":
        step_count += len(result_bin)
    
    return step_count

def compute_heatmap_data(max_a, max_b):
    heatmap = np.zeros((max_a - 1, max_b - 1))
    for a in range(2, max_a + 1):
        for b in range(2, max_b + 1):
            steps = []
            for _ in range(50):
                bin1 = ''.join(random.choice("01") for _ in range(a))
                bin2 = ''.join(random.choice("01") for _ in range(b))
                step_count = probabilistic_turing_machine_multiply(bin1, bin2)
                steps.append(step_count)
            heatmap[a - 2, b - 2] = np.mean(steps)
    return heatmap

def plot_heatmap(heatmap, max_a, max_b):
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap, annot=True, fmt=".1f", cmap="coolwarm", xticklabels=range(2, max_b+1), yticklabels=range(2, max_a+1))
    plt.xlabel("b")
    plt.ylabel("a")
    plt.title("Heatmap of Average Computation Steps ⟨n⟩ - Probabilistic TM")
    plt.savefig("heatmap_complexity_probabilistic.png")
    plt.show()

if __name__ == "__main__":
    max_a, max_b = 30, 30
    heatmap_data = compute_heatmap_data(max_a, max_b)
    plot_heatmap(heatmap_data, max_a, max_b)
