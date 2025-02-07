import os
import random
import matplotlib.pyplot as plt

def binary_to_decimal(binary_str):
    return int(binary_str, 2)

def decimal_to_binary(decimal):
    return bin(decimal)[2:]

def turing_machine_multiply(bin1, bin2):
    num1 = binary_to_decimal(bin1)
    num2 = binary_to_decimal(bin2)
    result = num1 * num2
    result_bin = decimal_to_binary(result)
    
    tape = list(bin1) + ["#"] + list(bin2) + ["$"] + ["B"] * 10
    head_position = 0
    state = "start"
    tape_states = []
    step_count = 0
    
    if state == "start":
        for i in range(len(bin1)):
            tape_states.append("".join(tape))
            tape[i] = "X"
            step_count += 1
        state = "write_result"
    
    if state == "write_result":
        tape_states.append("".join(tape))
        tape += list(result_bin)
        tape_states.append("".join(tape))
        step_count += len(result_bin)
    
    return tape_states, result_bin, step_count

def save_tape_states(bin1, bin2, tape_states):
    filename = f"multiplication_{bin1}_x_{bin2}.dat"
    with open(filename, "w") as file:
        for state in tape_states:
            file.write(state + "\n")
    print(f"Tape states saved to {filename}")

def save_state_count(count):
    with open("optimized_state_count.txt", "w") as file:
        file.write(f"Optimized State Count: {count}\n")
    print("State count saved to optimized_state_count.txt")

def compute_statistics(lengths):
    results = {}
    for La, Lb in lengths:
        steps = []
        for _ in range(100):  # Test 100 different inputs
            bin1 = ''.join(random.choice("01") for _ in range(La))
            bin2 = ''.join(random.choice("01") for _ in range(Lb))
            _, _, step_count = turing_machine_multiply(bin1, bin2)
            steps.append(step_count)
        results[(La, Lb)] = {
            "max": max(steps),
            "min": min(steps),
            "avg": sum(steps) / len(steps),
            "histogram": steps
        }
    return results

def plot_histograms(results):
    for (La, Lb), data in results.items():
        plt.hist(data["histogram"], bins=20, alpha=0.7, label=f"L({La},{Lb})")
        plt.xlabel("Steps (n)")
        plt.ylabel("Frequency")
        plt.title(f"Computation Complexity for L({La},{Lb})")
        plt.legend()
        plt.savefig(f"histogram_L{La}_{Lb}.png")
        plt.close()

def save_complexity_results(results):
    with open("complexity_statistics.txt", "w") as file:
        for (La, Lb), data in results.items():
            file.write(f"L({La},{Lb}): max={data['max']}, min={data['min']}, avg={data['avg']}\n")
    print("Complexity statistics saved to complexity_statistics.txt")

if __name__ == "__main__":
    lengths = [(2,3), (3,2), (3,5), (5,3), (3,12), (12,3)]
    results = compute_statistics(lengths)
    plot_histograms(results)
    save_complexity_results(results)
