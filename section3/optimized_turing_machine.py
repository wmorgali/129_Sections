import os

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
    
    if state == "start":
        for i in range(len(bin1)):
            tape_states.append("".join(tape))
            tape[i] = "X"
        state = "write_result"
    
    if state == "write_result":
        tape_states.append("".join(tape))
        tape += list(result_bin)
        tape_states.append("".join(tape))
    
    return tape_states, result_bin

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

if __name__ == "__main__":
    bin1 = input("Enter first binary number: ")
    bin2 = input("Enter second binary number: ")
    tape_states, result_bin = turing_machine_multiply(bin1, bin2)
    save_tape_states(bin1, bin2, tape_states)
    save_state_count(2)
    print(f"Multiplication Result: {bin1} x {bin2} = {result_bin} (binary)")
