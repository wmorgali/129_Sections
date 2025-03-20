#!/bin/bash

# Directory containing the original input file
INPUT="qe_file/pw.graphene.scf.in"
# Define the required pseudopotential directory
PSEUDO_DIR="/root/Desktop/pseudo"
# Create an output directory for the SCF result files if it doesn't exist
OUTDIR="qe_file/results_lattice"
mkdir -p "$OUTDIR"

# Range of lattice constants (b-axis) from 0.4 to 0.7 with 10 points
lattice_constants=($(seq 0.4 0.03 0.7))

# Output file to store the extracted energies
ENERGY_FILE="lattice_energies.txt"

# Clear the energy file if it already exists
> "$ENERGY_FILE"

# Loop over each lattice constant
for lattice in "${lattice_constants[@]}"; do
    # Create a new input file for this lattice constant
    new_input="${OUTDIR}/pw.scf_lattice_${lattice}.in"
    cp "$INPUT" "$new_input"

    # Update the celldm(3) parameter to the current lattice constant
    sed -i -r "s/celldm\(3\)\s*=\s*[0-9.]+/celldm(3) = ${lattice}/" "$new_input"
    # Ensure the pseudo_dir parameter is set to the required pseudopotential directory
    sed -i -r "s|pseudo_dir\s*=\s*['\"][^'\"]+['\"]|pseudo_dir = '${PSEUDO_DIR}'|" "$new_input"

    # Define an output file name based on the current lattice constant
    output_file="${OUTDIR}/scf_lattice_${lattice}.out"
    echo "Running SCF calculation with lattice constant (b-axis) = ${lattice}..."
    # Execute the pw.x calculation (make sure pw.x is in your PATH)
    pw.x < "$new_input" > "$output_file"

    # Extract the total energy from the output file
    total_energy=$(grep '!    total energy' "$output_file" | awk '{print $5}')

    # Check if the energy was found
    if [ -z "$total_energy" ]; then
        echo "Total energy not found in $output_file. Skipping lattice constant = $lattice."
    else
        # Save the lattice constant and total energy to the energy file
        echo "$lattice $total_energy" >> "$ENERGY_FILE"
        echo "Extracted total energy for lattice constant = $lattice: $total_energy Ry"
    fi
done

echo "All SCF calculations completed. Check the ${OUTDIR} directory for outputs."

# Check if the energy file is not empty
if [ ! -s "$ENERGY_FILE" ]; then
    echo "No energy data found. Exiting."
    exit 1
fi

# Python script to plot the energy vs. lattice constant
python3 <<EOF
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the file
data = np.loadtxt("$ENERGY_FILE")
lattice_constants = data[:, 0]  # First column: lattice constants
total_energy = data[:, 1]  # Second column: total energy values

# Plot the data
plt.plot(lattice_constants, total_energy, marker='o', linestyle='-', color='b')
plt.xlabel('Lattice Constant (b-axis)')
plt.ylabel('Total Energy (Ry)')
plt.title('Total Energy vs. Lattice Constant (b-axis)')
plt.grid(True)

# Save the plot as a PNG file
plot_file="energy_vs_lattice.png"
plt.savefig(plot_file)
print(f"Plot saved as {plot_file}")

# Show the plot (optional)
plt.show()
EOF