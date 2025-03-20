#!/bin/bash

# Directory containing the SCF output files
OUTDIR="qe_file/results"

# Output file to store the extracted energies
ENERGY_FILE="total_energies.txt"

# Clear the energy file if it already exists
> "$ENERGY_FILE"

# Loop through each ecutwfc value
for ecut in 10 15 20 25 30 35 40 60; do
    # Define the output file for this ecutwfc value
    output_file="${OUTDIR}/scf_${ecut}.out"

    # Check if the output file exists
    if [ ! -f "$output_file" ]; then
        echo "Output file $output_file not found. Skipping ecutwfc = $ecut."
        continue
    fi

    # Extract the total energy using grep and awk
    total_energy=$(grep '!    total energy' "$output_file" | awk '{print $5}')

    # Check if the energy was found
    if [ -z "$total_energy" ]; then
        echo "Total energy not found in $output_file. Skipping ecutwfc = $ecut."
    else
        # Save the ecutwfc and total energy to the energy file
        echo "$ecut $total_energy" >> "$ENERGY_FILE"
        echo "Extracted total energy for ecutwfc = $ecut: $total_energy Ry"
    fi
done

echo "Total energies extracted and saved to $ENERGY_FILE."

# Check if the energy file is not empty
if [ ! -s "$ENERGY_FILE" ]; then
    echo "No energy data found. Exiting."
    exit 1
fi

# Python script to plot the energy vs. ecutwfc
python3 <<EOF
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the file
data = np.loadtxt("$ENERGY_FILE")
ecutwfc = data[:, 0]  # First column: ecutwfc values
total_energy = data[:, 1]  # Second column: total energy values

# Plot the data
plt.plot(ecutwfc, total_energy, marker='o', linestyle='-', color='b')
plt.xlabel('ecutwfc (Ry)')
plt.ylabel('Total Energy (Ry)')
plt.title('Total Energy vs. ecutwfc')
plt.grid(True)

# Save the plot as a PNG file
plot_file="energy_vs_ecutwfc.png"
plt.savefig(plot_file)
print(f"Plot saved as {plot_file}")

EOF