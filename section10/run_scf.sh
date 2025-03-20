#!/bin/bash

# Path to the original input file
input="qe_file/pw.graphene.scf.in"
# Define the pseudopotential directory
pseudo="/root/Desktop/pseudo"
# Create an output directory for the SCF results
output="qe_file/results"
mkdir -p "$output"

ecutwfc_vals=(10 15 20 25 30 35 40 60)

# Loops over each value
for ecut in "${ecutwfc_vals[@]}"; do
    # Create a new input file for this cutoff value
    new_input="qe_file/pw.scf_${ecut}.in"
    cp "$input" "$new_input"

    # Update the ecutwfc parameter to the current value
    sed -i -r "s/ecutwfc\s*=\s*[0-9]+/ecutwfc = ${ecut}/" "$new_input"
    # Ensure the ecutrho parameter is set to 200.0
    sed -i -r "s/ecutrho\s*=\s*[0-9.]+/ecutrho = 200.0/" "$new_input"
    # Update the pseudo_dir parameter to the required pseudopotential directory
    sed -i -r "s|pseudo_dir\s*=\s*['\"][^'\"]+['\"]|pseudo_dir = '${pseudo}'|" "$new_input"

    # Define an output file name based on the current cutoff value
    output_file="${output}/scf_${ecut}.out"
    echo "Running SCF calculation with ecutwfc = ${ecut}..."
    # Execute the pw.x calculation
    pw.x < "$new_input" > "$output_file"
done