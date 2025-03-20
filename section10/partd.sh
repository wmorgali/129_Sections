#!/bin/bash

# Path to the original pp.in file
PP_IN="qe_file/pp.in"

# Define the bands for the highest occupied valence band and lowest unoccupied conduction band
# Replace these values with the correct band indices from your calculation
VALENCE_BAND=4  # Highest occupied valence band
CONDUCTION_BAND=5  # Lowest unoccupied conduction band

# Create a new input file for pp.x
NEW_PP_IN="pp_ks_orbitals.in"
cp "$PP_IN" "$NEW_PP_IN"

# Update the kband parameters in the new input file
sed -i -r "s/kband\(1\)\s*=\s*[0-9]+/kband(1) = ${VALENCE_BAND}/" "$NEW_PP_IN"
sed -i -r "s/kband\(2\)\s*=\s*[0-9]+/kband(2) = ${CONDUCTION_BAND}/" "$NEW_PP_IN"

# Run the pp.x calculation
echo "Running pp.x to calculate KS orbitals..."
pp.x < "$NEW_PP_IN" > pp_ks_orbitals.out

# Check if the calculation was successful
if grep -q "JOB DONE" pp_ks_orbitals.out; then
    echo "KS orbitals calculation completed successfully."
    echo "Output files:"
    echo "- pp_ks_orbitals.out: Log file for the pp.x calculation."
    echo "- <prefix>.wfc#: Wavefunction files for the specified bands."
else
    echo "Error: KS orbitals calculation failed. Check pp_ks_orbitals.out for details."
    exit 1
fi