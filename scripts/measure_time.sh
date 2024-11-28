#!/bin/bash

# List of scripts to measure
scripts=("scripts/attack_100_aspl.sh" "scripts/attack_100_mist.sh")

# File to store the timing results in CSV format
output_file="timing_results.csv"

# Initialize the CSV file with a header
echo "Script,Execution Time" > "$output_file"

# Loop through each script
for script in "${scripts[@]}"; do
    if [[ -f $script && -x $script ]]; then
        echo "Running $script..."

        # Record start time
        start_time=$(date +%s%N) # Seconds since epoch with nanoseconds

        # Execute the script
        bash "$script"

        # Record end time
        end_time=$(date +%s%N)

        # Calculate elapsed time in nanoseconds
        elapsed_ns=$((end_time - start_time))

        # Convert to milliseconds
        elapsed_ms=$((elapsed_ns / 1000000))

        # Convert milliseconds to HH:MM:SS.mmm format
        hours=$((elapsed_ms / 3600000))
        minutes=$(( (elapsed_ms % 3600000) / 60000 ))
        seconds=$(( (elapsed_ms % 60000) / 1000 ))
        milliseconds=$((elapsed_ms % 1000))

        # Format time as HH:MM:SS.mmm
        formatted_time=$(printf "%02d:%02d:%02d.%03d" $hours $minutes $seconds $milliseconds)

        # Write to CSV
        echo "$script,$formatted_time" >> "$output_file"
    else
        echo "$script,Not Found or Not Executable" >> "$output_file"
    fi
done

echo "Timing results have been saved to $output_file."
