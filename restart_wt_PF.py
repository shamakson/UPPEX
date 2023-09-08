import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from filterpy.monte_carlo import systematic_resample
import shutil
from scipy.linalg import svd
import numpy.ma as ma 

# Define the 10 points over Europe by longitude and latitude
# Example latitude and longitude coordinates for testing (10 points over Europe)
europe_points = [
    (48.8566, 2.3522),   # Paris, France
    (51.5074, -0.1278),  # London, UK
    (41.9028, 12.4964),  # Rome, Italy
    (52.5200, 13.4050),  # Berlin, Germany
    (40.4165, -3.70256),  # Madrid, Spain
    (44.8378, 0.5792), # Bordeaux, France
    (51.1079, 17.0385),# Wroclaw, Poland
    (47.3769, 8.5417), # Zurich, Switzerland
    (52.0116, 4.3571),# Delft, Netherlands
    (55.6761, 12.5683)   # Copenhagen, Denmark
]

# Define the paths to the model_data files
model_data_paths = [f"/scratch3/esamakin/particles/restart_WR_PF{str(i).zfill(3)}_17080103234500_echam.nc" for i in range(1, 21)]

# Load the observation file using NetCDF4
observation_path = "/scratch3/esamakin/ModE-Sim_set_1420-3_m041_1708_day_3.toasurf.nc"
observation_data = Dataset(observation_path)

# Create a dataframe to store temperature values
columns = ["lon", "lat", "obs"] + [f"particle{i:03d}" for i in range(1, 21)]
df = pd.DataFrame(columns=columns)

# Initialize particle filter parameters
num_particles = len(model_data_paths)
weights = np.ones(num_particles) / num_particles
cumulative_weights = np.zeros(num_particles)

# Loop through the Europe points and extract data
for lat, lon in europe_points:
    lon_idx = np.argmin(np.abs(observation_data.variables['lon'][:] - lon))
    lat_idx = np.argmin(np.abs(observation_data.variables['lat'][:] - lat))
    observation_value = observation_data.variables['temp2'][0, lat_idx, lon_idx]
    
    print(f"Processing point: Latitude {lat}, Longitude {lon}")
    print(f"Observation value: {observation_value:.2f}")
    
    particle_values = []
    for i, model_file in enumerate(model_data_paths):
        model_data = Dataset(model_file)
        particle_value = model_data.variables['temp2'][lat_idx, lon_idx]
        particle_values.append(particle_value)
        
        # Update particle weights using a simple Gaussian likelihood
        distance = abs(observation_value - particle_value)
        weights[i] *= np.exp(-0.5 * (distance ** 2))
 
        #print(f"Particle {i+1}: Weight = {weights[i]:.4f}")
    
    # Normalize the weights
    weights /= np.sum(weights)
    cumulative_weights += weights

    values = [lon, lat, observation_value] + particle_values
    df.loc[len(df)] = values

    # Resample particles
    resampled_indices = systematic_resample(weights)
    particles_updated = [model_data_paths[i] for i in resampled_indices]

# Calculate average weights
average_weights = cumulative_weights / len(europe_points)

# Select top 7 particles with largest average weights
top_particle_indices = np.argsort(average_weights)[-7:][::-1]

print("\nTop 5 selected particles:")
for idx, particle_index in enumerate(top_particle_indices):
    print(f"Particle {idx+1}: Weight = {average_weights[particle_index]:.4f}")

# Now, copy the selected files with the largest average weights
selected_files = []
for idx, particle_index in enumerate(top_particle_indices):
    input_model_file = model_data_paths[particle_index]
    output_file_name = os.path.basename(input_model_file).replace(".nc", f"_selected_{idx + 1}.nc")
    shutil.copy(input_model_file, output_file_name)
    selected_files.append(output_file_name)
    print (f"Input file with largest weight {idx + 1} selected and copied")


##################################################################################################
##################################################################################################
# Perturb the selected particles with model error
##################################################################################################
##################################################################################################
mu = 0.0  # Mean of model error perturbation
sigma = 1.0  # Standard deviation of model error perturbation

for idx, selected_file in enumerate(selected_files):
    selected_data_read = Dataset(selected_file, 'r+')
    
    # Check if the file was successfully opened
    if selected_data_read is not None and selected_data_read.isopen():
        try:
            # Get latitude and longitude dimensions from the original file
            latsdim = selected_data_read.dimensions['lat'].size
            lonsdim = selected_data_read.dimensions['lon'].size

            # Generate random white noise perturbation
            random_perturbation = np.random.normal(loc=mu, scale=sigma, size=(latsdim, lonsdim))

            # Modify 'temp2' variable with perturbation
            selected_data_read.variables['temp2'][:] += random_perturbation

            # Save the updated selected files with _perturbed suffix
            perturbed_file_name = selected_file.replace(".nc", f"_perturbed_{idx + 1}.nc")
            shutil.copy(selected_file, perturbed_file_name)
            print(f"Perturbed data saved to {perturbed_file_name}")
            
            # Close the modified NetCDF file
            selected_data_read.close()
        except Exception as e:
            print(f"An error occurred while processing {selected_file}: {str(e)}")
            selected_data_read.close()  # Close the file in case of an error
    else:
        print(f"Error: Unable to open {selected_file}. Skipping...")

##################################################################################################
##################################################################################################
# Perturbing the data using the first ten EOF modes multiplied by random scalars
##################################################################################################
##################################################################################################
# Define the number of EOF modes to use for perturbation
num_eof_modes = 10  # Adjust this as needed

# Initialize an array to store random scaling factors for EOF modes
random_scalars = np.random.normal(loc=0.0, scale=1.0, size=num_eof_modes)

for idx, selected_file in enumerate(selected_files):
    selected_data_read = Dataset(selected_file, 'r+')
    
    # Check if the file was successfully opened
    if selected_data_read is not None and selected_data_read.isopen():
        try:
            # Load temperature data 'temp' from the current selected file
            temp_data = selected_data_read.variables['temp2'][:]

            # Convert masked array to regular array
            if isinstance(temp_data, ma.MaskedArray):
                temp_data = temp_data.filled()

            # Flatten the temperature data to 2D (time, space)
            temp_data_2d = temp_data.reshape(temp_data.shape[0], -1)

            # Subtract the mean from each column (space) to obtain anomalies
            anomalies = temp_data_2d - temp_data_2d.mean(axis=0)

            # Perform Singular Value Decomposition (SVD)
            U, S, VT = svd(anomalies, full_matrices=False)

            # Use the first num_eof_modes EOF modes for perturbation
            eof_modes_to_use = U[:, :num_eof_modes]

            # Reshape random_scalars to have the shape (num_eof_modes, 1)
            random_scalars_reshaped = random_scalars.reshape(-1, 1)

            # Perturb the temperature data with EOF modes and random scalars
            perturbed_temp_data = temp_data_2d + np.dot(eof_modes_to_use, random_scalars_reshaped)
            
            # Reshape the perturbed data back to its original shape
            perturbed_temp_data = perturbed_temp_data.reshape(temp_data.shape)

            # Update the 'temp' variable with the perturbed values
            selected_data_read.variables['temp2'][:] = perturbed_temp_data

            # Save the updated selected files with _perturbed_EOF suffix
            perturbed_file_name = selected_file.replace(".nc", f"_perturbed_EOF_{idx + 1}.nc")
            shutil.copy(selected_file, perturbed_file_name)
            print(f"Perturbed data saved to {perturbed_file_name}")
            
            # Close the modified NetCDF file
            selected_data_read.close()
        except Exception as e:
            print(f"An error occurred while processing {selected_file}: {str(e)}")
            selected_data_read.close()  # Close the file in case of an error
    else:
        print(f"Error: Unable to open {selected_file}. Skipping...")
