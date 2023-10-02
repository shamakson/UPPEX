import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
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
model_data_paths = [f"/scratch3/esamakin/particles/restart_WR_PF{str(i).zfill(3)}_17080103234500_echam.nc" for i in range(1, 51)]

# Load the observation file using NetCDF4
observation_path = "/scratch3/esamakin/ModE-Sim_set_1420-3_m041_1708_day_1.temp2.nc"
observation_data = Dataset(observation_path)

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
        #distance = abs(observation_value - particle_value)
        #weights[i] *= np.exp(-0.5 * (distance ** 2))
         
         # Update particle weights using Euclidean distance
        distance = np.sqrt((particle_value - observation_value) ** 2)
        weights[i] *= 1.0 / (1.0 + distance)  

    # Normalize the weights
    weights /= np.sum(weights)
    cumulative_weights += weights

    values = [lon, lat, observation_value] + particle_values
    df.loc[len(df)] = values

# Calculate average weights
average_weights = cumulative_weights / len(europe_points)

# Calculate the sum of all cumulative weights
total_cumulative_weight = np.sum(cumulative_weights)
# Normalize the cumulative weights
cumulative_weights = cumulative_weights / total_cumulative_weight

# Select top 25 particles with largest average weights
top_particle_indices = np.argsort(cumulative_weights)[-25:][::-1]

print("\nTop 10 selected particles:")
for idx, particle_index in enumerate(top_particle_indices):
    print(f"Particle {idx+1}: Weight = {cumulative_weights[particle_index]:.4f}")

# Now, copy the selected files with the largest average weights
selected_files = []
for idx, particle_index in enumerate(top_particle_indices):
    input_model_file = model_data_paths[particle_index]
    output_file_name = os.path.basename(input_model_file).replace(".nc", f"_selected_{idx + 1}.nc")
    shutil.copy(input_model_file, output_file_name)
    selected_files.append(output_file_name)
    print(f"Input file with largest weight {idx + 1} selected and copied")


########################################################################################################
########################################################################################################
# Define the fill value for masked entries
fill_value = -9999  
# Combine all members of the model (temp2 variable)
combined_temp2 = []
for model_file in model_data_paths:
    try:
        model_data = Dataset(model_file)
        temp2_values = model_data.variables['temp2'][:]
        
        # Fill masked values with the specified fill_value
        temp2_values_filled = np.where(np.ma.getmask(temp2_values), fill_value, temp2_values)
        
        combined_temp2.append(temp2_values_filled)
    except Exception as e:
        print(f"An error occurred while processing {model_file}: {str(e)}")

# Stack temp2 values along a new axis to combine them
combined_temp2 = np.stack(combined_temp2, axis=0)
# Ensure that combined_temp2 is a 2D array
combined_temp2_2d = combined_temp2.reshape(combined_temp2.shape[0], -1)

# Perform Singular Value Decomposition (SVD)
U, S, VT = svd(combined_temp2_2d, full_matrices=False)

# Number of EOF modes to use
num_eof_modes = 10

# Select the first num_eof_modes EOFs
eof_modes_to_use = VT[:num_eof_modes, :]

# Assuming temp2_values.shape is (lat, lon)
lat, lon = temp2_values.shape

# Reshape the EOF modes back to the grid shape
eof_modes_grid_shape = eof_modes_to_use.reshape(num_eof_modes, lat, lon)

##################################################################################################
##################################################################################################
# Perturb the selected particles with model error 
##################################################################################################
##################################################################################################
mu = 0.0  # Mean of model error perturbation
sigma = 1.0  # Standard deviation of model error perturbation
# Sum the first 10 EOFs on the grid
sum_eof_modes_grid = np.mean(eof_modes_grid_shape, axis=0)

# Perturb the selected files with model error 
for idx, selected_file in enumerate(selected_files):
    try:
        selected_data_read = Dataset(selected_file, 'r')
        # Get latitude and longitude dimensions from the original file
        latsdim = selected_data_read.dimensions['lat'].size
        lonsdim = selected_data_read.dimensions['lon'].size

        
        # Generate random white noise perturbation with the same shape as sum_eof_modes_grid
        random_perturbation = np.random.normal(loc=mu, scale=sigma, size=(latsdim, lonsdim))
        
        # Create a copy of the selected file
        perturbed_file_name = selected_file.replace(".nc", f"_perturbed_{idx + 1}.nc")
        shutil.copy(selected_file, perturbed_file_name)
        
        # Open the copied file for modification
        perturbed_data = Dataset(perturbed_file_name, 'r+')
        
        # Modify 'temp2' variable with perturbation
        perturbation = sum_eof_modes_grid + random_perturbation
        perturbed_data.variables['temp2'][:] += perturbation
        
        # Close the copied file
        perturbed_data.close()
        
        print(f"Perturbed data saved to {perturbed_file_name}")
        
    except Exception as e:
        print(f"An error occurred while processing {selected_file}: {str(e)}")
    else:
        print(f"Successfully perturbed {selected_file} to {perturbed_file_name}.")

