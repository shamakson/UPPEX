import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from filterpy.monte_carlo import systematic_resample
import shutil

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

# Select top 5 particles with largest average weights
top_particle_indices = np.argsort(average_weights)[-5:][::-1]

print("\nTop 5 selected particles:")
for idx, particle_index in enumerate(top_particle_indices):
    print(f"Particle {idx+1}: Weight = {average_weights[particle_index]:.4f}")

# Now, copy the selected files with the largest average weights
selected_files = []
for idx, particle_index in enumerate(top_particle_indices):
    input_model_file = model_data_paths[particle_index]
    output_file_name = os.path.basename(input_model_file).replace(".nc", f"_selected_{idx + 1}.nc")
    shutil.copy(input_model_file, output_file_name)
    print (f"Input file with largest weight {idx + 1} selected and copied")

##################################################################################################
##################################################################################################
# Perturb the selected particles with model error
##################################################################################################
##################################################################################################
mu = 0.0  # Mean of model error perturbation
sigma = 1.0  # Standard deviation of model error perturbation

# Perturb the selected particles with model error
mu = 0.0  # Mean of model error perturbation
sigma = 1.0  # Standard deviation of model error perturbation

# Define the model error value (replace 0.1 with your desired value)
model_error_value = 0.1  # Adjust this value as needed

# Loop through the selected files and perturb them
for idx, selected_file in enumerate(selected_files):
    selected_data_read = Dataset(selected_file)
    
    # Get latitude and longitude dimensions
    latsdim = selected_data_read.dimensions['lat'].size
    lonsdim = selected_data_read.dimensions['lon'].size
    
    # Create a new NetCDF file for the perturbed data
    perturbed_output_file = f"perturbed_selected_{idx + 1}.nc"
    perturbed_data = Dataset(perturbed_output_file, 'w', format='NETCDF4')
    
    # Create latitude and longitude dimensions
    latdim = perturbed_data.createDimension('latitude', latsdim)
    londim = perturbed_data.createDimension('longitude', lonsdim)
    
    # Create latitude and longitude variables
    latitude = perturbed_data.createVariable('latitude', 'f8', ('latitude',))
    longitude = perturbed_data.createVariable('longitude', 'f8', ('longitude',))
    
    # Copy latitude and longitude values from the original file
    latitude[:] = selected_data_read.variables['lat'][:]
    longitude[:] = selected_data_read.variables['lon'][:]
    
    # Create perturbed variable (assuming 'temp2' is the variable you want to perturb)
    perturbed_temp2 = perturbed_data.createVariable('temp2', 'f4', ('latitude', 'longitude'))
    
    # Apply model error perturbation
    perturbed_temp2[:, :] = selected_data_read.variables['temp2'][:] + model_error_value
    
    # Set attributes if needed
    perturbed_temp2.long_name = "2m temperature"
    perturbed_temp2.units = "K"
    
    # Close the perturbed NetCDF file
    perturbed_data.close()

    print(f"Perturbed data saved to {perturbed_output_file}")
