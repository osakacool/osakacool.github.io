# ==============================================================================
# Script: generate_synthetic_dataset.R
# Description: Generates synthetic multi-source monitoring data for ancient 
#              timber halls based on the manuscript's FE simulation description.
# Author: AI Assistant
# ==============================================================================

set.seed(42) # For reproducibility

# --- Parameters from Manuscript Section 5.1 ---
n_samples <- 864000       # 7 days at 100Hz (approx, reduced for demo to 1 day)
sampling_rate <- 100      # Hz
duration_days <- 1        # Adjusted for R memory limits; scale up as needed
total_steps <- duration_days * 24 * 3600 * sampling_rate

# Environmental parameters
temp_daily_amp <- 5       # Daily temperature variation amplitude
temp_seasonal_base <- 15  # Base temperature
humidity_mean <- 60       # Mean relative humidity

# Seismic event injection parameters
event_times <- c(36000, 72000) # Event start indices (e.g., at 10h and 20h)
event_durations <- c(6000, 8000) # Duration in samples (60s, 80s)
pga_levels <- c(0.2, 0.4) # Peak Ground Acceleration for events

# --- Helper Functions ---
generate_environment <- function(n) {
  t <- seq_len(n) / sampling_rate
  # Sinusoidal daily cycle + seasonal envelope + noise
  temp <- temp_seasonal_base + temp_daily_amp * sin(2 * pi * t / 86400) + rnorm(n, 0, 0.5)
  hum <- humidity_mean + 10 * sin(2 * pi * t / 86400 + pi/4) + rnorm(n, 0, 2)
  return(list(temp = temp, hum = hum))
}

generate_ambient_vibration <- function(n) {
  # Low-amplitude white noise filtered to simulate structural modes
  noise <- rnorm(n, 0, 0.005)
  # Simple moving average to simulate low-pass structural filtering
  ambient <- filter(noise, rep(1/50, 50), sides = 2)
  ambient[is.na(ambient)] <- 0
  return(as.numeric(ambient))
}

generate_seismic_response <- function(duration, pga) {
  # Simulated non-linear response: decaying sinusoid + high freq content
  t <- seq_len(duration) / sampling_rate
  decay <- exp(-3 * t)
  response <- pga * decay * sin(2 * pi * 2.5 * t) * (1 + 0.5 * sin(2 * pi * 8 * t))
  response <- response + rnorm(duration, 0, pga * 0.05)
  return(response)
}

# --- Main Generation Loop ---
cat("Generating synthetic dataset...\n")
env_data <- generate_environment(total_steps)
acc_roof <- generate_ambient_vibration(total_steps)
strain_joint <- generate_ambient_vibration(total_steps) * 10 # Strain scale
disp_base <- generate_ambient_vibration(total_steps) * 0.5

# Initialize labels and vulnerability index
is_event <- rep(0, total_steps)
vulnerability_index <- rep(0.05, total_steps) # Baseline drift
damage_accumulator <- 0.05

# Inject Events
for (i in seq_along(event_times)) {
  start_idx <- event_times[i]
  end_idx <- min(start_idx + event_durations[i], total_steps)
  len <- end_idx - start_idx
  
  # Add seismic signal
  seismic_sig <- generate_seismic_response(len, pga_levels[i])
  acc_roof[start_idx:end_idx] <- acc