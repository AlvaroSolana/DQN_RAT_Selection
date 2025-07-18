import numpy as np

def path_loss(d, fc):
    """
    Calculates path loss using the COST 231 Walfisch-Ikegami model.
    Parameters:
        d (float): Distance between transmitter and receiver in meters.
        fc (float): Carrier frequency in Hz.
    Returns:
        Path loss in dB.
    """
    return -35.4 + 20 * np.log10(d) + 20 * np.log10(fc/10**6)


def rayleigh_fading():
    """
    Simulates Rayleigh fading and returns the fading factor in linear scale.
    """
    return np.random.rayleigh(scale=1.0)

def db_to_linear(db):
    """Convert dB value to linear scale."""
    return 10 ** (db / 10)

def linear_to_db(linear):
    """Convert linear value to dB scale."""
    return 10 * np.log10(linear)



def get_spectral_effciency(snr_db, rat_type):
    """
    Maps SNR (in dB) to spectral efficiency based on RAT type.
    Parameters:
        snr_db (float): Signal-to-noise ratio in dB.
        rat_type (int): 0 for LTE, 1 for WiFi.
    Returns:
        Spectral efficiency in bits/s/Hz.
    """
    snr = snr_db
    if rat_type == 0:  # LTE (based on 3GPP CQI Table)
        cqi_table = [
            (0.0, 0.1523), (0.5, 0.2344), (1.0, 0.3770), (2.0, 0.6016),
            (3.0, 0.8770), (4.0, 1.1758), (5.0, 1.4766), (6.0, 1.9141),
            (7.0, 2.4063), (8.0, 2.7305), (9.0, 3.3223), (10.0, 3.9023),
            (11.0, 4.5234), (12.0, 5.1152), (13.0, 5.5547)
        ]
        for threshold, efficiency in reversed(cqi_table):
            if snr >= threshold:
                return efficiency
        return 0.0

    else:  # WiFi (based on IEEE 802.11ac table)
        wifi_table = [
            (-3.83, 0.5), (0.0, 1), (2.62, 1.5), (4.77, 2), (8.45, 3),
            (11.67, 4), (13.35, 4.5), (14.91, 5), (17.99, 6)
        ]
        for threshold, efficiency in reversed(wifi_table):
            if snr >= threshold:
                return efficiency
        return 0.0


def get_rate(user_position, station_position, rat_type):
    """
    Computes the achievable data rate based on user and station positions and RAT type.
    Considers path loss, Rayleigh fading, noise, and bandwidth.
    Parameters:
        user_position (array-like): Coordinates of the user.
        station_position (array-like): Coordinates of the base station or AP.
        rat_type (int): 0 for LTE, 1 for WiFi.
    Returns:
        Achievable rate in Mbps. Returns -1 if rate is zero or user is out of WiFi range.
    """
    # Parameters
    bw_lte = 20e6 # Hz
    fc_lte = 2.6e9 # Hz
    bw_wifi = 80e6 # Hz
    fc_wifi = 5.8e9 # Hz
    tx_power_watt = 1  # W
    tx_power_dbm = 10 * np.log10(tx_power_watt * 1e3)  # dBm
    noise_density_dbm_hz = -120  # dBm/Hz (gaussian noise)
    
    # Select RAT-specific parameters
    if rat_type == 0:
        fc = fc_lte
        bw = bw_lte
    else:
        fc = fc_wifi
        bw = bw_wifi

    # Distance calculation
    d = np.linalg.norm(np.array(user_position) - np.array(station_position))
    if d < 1:
        d = 1  # Avoid log(0)

    # Path loss + fading
    pl = path_loss(d, fc)
    fading = rayleigh_fading()
    rx_power_dbm = tx_power_dbm - pl + linear_to_db(fading)
    # Noise power
    noise_power_dbm = noise_density_dbm_hz + 10 * np.log10(bw)
    # SNR calculation
    snr_db = rx_power_dbm - noise_power_dbm
    # Spectral efficiency
    spectral_eff = get_spectral_effciency(snr_db, rat_type)

    # Data rate = efficiency * bandwidth
    rate = spectral_eff * bw /1e6 # Convert to Mbps
    
    if rate==0 or (rat_type == 1 and d > 12):
        rate = -1 # For the learning to work better

    return rate
