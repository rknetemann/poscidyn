import numpy as np

def get_data():

    # Load nonlinear sweep data
    file = f"tests/chris/16_40_15-LDV_100-nonlinear_sweep2_data.txt"
    data = np.loadtxt(file, delimiter=",", skiprows=1)  # Skip the header row
    freq = data[:, 0]
    mag = data[:, 1]
    phase = data[:, 2]

    def measurement2meter(freq, mag, setup, *args):
        if setup == "ENIGMA with correction":
            # ENIGMA - using transduction
            transduction = args[0]
            delta = mag / transduction * 1e-9
            gamma = 4*np.pi / 632.8e-9  # 4 pi / lambda_red
            lin_error = 1/8*delta*2 * gamma*2
            return delta * (1 - lin_error)

        elif setup == "ENIGMA":
            # ENIGMA - using transduction
            transduction = args[0]
            return mag / transduction * 1e-9

        elif setup == "Polytec":
            # Polytec - using LDV settings
            LDV = args[0]
            return 2 * mag * LDV / (2 * np.pi * freq)

        else:
            print(f'Setup should be "ENIGMA" or "ENIGMA with correction" or "Polytec", not "{setup}"')

    settings = "LDV_100"
    setup = "Polytec"; calibration = float(settings.split('_')[-1]) * 1e-3
    total_points = 11; picked_point = 2

    x_ref = 1e-8
    omega_ref = 207.708e3

    freq_sweep = freq
    modeshape_coeff1 = np.sin((picked_point - 1) / (total_points - 1) * np.pi)
    mag_sweep = measurement2meter(freq_sweep, mag, setup, calibration) / modeshape_coeff1

    freq_sweep = freq_sweep / omega_ref
    mag_sweep = mag_sweep / x_ref

    return freq_sweep, mag_sweep

if __name__ == "__main__":
    freq_sweep, mag_sweep = get_data()

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(freq_sweep, mag_sweep, marker='o', linestyle='-')
    plt.xlabel('ω/ω₀')
    plt.ylabel('x/$x_{ref}$')
    plt.title('Magnitude Sweep')
    plt.grid(True)
    plt.show()