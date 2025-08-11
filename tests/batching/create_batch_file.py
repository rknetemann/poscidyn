# tests/batching/create_batch_file.py
import numpy as np
import h5py

Q = np.linspace(1, 100.0, 100)  
gamma = np.linspace(0.01, 1.0, 100)  
sweep_direction = np.array([-1, 1])

Q, gamma, sweep_direction = np.meshgrid(Q, gamma, sweep_direction)
Q = Q.flatten()
gamma = gamma.flatten()
sweep_direction = sweep_direction.flatten()

driving_frequencies = np.linspace(0.1, 2.0, 200)
driving_amplitudes = np.linspace(0.01, 1.0, 10)

def create_batch_file(filename: str):   
    with h5py.File(filename, 'w') as f:
        n_sim = len(Q)
        
        f.attrs['n_simulations'] = n_sim
        f.attrs['driving_frequencies'] = driving_frequencies
        f.attrs['driving_amplitudes'] = driving_amplitudes
        
        for sim_index in range(n_sim):
            sim_width = len(str(n_sim))
            sim_id = f"simulation_{sim_index:0{sim_width-1}d}"
            grp = f.create_group(sim_id)
            grp.attrs['Q'] = Q[sim_index]
            grp.attrs['gamma'] = gamma[sim_index]
            grp.attrs['sweep_direction'] = sweep_direction[sim_index]

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Create a batch file for simulations.")
    parser.add_argument(
        'filename',
        nargs='?',
        type=str,
        default=None,   
        help="The name of the HDF5 file to create (optional). "
             "If omitted, current date and time will be used."
    )
    args = parser.parse_args()

    if args.filename:
        filename = args.filename
    else:
        filename = datetime.now().strftime("batch_%Y-%m-%d_%H:%M:%S.hdf5")

    create_batch_file(filename)
    print(f"Batch file '{filename}' created with {len(Q)} simulations.")