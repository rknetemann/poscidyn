import argparse
import math
import h5py
import numpy as np


def copy_root_attrs(src: h5py.File, dst: h5py.File):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to existing HDF5 produced by your script")
    p.add_argument("--output", required=True, help="Path to NEW expanded HDF5")
    p.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    p.add_argument("--gzip_level", type=int, default=4)
    args = p.parse_args()

    comp = None if args.compression == "none" else args.compression
    comp_opts = None
    if comp == "gzip":
        comp_opts = args.gzip_level

    with h5py.File(args.input, "r") as f_in, h5py.File(args.output, "w") as f_out:
        # Copy root attrs and params
        copy_root_attrs(f_in, f_out)

        if "params" in f_in:
            f_out.create_dataset("params", data=f_in["params"][...])
        else:
            raise KeyError("Input file does not contain root dataset 'params'.")

        sims_in = f_in["simulations"]
        sim_keys = sorted(sims_in.keys())  # simulation_00000 ...
        n_sim = len(sim_keys)
        if n_sim == 0:
            raise ValueError("No simulations found in input file under '/simulations'.")

        # Determine n_forces from the first simulation group
        g0 = sims_in[sim_keys[0]]
        f_amps = np.asarray(g0.attrs["f_amps"])
        n_forces = int(f_amps.shape[0])

        total_groups = n_sim * n_forces

        # Create root-level folders
        forward_group = f_out.create_group("forward_sweeps")
        backward_group = f_out.create_group("backward_sweeps")
        unsweeped_modes_group = f_out.create_group("unsweeped_modes")
        unsweeped_total_group = f_out.create_group("unsweeped_total")

        # Root-level stats updated to reflect expansion
        f_out.attrs["n_simulations_original"] = n_sim
        f_out.attrs["n_forces_per_simulation"] = n_forces
        f_out.attrs["n_simulations_expanded"] = total_groups

        # Convert each simulation group into n_forces groups
        for orig_i, key in enumerate(sim_keys):
            gin = sims_in[key]

            # Read common attrs
            f_omegas = np.asarray(gin.attrs["f_omegas"])
            f_amps = np.asarray(gin.attrs["f_amps"])
            Q = np.asarray(gin.attrs["Q"])
            omega_0 = np.asarray(gin.attrs["omega_0"])
            gamma = np.asarray(gin.attrs["gamma"])
            alpha = np.asarray(gin.attrs["alpha"])
            modal_forces = np.asarray(gin.attrs["modal_forces"])
            success_rate = gin.attrs.get("success_rate", np.nan)

            # Read datasets (these were stored as full (freq, force) in your original layout)
            fwd = np.asarray(gin["forward_sweep"][...])     # (n_freq, n_forces) expected
            bwd = np.asarray(gin["backward_sweep"][...])    # (n_freq, n_forces) expected

            x_total = np.asarray(gin["unsweeped_total"][...])  # (n_forces,) expected
            x_modes = np.asarray(gin["unsweeped_modes"][...])  # (n_forces, n_modes) expected

            # Dataset attrs that contain force-dependent arrays we need to slice
            ref_idx = int(gin["forward_sweep"].attrs["ref_idx"])
            omega_ref = float(gin["forward_sweep"].attrs["reference_frequency"])

            x_ref_forward = np.asarray(gin["forward_sweep"].attrs["reference_displacement"])   # (n_forces,) expected
            x_ref_backward = np.asarray(gin["backward_sweep"].attrs["reference_displacement"]) # (n_forces,) expected

            scaled_omega_0 = np.asarray(gin["forward_sweep"].attrs["scaled_omega_0"])  # (2,) expected
            scaled_omega_0_fixed = scaled_omega_0 / omega_0[0] # Made a mistake in data genration, fix here

            scaled_gamma_forward = np.asarray(gin["forward_sweep"].attrs["scaled_gamma"])    # (n_forces, 2,2,2,2)
            scaled_gamma_backward = np.asarray(gin["backward_sweep"].attrs["scaled_gamma"])  # (n_forces, 2,2,2,2)

            scaled_alpha_forward = np.asarray(gin["forward_sweep"].attrs["scaled_alpha"])    # (n_forces, 2,2,2)
            scaled_alpha_backward = np.asarray(gin["backward_sweep"].attrs["scaled_alpha"])  # (n_forces, 2,2,2)

            scaled_f_omegas = np.asarray(gin["forward_sweep"].attrs["scaled_f_omegas"])  # (n_freq,) expected

            # In your original file this is (n_forces, n_forces). We will store a scalar for each group.
            scaled_f_amps_forward = np.asarray(gin["forward_sweep"].attrs["scaled_f_amps"])
            scaled_f_amps_backward = np.asarray(gin["backward_sweep"].attrs["scaled_f_amps"])

            # Write expanded groups
            for k in range(n_forces):
                out_index = orig_i * n_forces + k + 1  # 1-based indexing in output folders
                sim_id = f"simulation_{out_index}"

                # Store 1D sweeps per force into their own root-level folders
                ds_fwd = forward_group.create_dataset(
                    sim_id,
                    data=fwd[:, k],
                    compression=comp,
                    compression_opts=comp_opts,
                    shuffle=True if comp in ("gzip", "lzf") else False,
                )
                ds_bwd = backward_group.create_dataset(
                    sim_id,
                    data=bwd[:, k],
                    compression=comp,
                    compression_opts=comp_opts,
                    shuffle=True if comp in ("gzip", "lzf") else False,
                )

                # Unsweeped data per force
                ds_unsweeped_total = unsweeped_total_group.create_dataset(sim_id, data=np.asarray(x_total[k]))
                ds_unsweeped_modes = unsweeped_modes_group.create_dataset(sim_id, data=np.asarray(x_modes[k, ...]))

                common_attrs = {
                    "f_omegas": f_omegas,
                    "f_amp": np.asarray(f_amps[k, ...]),
                    "Q": Q,
                    "omega_0": omega_0,
                    "gamma": gamma,
                    "alpha": alpha,
                    "modal_forces": modal_forces,
                    "success_rate": success_rate,
                    "scaled_omega_0": scaled_omega_0_fixed,
                    "scaled_f_omegas": scaled_f_omegas,
                }

                def set_attrs(ds, extra_attrs):
                    for name, value in {**common_attrs, **extra_attrs}.items():
                        ds.attrs[name] = value

                # Forward/backward dataset attrs, sliced appropriately
                set_attrs(
                    ds_fwd,
                    {
                        "ref_idx": ref_idx,
                        "ref_frequency": omega_ref,
                        "ref_displacement": float(x_ref_forward[k]),
                        "scaled_gamma": scaled_gamma_forward[k, ...],
                        "scaled_alpha": scaled_alpha_forward[k, ...],
                        "scaled_f_amp": np.asarray(scaled_f_amps_forward[k, ...]),
                    },
                )
                set_attrs(
                    ds_bwd,
                    {
                        "ref_idx": ref_idx,
                        "ref_frequency": omega_ref,
                        "ref_displacement": float(x_ref_backward[k]),
                        "scaled_gamma": scaled_gamma_backward[k, ...],
                        "scaled_alpha": scaled_alpha_backward[k, ...],
                        "scaled_f_amp": np.asarray(scaled_f_amps_backward[k, ...]),
                    },
                )

                # Unsweeped datasets
                set_attrs(
                    ds_unsweeped_total,
                    {},
                )
                set_attrs(
                    ds_unsweeped_modes,
                    {},
                )

    print(f"Done. Wrote expanded file: {args.output}")


if __name__ == "__main__":
    main()
