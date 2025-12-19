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
        sim_width = len(str(total_groups - 1)) if total_groups > 1 else 1


        sims_out = f_out.create_group("simulations")

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
                out_index = orig_i * n_forces + k
                sim_id = f"simulation_{out_index:0{sim_width}d}"
                gout = sims_out.create_group(sim_id)

                # Store 1D sweeps per force
                ds_fwd = gout.create_dataset(
                    "forward_sweep",
                    data=fwd[:, k],
                    compression=comp,
                    compression_opts=comp_opts,
                    shuffle=True if comp in ("gzip", "lzf") else False,
                )
                ds_bwd = gout.create_dataset(
                    "backward_sweep",
                    data=bwd[:, k],
                    compression=comp,
                    compression_opts=comp_opts,
                    shuffle=True if comp in ("gzip", "lzf") else False,
                )

                # Unsweeped per force
                gout.create_dataset("unsweeped_total", data=np.asarray(x_total[k]))
                gout.create_dataset("unsweeped_modes", data=np.asarray(x_modes[k, ...]))

                # Group attrs (rest stays same; f_amp is scalar now)
                gout.attrs["f_omegas"] = f_omegas
                gout.attrs["f_amp"] = np.asarray(f_amps[k, ...]) 
                gout.attrs["Q"] = Q
                gout.attrs["omega_0"] = omega_0
                gout.attrs["gamma"] = gamma
                gout.attrs["alpha"] = alpha
                gout.attrs["modal_forces"] = modal_forces
                gout.attrs["success_rate"] = success_rate

                # Forward/backward dataset attrs, sliced appropriately
                ds_fwd.attrs["ref_idx"] = ref_idx
                ds_bwd.attrs["ref_idx"] = ref_idx

                ds_fwd.attrs["reference_frequency"] = omega_ref
                ds_bwd.attrs["reference_frequency"] = omega_ref

                ds_fwd.attrs["reference_displacement"] = float(x_ref_forward[k])
                ds_bwd.attrs["reference_displacement"] = float(x_ref_backward[k])

                ds_fwd.attrs["scaled_omega_0"] = scaled_omega_0
                ds_bwd.attrs["scaled_omega_0"] = scaled_omega_0

                ds_fwd.attrs["scaled_gamma"] = scaled_gamma_forward[k, ...]
                ds_bwd.attrs["scaled_gamma"] = scaled_gamma_backward[k, ...]

                ds_fwd.attrs["scaled_alpha"] = scaled_alpha_forward[k, ...]
                ds_bwd.attrs["scaled_alpha"] = scaled_alpha_backward[k, ...]

                ds_fwd.attrs["scaled_f_omegas"] = scaled_f_omegas
                ds_bwd.attrs["scaled_f_omegas"] = scaled_f_omegas

                ds_fwd.attrs["scaled_f_amp"] = np.asarray(scaled_f_amps_forward[k, ...])
                ds_bwd.attrs["scaled_f_amp"] = np.asarray(scaled_f_amps_backward[k, ...])

    print(f"Done. Wrote expanded file: {args.output}")


if __name__ == "__main__":
    main()
