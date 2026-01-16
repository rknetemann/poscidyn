from jax import tree_util

class FrequencySweepResult:
    def __init__(
        self,
        model,
        excitation,
        periodic_solutions,
        sweeped_periodic_solutions,
        n_successful,
        n_total,
        success_rate,
    ):
        self.model = model
        self.excitation = excitation
        self.periodic_solutions = periodic_solutions
        self.sweeped_periodic_solutions = sweeped_periodic_solutions
        self.n_successful = n_successful
        self.n_total = n_total
        self.success_rate = success_rate

    def keys(self):
        return (
            "model",
            "excitation",
            "periodic_solutions",
            "sweeped_periodic_solutions",
            "n_successful",
            "n_total",
            "success_rate",
        )


def _tree_flatten(obj: FrequencySweepResult):
    leaves = (
        obj.model,
        obj.excitation,
        obj.periodic_solutions,
        obj.sweeped_periodic_solutions,
        obj.n_successful,
        obj.n_total,
        obj.success_rate,
    )
    return leaves, None


def _tree_unflatten(aux_data, leaves):
    (
        model,
        excitation,
        periodic_solutions,
        sweeped_periodic_solutions,
        n_successful,
        n_total,
        success_rate,
    ) = leaves

    return FrequencySweepResult(
        model=model,
        excitation=excitation,
        periodic_solutions=periodic_solutions,
        sweeped_periodic_solutions=sweeped_periodic_solutions,
        n_successful=n_successful,
        n_total=n_total,
        success_rate=success_rate,
    )


tree_util.register_pytree_node(FrequencySweepResult, _tree_flatten, _tree_unflatten)
