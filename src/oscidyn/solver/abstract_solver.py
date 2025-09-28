import jax
import diffrax

from ..model.abstract_model import AbstractModel

class AbstractSolver:
    def __init__(self, rtol: float = 1e-4, atol: float = 1e-6, max_steps: int = 4096, progress_bar: bool = True):

        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self.progress_bar = progress_bar
    
    def solve(self, 
              model: AbstractModel,
              t0: float, 
              t1: float, 
              ts: jax.Array,
              y0: jax.Array,
              driving_frequency: float, 
              driving_amplitude: float, 
              ) -> diffrax.Solution:
                
        progress_meter = diffrax.TqdmProgressMeter() if self.progress_bar else diffrax.NoProgressMeter()
                
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model.rhs_jit),
            solver=diffrax.Tsit5(),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            t0=t0,
            t1=t1,
            dt0=None,
            max_steps=self.max_steps,
            y0=y0,
            throw=True,
            progress_meter=progress_meter,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol, pcoeff=0.0, icoeff=1.0, dcoeff=0.0),
            args=(driving_amplitude, driving_frequency),
        )
        return sol