from abc import ABC, abstractmethod

class AbstractMultistart(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def to_dtype(self, dtype):
        pass

'''
To determine grid of initial conditions:
- Calculate linear resonance peak for each frequency and create a grid around that peak thus for every frequency
- Calulate linear resonance peak and and create grid with that peak (-peak, +peak, n_points)

To determine initial conditions of segment shooting method:
- Warm start with that initial condition and divide the time interval into m segments
'''