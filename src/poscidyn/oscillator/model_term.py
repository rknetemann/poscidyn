from abc import abstractmethod, ABC

class OscillatorTermCoefficient:
    @abstractmethod
    def constant_value(self) -> float:
        pass

    @abstractmethod
    def modulated_value(self, t, args) -> float:
        pass

class OscillatorTermState:
    @abstractmethod
    def value(self, t, state, args) -> float:
        pass

class OscillatorTerm(ABC):
    coefficient: OscillatorTermCoefficient
    state: OscillatorTermState

    def __init__(self):
        pass
    
    @abstractmethod
    def value(self, t, state, args) -> float:
        (self.coefficient.modulated_value(t, args) + self.coefficient.constant_value()) * self.state.value(t, state, args)

class Oscillator(ABC):
    terms: list[OscillatorTerm]

    coeff1: OscillatorTermCoefficient

    def __init__(self):
        pass
    
    @abstractmethod
    def rhs(self, t, state, args) -> float:
        d2xdt2 = OscillatorTerm(lambda: self.coeff1, )


MODEL = poscidyn.NonlinearOscillator(Q=ModulatedParameter(Q, alpha=alpha, gamma=gamma, omega_0=omega_0)