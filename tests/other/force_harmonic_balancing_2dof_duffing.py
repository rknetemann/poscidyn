import math

def f(Qx, omega, omega0x, omega0y, alpha, eta1, eta2, gamma):
    radicand = (
        16 * Qx**2 * omega**4
        - 32 * Qx**2 * omega**2 * omega0x**2
        - 16 * Qx**2 * alpha**2 * eta1 * omega**2 * omega0x**2
        + 16 * Qx**2 * omega0x**4
        + 16 * Qx**2 * alpha**2 * eta1 * omega0x**4
        + 4 * Qx**2 * alpha**4 * eta1**2 * omega0x**4
        - (12 * math.sqrt(2) * Qx**2 * gamma * math.sqrt(eta1 * eta2)
           * omega**2 * omega0x * omega0y) / alpha
        + (12 * math.sqrt(2) * Qx**2 * gamma * math.sqrt(eta1 * eta2)
           * omega0x**3 * omega0y) / alpha
        + 6 * math.sqrt(2) * Qx**2 * alpha * gamma * eta1 \
          * math.sqrt(eta1 * eta2) * omega0x**3 * omega0y
        + (9 * Qx**2 * gamma**2 * eta1 * eta2 * omega0x**2 * omega0y**2)
          / (2 * alpha**2)
        + (8 * eta1 * eta2 * omega0x**6 * omega0y**2) / alpha**2
    )

    return (1.0 / (4.0 * Qx)) * math.sqrt(radicand)

print(f(10, 1.0, 1.0, 1.5, 3.74e-01, 1.0, 1.0, 2.67e-02))
