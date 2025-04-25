#── file: oscillators.eq ────────────────────────────────────────────────────
# 2–degree‐of‐freedom forced nonlinear oscillator, converted to 1st‐order form

# 1) state‐variables (2N = 4):
var q1, q2, p1, p2

# 2) parameters:
par c1, c2,
    k11, k12,
    k21, k22,
    alpha111, alpha112, alpha121, alpha122,
    alpha211, alpha212, alpha221, alpha222,
    gamma1111, gamma1112, gamma1121, gamma1122,
    gamma1211, gamma1212, gamma1221, gamma1222,
    gamma2111, gamma2112, gamma2121, gamma2122,
    gamma2211, gamma2212, gamma2221, gamma2222,
    f1, f2,
    omega

# 3) ODEs:
q1' = p1
q2' = p2

p1' = c1*p1
     - k11*q1 - k12*q2
     - ( alpha111*q1*q1 + alpha112*q1*q2
       + alpha121*q2*q1 + alpha122*q2*q2 )
     - ( gamma1111*q1*q1*q1 + gamma1112*q1*q1*q2
       + gamma1121*q1*q2*q1 + gamma1122*q1*q2*q2
       + gamma1211*q2*q1*q1 + gamma1212*q2*q1*q2
       + gamma1221*q2*q2*q1 + gamma1222*q2*q2*q2 )
     + f1*cos(omega*t)

p2' = c2*p2
     - k21*q1 - k22*q2
     - ( alpha211*q1*q1 + alpha212*q1*q2
       + alpha221*q2*q1 + alpha222*q2*q2 )
     - ( gamma2111*q1*q1*q1 + gamma2112*q1*q1*q2
       + gamma2121*q1*q2*q1 + gamma2122*q1*q2*q2
       + gamma2211*q2*q1*q1 + gamma2212*q2*q1*q2
       + gamma2221*q2*q2*q1 + gamma2222*q2*q2*q2 )
     + f2*cos(omega*t)

# 4) terminator
done
