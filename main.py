"""
Code by: Radoslaw Kamil Izak, AeroTrim Tool
if using input, use variable = input("NAME")
if converting from deg to rad, use VALUE/57.3 OR np.radians(VARIABLE)
if converting from ft to m, use VARIABLE/3.281
if converting from kt to m/s, use VARIABLE*0.515
"""
import math
import sys
import numpy as np
import sympy as sym
import pylint
import matplotlib.pyplot as plt
from tabulate import tabulate

# STEP 1

ALT, MASS, H, GAMMA_E = 300, 10.0, 0.345, 0
L_0, L_11, R, P_0 = -0.0065, 0, 287.05, 101325
RHO_0, T_0, A_0, G_0 = 1.225, 288.15, 340.294, 9.80665

""""
Altitude (ft) || Mass (kg) || CG Position (% of chord) || Flight Path Angle (deg)
LR for tropo.|| LR for low. strato. || Gas Const (J/kgK) || Press. (Pa)
Density (kg/m^3) || Temp. (K) || SoS (m/s) || Gravitational Const. (m/s^2)
"""

# STEP 2

if 0 <= ALT <= 36089:  # troposphere

    T = T_0 + L_0 * ((ALT / 3.281) - 0)  # in K
    P = P_0 * (1 + (L_0 / T_0) * (ALT / 3.281)) ** (-G_0 / (R * L_0))
    RHO = P / (R * T)
    SIGMA = RHO / RHO_0

elif 36089 <= ALT <= 65617:  # lower stratosphere

    T = T_0 + L_0 * ((36089 / 3.281) - 0)  # in K
    P_11 = P_0 * (1 + (L_0 / T_0) * (36089 / 3.281)) ** (-G_0 / (R * L_0))
    P = P_11 * math.exp((-G_0 / (R * T)) * ((ALT / 3.281) - 36089))
    RHO = P / (R * T)
    SIGMA = RHO / RHO_0

else:

    print("Altitude Value outside of the ISA model, please try again :)")
    sys.exit()

# STEP 3

STEPS, INCR, ST_VEL = 10, 5, 25  # in kt
V = np.zeros(STEPS + 1, dtype=int)

for NX in range(0, STEPS + 1):
    V[NX] = ST_VEL + NX * INCR  # in kt

V_I = V * 0.515  # in m/s
V_EAS = V * math.sqrt(SIGMA)  # in kt

# STEP 4

# WING GEOMETRY
S, B, C_W, LAMB = 0.6618, 3.0, 0.223, 2.0
Z_W, ALPHA_WR = 0.05, 6.4

"""
Wing area in (m^2) || Wingspan (m) || Wing mean chord (m) || sweep 1/4c_w (deg)
z coord. of 1/4cw up/down -(−ve)/(+ve) ox body (m) || Wing rigging angle (deg)
"""

# TAILPLANE GEOMETRY
S_T, B_T, L_T_SMALL = 0.0811, 0.62, 1.05
Z_T, N_T = -0.16, 3
F_D = 0.21

"""
Tailplane area in (m^2) || Tailplane span (m) || Tail arm, 1/4cw to 1/4ct (m))
z coord. of 1/4cw up/down -(−ve)/(+ve) ox body (m) || Tail setting angle (deg))
Fuselage diameter or width (m))
"""

# ENGINE INSTALLATION
Z_TAU, KAPPA = 0.025, 5

"""
Thrust line z coordinate above(−ve) or below(+ve) ox body axis in m
Engine thrust line angle (deg) relative to ox body axis (+nose up) in deg
"""

# STEP 5

A, C_L_MAX, C_M_0 = 5.4776, 1.27, -0.05
C_D_0, ALPHA_W0, H_0 = 0.02, -2.0, 0.08

"""
Wing-body CL-α (per rad) || Maximum lift coefficient || Zero lift pitching moment
Zero lift drag coefficient || Zero lift angle of attack (deg) || Wing-body aero centre
"""

# STEP 6

A_1, A_2, EPSILON_0 = 4.4187, 2.65, 2.0

"""
Tailplane CL-α (per rad) || Elevator CL-η (per rad) || Zero lift down-wash angle (deg)
"""

# STEP 7

AR = B ** 2 / S
S_SMALL = B / 2
L_T_LARGE = L_T_SMALL - C_W * (H - 0.25)
V_T = ((S_T * L_T_LARGE) / (S * C_W))

"""
Aspect Ratio
Wing Semi-span (m)
Tail arm, cg to 1/4ct (m)
Tail volume
"""

# STEP 8

X = L_T_SMALL / B
Z = (Z_W - Z_T) / B
D_EPS_ALPHA = 0

for FI in range(5, 176):
    CVS = (A / (math.pi ** 2 * AR)) \
          * (0.5 * math.cos((FI * math.pi) / 180)) ** 2 \
          / math.sqrt(X ** 2 + (0.5 * math.cos((FI * math.pi) / 180)) ** 2
                      + Z ** 2) * (math.pi / 180) * \
          (((X + math.sqrt(X ** 2 + (0.5 * math.cos((FI * math.pi) / 180)) ** 2 + Z ** 2)) /
            ((0.5 * math.cos((FI * math.pi) / 180)) ** 2 + Z ** 2))
           + (X / (X ** 2 + Z ** 2)))
    D_EPS_ALPHA += CVS

# STEP 9

S_D = 0.9998 + 0.0421 * (F_D / B) - 2.6286 * (F_D / B) ** 2 + 2 * (F_D / B) ** 3
K_D = -3.333 * 1e-4 * LAMB ** 2 + 6.667 * 1e-5 * LAMB + 0.38
E = 1 / (math.pi * AR * K_D * C_D_0 + (1 / (0.99 * S_D)))
K = 1 / (math.pi * AR * E)

"""
Fuselage drag factor
Empirical constant
Oswald efficiency factor
Induced drag factor
"""

# STEP 10
# calculate standard performance and stability parameters

V_MD = (math.sqrt((2 * MASS * G_0) / (RHO * S)) * (K / C_D_0) ** 0.25) * (1 / 0.515)
V_MD_EAS = V_MD * math.sqrt(SIGMA)
V_STALL = (math.sqrt((2 * MASS * G_0) / (RHO * S * C_L_MAX))) * (1 / 0.515)
V_STALL_EAS = V_STALL * math.sqrt(SIGMA)
H_N = H_0 + V_T * (A_1 / A) * (1 - D_EPS_ALPHA)
K_N = H_N - H

"""
Minimum drag speed (knots)
Equivalent minimum drag speed (knots)
Stall speed (knots)
Equivalent stall speed (knots)
Neutral point - controls fixed
Static margin - controls fixed
"""

# STEP 11 + 12

C_L_SEED, C_LW_SEED, C_D_SEED = 0.7, 0.5, 0.02
C_TAU_SEED, ALPHA_E_SEED, C_LT_SEED = 0.4, 0.1, 0.1

ALPHA_E = np.zeros(11, dtype=float)
C_TAU = np.zeros(11, dtype=float)
C_D = np.zeros(11, dtype=float)
C_LT = np.zeros(11, dtype=float)
C_LW = np.zeros(11, dtype=float)
C_L = np.zeros(11, dtype=float)

for NX in range(0, STEPS + 1):
    ALPHA_E_I, C_TAU_I, C_D_I, C_LT_I, C_LW_I, C_L_I = \
        sym.symbols('ALPHA_E_I, C_TAU_I, C_D_I, C_LT_I, C_LW_I, C_L_I')

    EQ_1 = sym.Eq(2 * ((MASS * G_0) / (RHO * V_I[NX] ** 2 * S)) *
                  sym.sin(ALPHA_E_I + np.radians(GAMMA_E)),
                  C_TAU_I * sym.cos(np.radians(KAPPA)) - C_D_I *
                  sym.cos(ALPHA_E_I) + C_L_I * sym.sin(ALPHA_E_I))

    EQ_2 = sym.Eq(2 * ((MASS * G_0) / (RHO * V_I[NX] ** 2 * S)) *
                  sym.cos(ALPHA_E_I + np.radians(GAMMA_E)),
                  C_L_I * sym.cos(ALPHA_E_I) + C_D_I * sym.sin(ALPHA_E_I)
                  + C_TAU_I * sym.sin(np.radians(KAPPA)))

    EQ_3 = sym.Eq(0, (C_M_0 + (H - H_0) * C_LW_I) - V_T * C_LT_I
                  + C_TAU_I * Z_TAU / C_W)

    EQ_4 = sym.Eq(C_L_I, C_LW_I + C_LT_I * S_T / S)

    EQ_5 = sym.Eq(C_D_I, C_D_0 + K * C_L_I ** 2)

    EQ_6 = sym.Eq(C_LW_I, A * (ALPHA_E_I + np.radians(ALPHA_WR) - np.radians(ALPHA_W0)))

    RESULT = sym.nsolve((EQ_1, EQ_2, EQ_3, EQ_4, EQ_5, EQ_6),
                        (ALPHA_E_I, C_TAU_I, C_D_I, C_LT_I, C_LW_I, C_L_I),
                        (ALPHA_E_SEED, C_TAU_SEED, C_D_SEED, C_LT_SEED, C_LW_SEED, C_L_SEED))

    RESULT = np.array(RESULT).astype(np.float64)

    ALPHA_E[NX], C_TAU[NX], C_D[NX] = RESULT[0, 0], RESULT[1, 0], RESULT[2, 0]
    C_LT[NX], C_LW[NX], C_L[NX] = RESULT[3, 0], RESULT[4, 0], RESULT[5, 0]

ALPHA_W = ALPHA_E + np.radians(ALPHA_WR)

N_E = (C_LT / A_2) - (A_1 / A_2) * \
      (ALPHA_W * (1 - D_EPS_ALPHA) + np.radians(N_T) -
       np.radians(ALPHA_WR) - np.radians(EPSILON_0))

THETA_E = np.radians(GAMMA_E) + ALPHA_W - np.radians(ALPHA_WR)

ALPHA_T = ALPHA_W * (1 - D_EPS_ALPHA) + np.radians(N_T) - \
          np.radians(EPSILON_0) - np.radians(ALPHA_WR)

L_D = (C_LW / C_D)

"""
Wing incidence
Trim elevator angle
Pitch attitude
Tail angle of attack
Lift to drag ratio
"""

# STEP 13

ALPHA_W, ALPHA_E, THETA_E = np.rad2deg(ALPHA_W), np.rad2deg(ALPHA_E), np.rad2deg(THETA_E)
ALPHA_T, N_E, GAMMA_E = np.rad2deg(ALPHA_T), np.rad2deg(N_E), np.rad2deg(GAMMA_E)

# STEP 14

L, D, T = np.zeros(11, dtype=float), np.zeros(11, dtype=float), np.zeros(11, dtype=float)

for NX in range(0, STEPS + 1):
    L[NX] = 0.5 * RHO * V_I[NX] ** 2 * S * C_L[NX]
    D[NX] = 0.5 * RHO * V_I[NX] ** 2 * S * C_D[NX]
    T[NX] = 0.5 * RHO * V_I[NX] ** 2 * S * C_TAU[NX]

# STEP 15

DATA_1 = [["Aircraft Weight (N) ", str(MASS * G_0), "Minimum Drag Speed (kt)", str(V_MD)],
          ["Altitude (ft) ", str(ALT), "Minimum Drag Speed (kt)", str(V_MD)],
          ["Flight path angle (deg) ", str(GAMMA_E), "Stall speed (knots) (kt) ", str(V_STALL)],
          ["cg position (%cw) ", str(H), "Equivalent Stall speed (knots) (kt) ",
           str(V_STALL_EAS)], ["Neutral point - controls fixed ",
                               str(H_N), "Static margin - controls fixed ", str(K_N)]]

HEAD_1 = ["Name", "Value", "Name", "Value"]
print(tabulate(DATA_1, headers=HEAD_1, tablefmt="grid"))

# STEP 15
# Angles in degrees, velocity in m/s, forces in N except where indicated otherwise


DATA_2 = [[(str(V.reshape(11, 1))), (str(V_I.reshape(11, 1))), (str(C_L.reshape(11, 1))),
           (str(C_D.reshape(11, 1))), (str(C_LW.reshape(11, 1))), (str(C_LT.reshape(11, 1))),
           (str(L_D.reshape(11, 1))), (str(C_TAU.reshape(11, 1))), (str(ALPHA_W.reshape(11, 1))),
           (str(ALPHA_E.reshape(11, 1))), (str(THETA_E.reshape(11, 1))),
           (str(ALPHA_T.reshape(11, 1))), (str(N_E.reshape(11, 1))),
           (str(L.reshape(11, 1))), (str(D.reshape(11, 1))), (str(T.reshape(11, 1)))]]

HEAD_2 = ["V_knots_i", "V_i", "CL_i", "CD_i", "CLw_i", "CLT_i", "LD_i", "Cτ_i",
          "αw_i", "αe_i", "θe_i", "αT_i", "ηe_i", "L_i", "D_i", "T_i"]

print(tabulate(DATA_2, headers=HEAD_2, tablefmt="grid"))

# STEP 17

plt.figure(1)
plt.plot(V, L_D)
plt.ylabel('LDi')
plt.xlabel('V_i [kt]')
plt.axis([min(V) - 0.1 * min(V), max(V) + 0.1 * max(V),
          min(L_D) - 0.1 * min(L_D), max(L_D) + 0.1 * max(L_D)])
plt.grid(True)

plt.figure(2)
plt.plot(V, N_E)
plt.ylabel('N_Ei')
plt.xlabel('V_i [kt]')
plt.axis([min(V) - 0.1 * min(V), max(V) + 0.1 * max(V),
          min(N_E) - 0.1 * min(N_E), max(N_E) + 0.1 * max(N_E)])
plt.grid(True)

plt.figure(3)
plt.plot(V, D)
plt.ylabel('D_i')
plt.xlabel('V_i [kt]')
plt.axis([min(V) - 0.1 * min(V), max(V) + 0.1 * max(V),
          min(D) - 0.1 * min(D), max(D) + 0.1 * max(D)])
plt.grid(True)

plt.figure(4)
plt.plot(C_D, C_L)
plt.ylabel('C_Li')
plt.xlabel('C_Di')
plt.axis([min(C_D) - 0.1 * min(C_D), max(C_D) + 0.1 * max(C_D),
          min(C_L) - 0.1 * min(C_L), max(C_L) + 0.1 * max(C_L)])
plt.grid(True)

plt.show()

print(K_N)

sys.argv = ["pylint", "main"]
pylint.run_pylint()
