import control
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Función escalón para representar los caudales de entrada
def f(t, t0, t1, a):
    if t >= t0 and t1 > t:
        return a
    else: return 0

def nolineal(t, y):

    # Definimos parámetros del sistema
    t0 = 0  # Tiempo inicial en minutos
    t1 = 0.25  # Tiempo final en minutos
    k12 = 20*np.sqrt(2)
    k2 = 20*np.sqrt(2)
    A1 = 0.5 # En [m^2]
    A2 = 0.25 # En [m^2]
    q1 = 2 # En [m^3/min]
    q2 = 0 # En [m^3/min]


    # Definimos entradas
    Q1 = f(t, t0, t1, q1) # En [m^3/min]
    Q2 = f(t, t0, t1, q2) # En [m^3/min]

    # Igualamos variables a condiciones iniciales
    h1 = y[0]
    h2 = y[1]

    return [(1/A1) * (Q1 - k12 * np.sqrt(h1 - h2)), (1/A2) * (Q2 + k12 * np.sqrt(h1 - h2) - k2 * np.sqrt(h2))]

def linealizadoydesviado(t, y):

    # Definimos parámetros del sistema
    t0 = 0  # Tiempo inicial en minutos
    t1 = 0.25  # Tiempo final en minutos
    R12 = 1/200 # En [min/m^2]
    R2 = 1/200 # En [min/m^2]
    A1 = 0.5 # En [m^2]
    A2 = 0.25 # En [min/m^2]
    q1 = 2 # En [m^3/min]
    q2 = 0 # En [m^3/min]

    # Definimos entradas
    Q1 = f(t, t0, t1, q1)
    Q2 = f(t, t0, t1, q2)

    # Desviamos entrada
    Q1d = Q1 - 2
    Q2d = Q2 - 0

    # Igualamos variables a condiciones iniciales
    h1 = y[0]
    h2 = y[1]

    return [(1/A1) * (Q1d - (h1 - h2)/R12), (1/A2) * (Q2d + (h1 - h2)/R12 - h2/R2)]


# Definimos arreglo de tiempo
t = np.linspace(0, 0.25, 1000)
t_span = (0,0.25)


# Definimos condiciones iniciales
cinolineal = [0, 0]
cidesviado = [-0.01, -0.005]

# Resolvemos
nolineal = solve_ivp(nolineal, t_span, cinolineal, method='RK45', t_eval=t)
lineal = solve_ivp(linealizadoydesviado, t_span, cidesviado, method='RK45', t_eval=t)

# Ploteamos resultado

#Imprimimos comparacion entre no lineal y lineal y desviado
plt.plot(nolineal.t, nolineal.y[0], label='h(1) no lineal', linewidth=2, color='red')
plt.plot(lineal.t, lineal.y[0]+0.01, label='h(2) linealizado', linewidth=2, color='green')
plt.ylabel('Altura del tanque [m]')
plt.xlabel('Tiempo [s]')
plt.title('Alturas de tanque 1 de ambos modelos para una entrada Q1=2 y Q2=0')
plt.legend()
plt.show()

plt.plot(nolineal.t, nolineal.y[1], label='h(2) no lineal', linewidth=2, color='red')
plt.plot(lineal.t, lineal.y[1]+0.005, label='h(2) linealizado', linewidth=2, color='green')
plt.ylabel('Altura del tanque [m]')
plt.xlabel('Tiempo [s]')
plt.title('Alturas de tanque 2 de ambos modelos para una entrada Q1=2 y Q2=0')
plt.legend()
plt.show()

# INCISO F
# Planteamos FT
ft = control.TransferFunction([200], [0.125, 250, 40000])

Q1 = 5
Q2 = 0

Q1desviado = 5-2
Q2desviado = 0-0

Q1final = Q1desviado - (-1)

tiempoft = np.arange(0, 0.25, 0.001)

Q1input = Q1final * np.heaviside(tiempoft,1)

T, RespuestaFT = control.forced_response(ft,tiempoft,Q1input)

# NOTA: PARA QUE LA SIMULACION SEA CORRECTA EN ESTE EJERCICIO, DEBE ACTUALIZARSE q1 = 5 EN LINEA 44

plt.plot(T, RespuestaFT, label='h(2) con FT', linewidth=2, color='red')
plt.plot(lineal.t, lineal.y[1]+0.005, label='h(2) linealizado', linewidth=2, color='green')
plt.ylabel('Altura del tanque [m]')
plt.xlabel('Tiempo [s]')
plt.title('Comparación de altura de tanque 2 entre FT y modelo lineal para una entrada Q1=5 y Q2=0')
plt.legend()
plt.show()

