import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Datos
data = pd.read_csv("T1_Housing.csv") 
X = data["Avg Area Income"]
Y = data["Price"]
# Entrenamiento.
xEnt = data[data["In Sample"] == 1]["Avg Area Income"]
yEnt = data[data["In Sample"] == 1]["Price"]
# Validación.
xVal = data[data["In Sample"] == 0]["Avg Area Income"]
yVal = data[data["In Sample"] == 0]["Price"]


def plotData():
    """
    Plotea los dato.
    """
    plt.plot(xEnt/10000, yEnt/100000, "*", label="Entrentamiento")
    plt.plot(xVal/10000, yVal/100000, "*", label="Validación")
    plt.legend(loc='upper left', shadow=True, fontsize='10')
    plt.title("Price vs average income")
    plt.xlabel("$Average\ income\ 10^4$")
    plt.ylabel("$Price\ 10^5$")
    plt.grid()
    plt.show()

def reg_lineal(X,Y,rho):
    """
    Implementa el método de mínimos cuadrados regularizados y retorna estas estimaciones 
    como un arreglo theta.
    param X: DataFrame de una columna.
    param Y: DataFrame de una columna.
    param rho: Double.
    output theta: Arreglo 1D. 
    """
    #Transforma los DataFrame en numpy arrays.
    X = np.array(X).reshape(len(X), 1)
    Y = np.array(Y).reshape(len(Y), 1)
    unos = np.ones(len(X)).reshape(len(X),1)
    xTilda = np.concatenate([X, unos], 1)
    I = np.identity(2)
    #Calcula theta.
    theta = np.linalg.inv(np.dot(xTilda.transpose(), xTilda) + rho * I)
    theta = np.dot(theta, np.dot(xTilda.transpose(), Y))
    return theta
    
def plotParam():
    """
    Usa datos de entrenamiento para obtener parámetros. Plotea los parámetros de mínimos cuadrados
    regularizados en función de rho.
    """
    p = 10 #Valores de rho.
    x = np.arange(p + 1)
    a = []
    b = []
    for rho in x:
        a.append(reg_lineal(xEnt, yEnt, rho)[0])
        b.append(reg_lineal(xEnt, yEnt, rho)[1])
    plt.plot(x, a, "*")
    plt.title(r"$Pendiente\ vs\ \rho$")
    plt.xlabel(r"$\rho$")
    plt.ylabel("$Pendiente$")
    plt.grid()
    plt.show()
    plt.plot(x, np.array(b)/100000, "*")
    plt.title(r'$Coeficiente\ de\ posición\ vs\ \rho$')
    plt.xlabel(r"$\rho$")
    plt.ylabel("$Coeficiente\ de\ posición\ 10^{5}$")
    plt.grid()
    plt.show()

def ecm(x,y,theta):
    """
    Calcula el error cuadrático medio entre theta^Tx e y.
    param y: 1D DataFrame.
    param x: 1D DataFrame. len(x) = len(y).
    param theta: 1D array con los parámetros del modelo.
    output ecm: Double.
    """
    a, b = theta
    x = np.array(x)
    y = np.array(y)
    ecm = (1/len(x)) * sum((y[i] - a*x[i] - b)**2 for i in np.arange(len(y)))
    return ecm

def plotEcmVar():
    """
    Plotea el ecm y la varianza de la estimación en función de rho.
    """
    p = 10
    x = np.arange(p + 1)
    # Se calcula y plotea el ecm asociado a cada rho para entrenamiento y validación.
    arrEcmEnt = []
    arrEcmVal = []
    for rho in x:
        arrEcmEnt.append(ecm(xEnt, yEnt, reg_lineal(xEnt, yEnt, rho)))
        arrEcmVal.append(ecm(xVal, yVal, reg_lineal(xEnt, yEnt, rho)))
    plt.plot(x, arrEcmEnt, "*", label="Entrenamiento")
    plt.plot(x, arrEcmVal, "*", label="Validación")
    plt.title(r"$Error\ cuadrático\ medio\ (ecm)\ vs\ \rho$")
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$ecm$")
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    # Se calcula y plotea el ecm asociado a cada rho.
    
def plotModelo():
    p = 11
    # Calcular parámetros con datos de entrenamiento.
    for rho in np.arange(p):
        a = reg_lineal(xEnt, yEnt, rho)[0] #Optimizar
        b = reg_lineal(xEnt, yEnt, rho)[1]
        yGorro = [] # Vector de predicciones.
        for x in xEnt:
            yGorro.append(a*x + b) 
        plt.plot(xEnt, yGorro)
    plt.plot(X, Y, "*") 
    plt.show()

#plotData()
#plotParam()
#plotEcmVar()
#plotModelo()
