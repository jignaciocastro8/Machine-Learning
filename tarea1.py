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
    
def plotParam(flag = 1):
    """
    Usa datos de entrenamiento para obtener parámetros. Plotea los parámetros de mínimos cuadrados
    regularizados en función de rho.
    """

    p = 10 #Valores de rho.
    x = np.arange(p + 1)
    a = []
    b = []
    fig, ax = plt.subplots()
    for rho in x:
        aux = reg_lineal(xEnt, yEnt, rho)
        a.append(aux[0])
        b.append(aux[1])
        plt.plot(aux[0], aux[1]/100000, '*', label=r"$\rho = $" + str(rho))
    plt.title(r"$Parámetros\ en\ R^{2}$")
    plt.xlabel(r"$Pendiente$")
    plt.ylabel(r"$Coeficiente\ de\ posición\ 10^{5}$")
    plt.xlim([0, 50])
    plt.legend()
    plt.grid()
    plt.show()
    if flag != 1:
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

def plotEcm():
    """
    Plotea el ecm de la estimación en función de rho.
    """
    p = 10
    x = np.arange(p + 1)
    # Se calcula y plotea el ecm c/r a rho para entrenamiento y validación.
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
 
def plotVar():
    """
    Plotea la de la estimación en función de rho.
    """
    p = 10
    x = np.arange(p + 1)
    # Se calcula y plotea la varianza c/r a rho para entrenamiento y validación.
    arrVarEnt = []
    arrVarVal = []
    for rho in x:
        print("Hola")
    plt.plot(x, arrVarEnt, "*", label="Entrenamiento")
    plt.plot(x, arrVarVal, "*", label="Validación")
    plt.title(r"$Varianza\ vs\ \rho$")
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$Varianza$")
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    
def plotModelo():
    """
    Plotea las rectas asociadas a cada rho junto con los datos de validación y entrenamiento.
    """
    p = 11
    # Calcular parámetros con datos de entrenamiento.
    xMax = max(max(xEnt), max(xVal))
    xMin = min(min(xEnt), min(xVal))
    xArr = np.array([xMin, xMax]) # Como son rectas bastan dos puntos.
    for rho in np.arange(0, p, 2):
        a, b = reg_lineal(xEnt, yEnt, rho)
        yArr = [] # Vector de predicciones.
        for x in xArr:
            yArr.append(a*x + b) 
        plt.plot(xArr/10000, np.array(yArr)/100000, label=r"$\rho\ =$" + str(rho))
    plt.plot(xEnt/10000, yEnt/100000, '*', label=r"$Entrenamiento$")
    plt.plot(xVal/10000, yVal/100000, '*', label=r"$Validación$")
    plt.title(r"$Rectas\ del\ modelo\ para\ distintos\ valores\ de\ \rho$")
    plt.ylabel(r"$Price\ 10^{5}$")
    plt.xlabel(r"$Average\ income\ 10^{4}$")
    plt.legend()
    plt.grid()
    plt.show()

def estimarSigma(rho):
    a, b = reg_lineal(xEnt, yEnt, rho)

#plotData()
plotParam()
#plotEcmVar()
#plotModelo()
