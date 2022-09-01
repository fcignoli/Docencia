#%% Importamos librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio #librería para hacer las animaciones (cómo instalar ---> https://pypi.org/project/imageio/)
plt.close('all')

#%% Definimos la carpeta donde guardaremos las animaciones
nombre_carpeta=os.getcwd() # Obtiene el nombre del directorio para guardar ahi los archivos
#nombre_carpeta='C:/Users/Usuario/Desktop/Practica Computacional/Kepler/'
if nombre_carpeta[-1]!='/':
    nombre_carpeta=nombre_carpeta+'/'

#%% Definimos las funciones que haran las animaciones
def hacer_gif(x_tierra,y_tierra,f_x_lista,f_y_lista,vel):
    plt.figure(figsize=(5, 5))
    images = [] #creo una lista para ir guardando los frames
    eje=1.1 
    lims=max(abs(min(np.min(x_tierra),np.min(y_tierra))),abs(max(np.max(x_tierra),np.max(y_tierra))))*eje
    plt.xlim(-lims,lims)
    plt.ylim(-lims,lims)    
    left, right = plt.xlim()  # return the current xlim
    down, up = plt.ylim()
    N = str(len(dias))
    for i in range(1,len(dias),vel):
      plt.xlabel('x')
      plt.ylabel('y')
      plt.title(nombre)
      plt.plot(x_tierra, y_tierra, color=[0.5,0.5,0.5])
      plt.plot(x_tierra[i], y_tierra[i],'o',color=[0.5,0.5,1], ms=10)
      plt.plot(0, 0, 'yo',ms=20)
      plt.xlim((left, right)) 
      plt.ylim((down, up)) 
      plt.arrow(x_tierra[i], y_tierra[i], f_x_lista[i]*5, f_y_lista[i]*5,length_includes_head=True,width=10**9.5,color=[0.75,0.75,0.5])
      plt.savefig('fig.png')
      plt.clf()
      images.append(imageio.imread('fig.png')) #función que agrega el png guardado a la lista "images" que contiene los frames

      # imprimo un contador en pantalla porque el video tarda
      if (i-1)%100==0:
          print('Frame '+str(i-1)+' de '+N)    
          
    imageio.mimsave(nombre_carpeta+'Kepler_'+nombre+'.gif', images) #Función que hace el gif
    plt.close() 
    print('animacion terminada')
#%% Definimos las funciones que haran los graficos  
def graficar(x_tierra,y_tierra,dias,delta_sol_tierra,v_tierra):
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(dias, delta_sol_tierra, color=[0.5,0.5,1],lw=3)
    #plt.plot(X,Y,'ko')
    plt.xlabel('tiempo (dias)')
    plt.ylabel('distancia (m)')
    
    plt.subplot(2,1,2)
    plt.plot(dias[1:], v_tierra, color=[0.5,1,0.5],lw=3)
    #plt.plot(X,Y,'ko')
    plt.xlabel('tiempo (dias)')
    plt.ylabel('Velocidad (m/s)')
    plt.show()

#%% Definimos la funcion que grafica una foto
def hacer_foto(i,x_tierra,y_tierra,f_x_lista,f_y_lista):
    plt.figure(figsize=(5, 5))
    eje=1.1 
    lims=max(abs(min(np.min(x_tierra),np.min(y_tierra))),abs(max(np.max(x_tierra),np.max(y_tierra))))*eje
    plt.xlim(-lims,lims)
    plt.ylim(-lims,lims)    
    left, right = plt.xlim()  
    down, up = plt.ylim()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(nombre)
    #grafico trayectoria y posiciones de la Tierra y el Sol
    plt.plot(x_tierra, y_tierra, color=[0.5,0.5,0.5], label = 'trayectoria')
    plt.plot(x_tierra[i], y_tierra[i],'o',color=[0.5,0.5,1], ms=10, label = 'Tierra')
    plt.plot(0, 0, 'yo', ms=20, label = 'Sol')
    plt.xlim((left, right)) 
    plt.ylim((down, up)) 
    #grafico la fuerza gravitatoria
    plt.arrow(x_tierra[i], y_tierra[i], f_x_lista[i]*5, f_y_lista[i]*5,length_includes_head=True,width=10**9.5,color=[0.75,0.75,0.5], label = 'Fuerza Gravedad')
    plt.legend()
#%% Definimos los parametros del problema
# masas
m_sol = 1.989e30
m_tierra = 5.972e24

# radios
radio_sol =695508000
radio_tierra = 6371000

G = 6.693E-11 # constante universal de la fuerza gravitatoria en m^3/s^/kg
dt = 60 * 60 * 24      #paso temporal en segundos (cada dt es un dia)

#%% Definimos las condiciones iniciales

# Elipse
nombre='Elipse'
posicion_tierra = [-300000000000.0,500000000.0]
v_tierra = [0.0, -11574.074074074075]
n_vueltas=4


#%% Algoritmo de Verlet
dias = [-1, 0]

x_tierra = [posicion_tierra[0], posicion_tierra[0]+v_tierra[0]*dt]
y_tierra = [posicion_tierra[1], posicion_tierra[1]+v_tierra[1]*dt]

v_tierra = [(v_tierra[0]**2+v_tierra[1]**2)**0.5]

tiempo_total = 365*n_vueltas #número de pasos de la simulación

f_x_lista=[]
f_y_lista=[]
delta_sol_tierra = [np.sqrt(x_tierra[0]**2+y_tierra[0]**2),np.sqrt(x_tierra[1]**2+y_tierra[1]**2)]
x_sol = 0
y_sol = 0
for i in range(1, tiempo_total - 1):
    #calculo los deltas
    delta_x = x_sol - x_tierra[i]
    delta_y = y_sol - y_tierra[i]

    # calculo las fuerzas
    suma_deltas_cuadrado = (delta_x**2 + delta_y**2)

    f_x = (G * m_sol * delta_x) / (suma_deltas_cuadrado * np.sqrt(suma_deltas_cuadrado))
    f_y = (G * m_sol * delta_y) / (suma_deltas_cuadrado * np.sqrt(suma_deltas_cuadrado))

    f_x_lista.append(f_x*10**13)
    f_y_lista.append(f_y*10**13)
    
    #actualizo la posicion x
    x_actual = x_tierra[i]
    x_prev = x_tierra[i-1]

    x_nueva = 2 * x_actual - x_prev + f_x * dt**2
    x_tierra.append(x_nueva)

    #actualizo la posicion y
    y_actual = y_tierra[i]
    y_prev = y_tierra[i-1]

    y_nueva = 2 * y_actual - y_prev + f_y * dt**2
    y_tierra.append(y_nueva)
    
    v_tierra.append(np.sqrt((x_actual-x_prev)**2+(y_actual-y_prev)**2)/dt)
    delta_sol_tierra.append(np.sqrt(suma_deltas_cuadrado))
    
    #actualizo el tiempo
    dias.append(i)

#%% Graficamos y hacemos la animacion
graficar(x_tierra,y_tierra,dias,delta_sol_tierra,v_tierra)
i=0
hacer_foto(i,x_tierra,y_tierra,f_x_lista,f_y_lista)
vel=5
hacer_gif(x_tierra,y_tierra,f_x_lista,f_y_lista,vel)

