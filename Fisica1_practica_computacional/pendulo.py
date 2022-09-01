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

#%% Definimos la funcion que hara las animaciones

def hacer_gif(tiempo, tita, tension, vel=10):
    N = str(len(tiempo))
    images = [] #creo una lista para ir guardando cada foto
    plt.figure(figsize=(5, 5)) # abro una figura para dibujar las fotos
    for i in range(1,len(tiempo),vel): #hago un loop para iterar en el tiempo
      plt.xlabel('x (m)') #etiqueto el eje x
      plt.ylabel('y (m)') #etiqueto el eje y
      plt.title(nombre) #agrego un título
      plt.plot(-np.sin(tita), -np.cos(tita), color=[0.75,0.75,0.75]) #dibujo la trayectoria en gris claro
      plt.plot(-np.sin(tita[i]), -np.cos(tita[i]),'o', color=[0.5,0.5,1], ms=10) #dibujo la posición actual en azul
      plt.plot([0, -np.sin(tita[i])], [0, -np.cos(tita[i])], 'y', lw=1) #dibujo el centro de la barra
      plt.xlim(-1.5,1.5) #seteo los límites en el eje x
      plt.ylim(-1.5,1.5) #seteo los límites en el eje y
      #Dibujo la tension
      plt.arrow(-np.sin(tita[i]), -np.cos(tita[i]), np.sin(tita[i])*tension[i]*10**-1.5, np.cos(tita[i])*tension[i]*10**-1.5,color=[1,0.5,0.5],length_includes_head=True,width=10**-2)
      #Dibujo el peso
      plt.arrow(-np.sin(tita[i]), -np.cos(tita[i]), 0, -g*10**-1.5, color=[0.5,1,0.5],length_includes_head=True,width=10**-2)
      #Dibujo la fuerza neta
      plt.arrow(-np.sin(tita[i]), -np.cos(tita[i]), np.sin(tita[i])*tension[i]*10**-1.5, np.cos(tita[i])*tension[i]*10**-1.5-g*10**-1.5,color=[0.5,0.5,0.25],length_includes_head=True,width=10**-2)
      plt.savefig('fig.png')
      plt.clf()
      images.append(imageio.imread('fig.png')) #función que agrega el png guardado a la lista "images" que contiene los frames

      # imprimo un contador en pantalla porque el video tarda
      if (i-1)%100==0:
          print('Frame '+str(i-1)+' de '+N)     
          
    imageio.mimsave(nombre_carpeta+'Pendulo_'+nombre+'.gif', images) # Creo el Gif y lo guardo con ese nombre
    plt.close()

#%% Definimos la funcion que hara los graficos
def graficar(tiempo, tita, v_tita, a_lista, Tension):
    plt.figure()
    plt.subplot(411)
    plt.plot(tiempo, (np.asarray(tita))/np.pi, color=[0.5,0.5,1],lw=3)
    plt.plot(tiempo, np.ones(len(tiempo))*0,'k--',lw=1)
    #plt.plot(tiempo, np.ones(len(tiempo))*2*np.pi,'k--')
    #plt.plot(tiempo, -np.ones(len(tiempo))*2*np.pi,'k--')
    plt.xlabel('tiempo (s)')
    plt.ylabel(r'$\theta$ '+r'($\pi$)')
    plt.legend()
    
    plt.subplot(412)
    plt.plot(tiempo[1:], v_tita, color=[0.5,0.75,0.75] ,lw=3)
    plt.plot(tiempo, np.zeros(len(tiempo)),'k--',lw=1)

    plt.xlabel('tiempo (s)')
    plt.ylabel('v (1/s)')
    plt.legend()
    
    plt.subplot(413)
    plt.plot(tiempo, a_lista, color=[0.75,0.75,0.5] ,lw=3)
    plt.plot(tiempo, np.zeros(len(tiempo)),'k--',lw=1)
    plt.xlabel('tiempo (s)')
    plt.ylabel(r'$a (1/s^2)$')
    plt.legend()
    plt.show()
    
    plt.subplot(414)
    plt.plot(tiempo[1:], Tension, color=[1,0.5,0.5],lw=3, label='tension')
    plt.plot(tiempo[1:], -np.ones(len(tiempo)-1)*g, color=[0.5,1,0.5],lw=3,label='peso')
    plt.plot(tiempo, np.ones(len(tiempo))*0,'k--',lw=1)
    plt.xlabel('tiempo (s)')
    plt.ylabel('F (N)')
    plt.legend()
    plt.show()
    
#%% Definimos la funcion que grafica una foto
def hacer_foto(i, tita, tension, vel=10):
    plt.figure(figsize=(5, 5)) # abro una figura para dibujar las fotos
    plt.xlabel('x (m)') #etiqueto el eje x
    plt.ylabel('y (m)') #etiqueto el eje y
    plt.title(nombre)   #agrego un título a la figura
    plt.plot(-np.sin(tita), -np.cos(tita), color=[0.75,0.75,0.75], label = 'trayectoria') #dibujo la trayectoria en gris claro
    plt.plot(-np.sin(tita[i]), -np.cos(tita[i]),'o', color=[0.5,0.5,1], ms=10, label = 'posicion') #dibujo la posición actual en azul
    plt.plot([0, -np.sin(tita[i])], [0, -np.cos(tita[i])], 'y', lw=1, label = 'barrita') #dibujo la barra
    plt.xlim(-1.5,1.5) #seteo los límites en el eje x
    plt.ylim(-1.5,1.5) #seteo los límites en el eje y
    #Dibujo la tension
    plt.arrow(-np.sin(tita[i]), -np.cos(tita[i]), np.sin(tita[i])*tension[i]*10**-1.5, np.cos(tita[i])*tension[i]*10**-1.5,color=[1,0.5,0.5],length_includes_head=True,width=10**-2, label = 'tension')
    #Dibujo el peso
    plt.arrow(-np.sin(tita[i]), -np.cos(tita[i]), 0, -g*10**-1.5, color=[0.5,1,0.5],length_includes_head=True,width=10**-2, label = 'peso')
    #Dibujo la fuerza neta
    plt.arrow(-np.sin(tita[i]), -np.cos(tita[i]), np.sin(tita[i])*tension[i]*10**-1.5, np.cos(tita[i])*tension[i]*10**-1.5-g*10**-1.5,color=[0.5,0.5,0.25],length_includes_head=True,width=10**-2, label = 'aceleracion')
    plt.legend()
#%% Definimos parametros del problema
L=1 # largo del pendulo (m)
g=10 # gravedad
viscocidad=0.0
#%% Definimos condiciones iniciales
# Sin viscosidad

nombre='Pequenas Oscilaciones Seno'
posicion_inicial=0
velocidad_inicial=1

# nombre='Pequenas Oscilaciones Coseno'
# posicion_inicial=0.5
# velocidad_inicial=0

# nombre='Acotado angosto'
# posicion_inicial=0
# velocidad_inicial=4

# nombre='Acotado amplio'
# posicion_inicial=0
# velocidad_inicial=6

# nombre='Critico Acotado Desde Abajo'
# posicion_inicial=0
# velocidad_inicial=6.324

# nombre='Critico Acotado Desde Arriba'
# posicion_inicial=np.pi-0.01
# velocidad_inicial=0.0

# nombre='Critico Libre Desde Arriba'
# posicion_inicial=np.pi
# velocidad_inicial=0.1

# nombre='Crítico Libre Desde Abajo'
# posicion_inicial=0
# velocidad_inicial=6.325

# nombre='Libre'
# posicion_inicial=0
# velocidad_inicial=7

#%% Con viscosidad

# viscocidad=0.5

# nombre='Pequenas Oscilaciones Viscoso'
# posicion_inicial=0
# velocidad_inicial=1

# nombre='Acotado angosto Viscoso'
# posicion_inicial=0
# velocidad_inicial=4

# nombre='Acotado amplio Viscoso'
# posicion_inicial=0
# velocidad_inicial=6

# nombre='Critico Viscoso'
# posicion_inicial=0
# velocidad_inicial=10

# nombre='Libre Viscoso'
# posicion_inicial=0
# velocidad_inicial=20

#%% Algoritmo de Verlet
pasos = 1000 #número de pasos de la simulación
dt=0.01
t0=0

tita=[]
tita.append(posicion_inicial)
tita.append(posicion_inicial+velocidad_inicial*dt)

tiempo = [t0-dt, t0]
velocidad_tangencial = [(tita[1]-tita[0])/dt]
aceleracion=[-g/L*np.sin(tita[0]),-g/L*np.sin(tita[1])-viscocidad*velocidad_tangencial[-1]]
tension=[g*np.cos(tita[0])+L*velocidad_tangencial[0]**2]

tiempo_actual=tiempo[1]
for i in range(1, pasos - 1):
    #actualizo las posiciones actual y previa
    tita_actual = tita[i]
    tita_prev = tita[i-1]
    #actualizo la velocidad actual
    velocidad_tangencial.append((tita_actual-tita_prev)/dt)
    #calculo la aceleración actual y la guardo
    a_tita = -g/L*np.sin(tita[i])-viscocidad*velocidad_tangencial[-1]
    aceleracion.append(a_tita)
    #calculo la posición nueva y la guardo
    tita_nueva = 2 * tita_actual - tita_prev + a_tita * dt**2
    tita.append(tita_nueva)
    #calculo la tension y la guardo
    tension.append(g*np.cos(tita_actual)+L*velocidad_tangencial[-1]**2)
    #actualizo el tiempo y lo guardo
    tiempo_actual=tiempo_actual+dt
    tiempo.append(tiempo_actual)

# calculo el periodo
T=2*np.pi*np.sqrt(L/g)

#%% Grafico y hago la animacion
i = 0 #indico a que tiempo hacer la foto
hacer_foto(i, tita, tension)
graficar(tiempo, tita, velocidad_tangencial, aceleracion, tension)
hacer_gif(tiempo, tita, tension)