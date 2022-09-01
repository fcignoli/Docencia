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

# calculo los (x,y) min y max permitidos por r_min y r_max para las dos masas
def calcular_xy_max_min(r_lista):
    xx_1=np.linspace(-np.max(r_lista)*m1/(m1+m2),np.max(r_lista)*m1/(m1+m2),1000)
    xx_2=np.linspace(-np.max(r_lista)*m2/(m1+m2),np.max(r_lista)*m2/(m1+m2),1000)
    yy_11=[]
    yy_12=[]
    yy_21=[]
    yy_22=[]
    for j in range(len(xx_1)):
        xx_1_resta=(np.min(r_lista)*m1/(m1+m2))**2-xx_1[j]**2
        xx_2_resta=(np.min(r_lista)*m2/(m1+m2))**2-xx_2[j]**2
        if xx_1_resta>0:
            yy_11.append(np.sqrt(xx_1_resta))
        else:
            yy_11.append(0)
        if xx_2_resta>0:
            yy_21.append(np.sqrt(xx_2_resta))
        else:
            yy_21.append(0)
        yy_12.append(np.sqrt((np.max(r_lista)*m1/(m1+m2))**2-xx_1[j]**2))
        yy_22.append(np.sqrt((np.max(r_lista)*m2/(m1+m2))**2-xx_2[j]**2))
    yy_11=np.asarray(yy_11)
    yy_12=np.asarray(yy_12)
    yy_21=np.asarray(yy_21)
    yy_22=np.asarray(yy_22)
    return xx_1,xx_2,yy_11,yy_12,yy_21,yy_22

# armo el video
def hacer_gif(x1,y1,x2,y2,f_x_lista,f_y_lista,tiempo,frames,m1,m2,r_lista):

    plt.figure(figsize=(5, 5))
    images = [] 
    eje=1.1 
    lims=max(abs(min(np.min(x1),np.min(y1))),abs(max(np.max(x1),np.max(y1))))*eje
    plt.xlim(-lims,lims)
    plt.ylim(-lims,lims)    
    left, right = plt.xlim()  
    down, up = plt.ylim()
    
    #calculo los (x,y) min y max permitidos por r_min y r_max para las dos masas
    x1_rango,x2_rango,y1_min,y1_max,y2_min,y2_max = calcular_xy_max_min(np.asarray(r_lista))
    
    N = str(len(tiempo))
    for i in range(1,len(tiempo),frames):
      # escribo nombre y labels
      plt.xlabel('x (m)')
      plt.ylabel('y (m)')
      plt.title(nombre)
      plt.plot([x1[i], x2[i]],[y1[i], y2[i]], 'k', Linewidth=1)
      
      # grafico trayectorias
      plt.plot(x1, y1, Color=[0.5,0.5,0.0])
      plt.plot(x2, y2, Color=[0.5,0.0,0.5])
      
      # grafico masas en su posicion
      plt.plot(x1[i], y1[i],'o',Color=[0.5,0.5,0.0], ms=10*m1)
      plt.plot(x2[i], y2[i],'o',Color=[0.5,0.0,0.5], ms=10*m2)

      # grafico CM
      plt.plot(0, 0, 'o', Color=[0.0,0.5,0.5], ms=5)
      plt.xlim((left, right)) 
      plt.ylim((down, up)) 
      
      # grafico flechas de fuerza elastica
      plt.arrow(x1[i], y1[i], f_x_lista[i]/max(f_x_lista)*lims/2, f_y_lista[i]/max(f_x_lista)*lims/2,length_includes_head=True,width=0.02,Color=[0.5,0.5,0.0])
      plt.arrow(x2[i], y2[i], -f_x_lista[i]/max(f_x_lista)*lims/2, -f_y_lista[i]/max(f_x_lista)*lims/2,length_includes_head=True,width=0.02,Color=[0.5,0.0,0.5])

      # grafico regiones entre r_min y r_max
      plt.fill_between(x2_rango,y2_min ,y2_max ,facecolor=[0.5,0.5,0.0], alpha=0.25)
      plt.fill_between(x2_rango,-y2_min,-y2_max,facecolor=[0.5,0.5,0.0], alpha=0.25)   
      plt.fill_between(x1_rango,y1_min ,y1_max ,facecolor=[0.5,0.0,0.5], alpha=0.25)
      plt.fill_between(x1_rango,-y1_min,-y1_max,facecolor=[0.5,0.0,0.5], alpha=0.25)
      
      # guardo el frame y lo agrego a la lista de la animacion
      plt.savefig('fig.png')
      plt.clf()
      images.append(imageio.imread('fig.png')) #función que agrega el png guardado a la lista "images" que contiene los frames
      
      # imprimo un contador en pantalla porque el video tarda
      if (i-1)%100==0:
          print('Frame '+str(i-1)+' de '+N)          

    imageio.mimsave(nombre_carpeta+'Resorte_'+nombre+'.gif', images) #Función que hace el gif
    plt.close()
    print('animacion terminada')
    
#%% Definimos la funcion que grafica una foto
def hacer_foto(x1,y1,x2,y2,f_x_lista,f_y_lista,tiempo,frame,m1,m2,r_lista):
    plt.figure(figsize=(5, 5))
    eje=1.1 
    lims=max(abs(min(np.min(x1),np.min(y1))),abs(max(np.max(x1),np.max(y1))))*eje
    plt.xlim(-lims,lims)
    plt.ylim(-lims,lims)    
    left, right = plt.xlim()  # return the current xlim
    down, up = plt.ylim()

    #calculo los (x,y) min y max permitidos por r_min y r_max para las dos masas
    x1_rango,x2_rango,y1_min,y1_max,y2_min,y2_max = calcular_xy_max_min(np.asarray(r_lista))
    
    i=frame
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(nombre)
    
    # grafico trayectorias
    plt.plot(x1, y1, Color=[0.5,0.5,0.0], label='trayectoria $m_1$')
    plt.plot(x2, y2, Color=[0.5,0.0,0.5], label='trayectoria $m_2$')
    
    # grafico masas en su posicion
    plt.plot(x1[i], y1[i],'o',Color=[0.5,0.5,0.0], ms=10*m1, label='$m_1$')
    plt.plot(x2[i], y2[i],'o',Color=[0.5,0.0,0.5], ms=10*m2, label='$m_2$')
    
    # grafico el vector r
    plt.plot([x1[i], x2[i]],[y1[i], y2[i]], Color=[0.0,0.5,0.5], Linewidth=1, label=r'$\vec r$')
    
    # grafico el CM
    plt.plot(0, 0, 'o', Color=[0.0,0.5,0.5], ms=5, label='CM')
    
    plt.xlim((left, right)) 
    plt.ylim((down, up)) 
    
    # grafico las fuerzas elasticas
    plt.arrow(x1[i], y1[i], f_x_lista[i]/max(f_x_lista)*lims/4, f_y_lista[i]/max(f_x_lista)*lims/4,length_includes_head=True,width=0.02,Color=[0.5,0.5,0.0], label='$F_1$')
    plt.arrow(x2[i], y2[i], -f_x_lista[i]/max(f_x_lista)*lims/4, -f_y_lista[i]/max(f_x_lista)*lims/4,length_includes_head=True,width=0.02,Color=[0.5,0.0,0.5], label='$F_2$')
    
    # grafico regiones entre r_min y r_max
    plt.fill_between(x1_rango,y1_min ,y1_max ,facecolor=[0.5,0.0,0.5], alpha=0.25, label = r'$[r_{2_{min}};r_{2_{max}}]$')
    plt.fill_between(x1_rango,-y1_min,-y1_max,facecolor=[0.5,0.0,0.5], alpha=0.25)
    plt.fill_between(x2_rango,y2_min ,y2_max ,facecolor=[0.5,0.5,0.0], alpha=0.25, label = r'$[r_{1_{min}};r_{1_{max}}]$')
    plt.fill_between(x2_rango,-y2_min,-y2_max,facecolor=[0.5,0.5,0.0], alpha=0.25)   

    
    plt.title('trayectorias')
    plt.legend()
    plt.savefig(nombre+'.eps', format='eps')

#%% Definimos las funciones que haran los graficos  
def graficar(x1,y1,x2,y2,tiempo,fuerza,r_lista,E_lista,L0_lista,v1_lista,v2_lista,v1_tita,v2_tita):
    r_lista=np.asarray(r_lista)

    plt.figure()
    
    plt.subplot(5,1,1)
    plt.plot(tiempo, r_lista, Color=[0.0,0.5,0.5],Linewidth=3)
    plt.plot(tiempo, m2/m1*r_lista, Color=[0.5,0.5,0.0],Linewidth=3)
    plt.plot(tiempo, m1/m2*r_lista, Color=[0.5,0.0,0.5],Linewidth=3)
    plt.plot([tiempo[0],tiempo[-1]], [l0,l0],'k',Linewidth=3)
    plt.xlabel('t (s)')
    plt.ylabel('r (m)')

    plt.subplot(5,1,2)
    plt.plot(tiempo, fuerza, Color=[0.0,0.5,0.5],Linewidth=3)
    plt.xlabel('t (s)')
    plt.ylabel('F (N)')
    
    plt.subplot(5,1,3)
    plt.plot(tiempo[1:], v1_tita[1:], Color=[0.5,0.5,0.0],Linewidth=3)
    plt.plot(tiempo[1:], v2_tita[1:], Color=[0.5,0.0,0.5],Linewidth=3)
    plt.xlabel('t (s)')
    plt.ylabel('$\dot \Theta$ (1/s)')
    
    plt.subplot(5,1,4)
    plt.plot(tiempo[1:], E_lista[1:], Color=[0.5,0.5,0.0],Linewidth=3)
    plt.ylim([0,1.1*np.max(E_lista[1:])])
    plt.xlabel('t (s)')
    plt.ylabel('E (J)')
    
    plt.subplot(5,1,5)
    plt.plot(tiempo[1:], L0_lista[1:], Color=[0.5,0.5,0.0],Linewidth=3)
    plt.ylim([0,1.1*np.max(L0_lista[1:])])
    plt.xlabel('t (s)')
    plt.ylabel('$L_0 (kgm^2/s)$')
    
    plt.show()
    
#%% Seteamos los parametros del problema

# Flor de 5 petalos
nombre='Flor_5'
m1 = 1      # masa1 en kg
m2 = 2      # masa2 en kg
k  = 1      # k del resorte
r0 = 1.0    # distancia inicial
v0 = 0.5    # velocidad inicial
l0 = 2      # long natural resorte
tiempo_total = 2200    # tiempo de integracion

# # Elipse
# nombre='Elipse'
# m1 = 1 
# m2 = 2 
# k  = 1
# r0 = 1.0
# v0 = 0.5
# l0 = 0
# tiempo_total = 2200

# # Margarita
# nombre='Margarita'
# m1 = 1 
# m2 = 2 
# k  = 1
# r0 = 1.0
# v0 = 0.5
# l0 = 5
# tiempo_total = 4500

# # Triangulo
# nombre='Triangulo'
# m1 = 1
# m2 = 2
# k  = 1
# r0 = 1.0
# v0 = 0.5
# l0 = 1
# tiempo_total = 2200

# # Circulo
# nombre='Circulo'
# m1 = 1 # en kg
# m2 = 1 # en kg
# k  = 1
# r0 = 1.0
# v0 = np.sqrt(k/(m1/2))*r0/2
# l0 = 0
# tiempo_total = 2200

#%% Algoritmo de Verlet para integrar
mt = m1+m2
x1 = [r0*m2/mt]
x2 = [-r0*m1/mt]
y1 = [0.0]
y2 = [0.0]
vx1 = [0.0]
vx2 = [0.0]
vy1 = [v0]
vy2 = [-m1/m2*v0]

dt = 0.01 #paso temporal en segundos
tiempo = [-dt, 0]

x1.append(x1[0]+vx1[0]*dt)
x2.append(x2[0]+vx2[0]*dt)
y1.append(y1[0]+vy1[0]*dt)
y2.append(y2[0]+vy2[0]*dt)

r_lista=[np.sqrt((x1[0]-x2[0])**2+(y1[0]-y2[0])**2)]

f_x_lista=[x1[0]-x2[0]]
f_y_lista=[y1[0]-y2[0]]
fuerza=[-k*(r_lista[0]-l0)]
v1_lista=[np.sqrt(vx1[0]**2+vy1[0]**2)/2]
v2_lista=[np.sqrt(vx2[0]**2+vy2[0]**2)/2]

v1_tita=[-vx1[0]*np.sin(x1[0]*m1/r_lista[0]/mt)+vy1[0]*np.cos(x1[0]*m1/r_lista[0]/mt)]
v2_tita=[-vx2[0]*np.sin(x2[0]*m2/r_lista[0]/mt)+vy2[0]*np.cos(x2[0]*m2/r_lista[0]/mt)]
E_lista=[0.5*m1*v1_lista[0]**2+0.5*m2*v2_lista[0]**2+0.5*k*(r_lista[0]-l0)**2]
L0_lista=[m2*r_lista[0]*v1_tita[0]+m1*r_lista[0]*v2_tita[0]]

for i in range(1, tiempo_total - 1):
    #calculo los deltas
    rx = x1[i]-x2[i]
    ry = y1[i]-y2[i]
    r_lista.append(np.sqrt(rx**2+ry**2))
    fuerza.append(-k*(r_lista[i]-l0))
    
    f_x = fuerza[i]*rx/r_lista[i]
    f_y = fuerza[i]*ry/r_lista[i]

    f_x_lista.append(f_x)
    f_y_lista.append(f_y)

    x1.append(2*x1[i]-x1[i-1]+f_x/m1*dt**2)
    x2.append(2*x2[i]-x2[i-1]-f_x/m2*dt**2)
    y1.append(2*y1[i]-y1[i-1]+f_y/m1*dt**2)
    y2.append(2*y2[i]-y2[i-1]-f_y/m2*dt**2)

    vx1 = (x1[i]-x1[i-1])/dt
    vy1 = (y1[i]-y1[i-1])/dt
    vx2 = (x2[i]-x2[i-1])/dt
    vy2 = (y2[i]-y2[i-1])/dt
    
    v1_tita.append(-vx1*y1[i]/r_lista[i]*m1/m2+vy1*x1[i]/r_lista[i]*m1/m2)
    v2_tita.append(-vx2*y2[i]/r_lista[i]*m2/m1+vy2*x2[i]/r_lista[i]*m2/m1)
          
    v1_lista.append(np.sqrt(vx1**2+vy1**2))
    v2_lista.append(np.sqrt(vx2**2+vy2**2))
    
    E_lista.append(0.5*m1*v1_lista[i]**2+0.5*m2*v2_lista[i]**2+0.5*k*(r_lista[i]-l0)**2)
    L0_lista.append(m2*r_lista[i]*v1_tita[i]+m1*r_lista[i]*v2_tita[i])
    
    tiempo.append(tiempo[i]+dt)
    
fuerza.append(fuerza[-1])
r_lista.append(r_lista[-1])   
v1_lista.append(v1_lista[-1])  
v2_lista.append(v2_lista[-1])  
E_lista.append(E_lista[-1])   
L0_lista.append(L0_lista[-1])   
v1_tita.append(v1_tita[-1]) 
v2_tita.append(v2_tita[-1]) 

#%% Graficamos y hacemos la animacion
graficar(x1,y1,x2,y2,tiempo,fuerza,r_lista,E_lista,L0_lista,v1_lista,v2_lista,v1_tita,v2_tita)
frames=10
hacer_gif(x1,y1,x2,y2,f_x_lista,f_y_lista,tiempo,frames,m1,m2,r_lista)
hacer_foto(x1,y1,x2,y2,f_x_lista,f_y_lista,tiempo,frames,m1,m2,r_lista)

