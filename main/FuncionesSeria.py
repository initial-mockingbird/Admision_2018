import numpy as np
from PIL import *
from math import *
from random import *
import cv2
import imageio
from scipy.spatial import *
from scipy.misc import *
import os

class Individuo:

	def TriDrawColor(self):
		'''
		Procedimiento que dada una matriz de coordenadas nx2 y una matriz de vertices de triangulo px3
		dibuja en un surface una triangulacion de las coordenadas. 

		Nota: surface es una instancia de ImageDraw.draw
		'''
		for i in range(0,len(self.tri.simplices)):
			#Saca cada vertice del triangulo
			v0 = [self.coordenadas[self.tri.simplices[i,0]], self.coordenadas[self.tri.simplices[i,1]] ]
			v1 = [self.coordenadas[self.tri.simplices[i,1]], self.coordenadas[self.tri.simplices[i,2]] ]
			v2 = [self.coordenadas[self.tri.simplices[i,2]], self.coordenadas[self.tri.simplices[i,0]]]
			#dibuja el triangulo por vertice.
			self.Surface.line([tuple(v0[0]),tuple(v0[1])],fill=tuple(self.color[i]))
			self.Surface.line([tuple(v1[0]),tuple(v1[1])],fill=tuple(self.color[i]))
			self.Surface.line([tuple(v2[0]),tuple(v2[1])],fill=tuple(self.color[i]))

	#Version iterativa del floodfill con un super qeue
	def FloodFillBFS(self):
		#Llenamos cada triangulo con un color mediante floodfill
		for k in range(len(self.tri.simplices)):
			q = []
			q.append(self.centroide[k])

			while len(q)>=1:
				coord = q.pop()
				x = coord[0]
				y = coord[1]
				self.pixels[x,y] = tuple(self.color[k]) 
				if (x+1<self.width) and not(all(abs(self.pixels[x+1,y][i]-self.color[k][i])<=self.threshold for i in range(3))) and not(any(self.pixels[x+1,y][i] != 255 for i in range(3))): 
					q.append([x+1,y])
				if (x-1>=0) and not(all(abs(self.pixels[x-1,y][i]-self.color[k][i])<=self.threshold for i in range(3))) and not(any(self.pixels[x-1,y][i] != 255 for i in range(3))):
					q.append([x-1,y])
				if (y+1<self.height) and not(all(abs(self.pixels[x,y+1][i]-self.color[k][i])<=self.threshold for i in range(3))) and not(any(self.pixels[x,y+1][i] != 255 for i in range(3))):
					q.append([x,y+1])
				if (y-1>=0) and not(all(abs(self.pixels[x,y-1][i]-self.color[k][i])<=self.threshold for i in range(3))) and not(any(self.pixels[x,y-1][i] != 255 for i in range(3))):
					q.append([x,y-1])

	def Fitness(self,imgOriginal):
		#im1 = imgOriginal.resize((6, 6), Image.NEAREST)
		#im2 = imgFinal.resize((6, 6), Image.NEAREST)
		im1 = imgOriginal.resize((6,6), Image.BILINEAR)
		im2 = self.imgFinal.resize((6,6), Image.BILINEAR)
		
		pixels1 = im1.load()
		pixels2 = im2.load() 
		distance = 0
		for i in range(6):
			for j in range(6):
				distance += (pixels1[i,j][0]-pixels2[i,j][0])**2 + (pixels1[i,j][1]-pixels2[i,j][1])**2 + (pixels1[i,j][2]-pixels2[i,j][2])**2

		return distance-min(self.maxpoints,self.rate*self.width*self.height)
		#return distance+ min(int(self.maxpoints),int(self.rate*self.width*self.height))

	def displayImg(self,ruta):
		self.canvas.save(ruta+"/Result.jpg")

	def FillBlanks(self):
		for i in range(1,self.width-1):
			for j in range(1,self.height-1):
				if all(abs(255-self.pixels[i,j][k])<=5 for k in range(3)):
					if self.pixels[i-1,j]!= (255,255,255):
						self.pixels[i,j] =  self.pixels[i-1,j]
					elif self.pixels[i+1,j]!= (255,255,255):
						self.pixels[i,j] =  self.pixels[i+1,j]
					elif self.pixels[i,j-1]!= (255,255,255):
						self.pixels[i,j] =  self.pixels[i,j-1]
					else: 
						self.pixels[i,j] =  self.pixels[i,j+1]

	#aaaa, creo que tengo que crear un canvas en blanco each time pase una mutacion, para actualizar el coso
	def __init__(self,blur, rate, maxpoints,threshold,coordenadas,imgOriginal,canvas,width,height):

		#we always get non normalized values
		self.blur = blur
		self.rate = rate
		self.maxpoints = maxpoints
		self.threshold = int(threshold)
		self.coordenadas = coordenadas
		#self.imgOriginal = imgOriginal
		self.canvas = canvas
		self.color = []
		self.centroide = []
		self.width = width
		self.height = height
		#creamos un objeto que contenga los colores de la  imagen original de PIL
		self.rgb_im = imgOriginal.convert('RGB')
		#Creamos una superficie sobre la cual dinujar:
		self.Surface = ImageDraw.Draw(self.canvas)
		#Formamos la matriz de pixeles para manipular la imagen:
		self.pixels = self.canvas.load()

		#creamos un ibjeto de tipo Delaunay para la triangulacion
		self.tri = Delaunay(self.coordenadas)




		#creamos una superficie sobre el canvas blanco en la cual dibujaremos el resultado
		self.Surface = ImageDraw.Draw(self.canvas)

		#Este iterador  rellena los centroides y el color de cada triangulo
		#Nota: tri.simplices es un arreglo de p entradas (p triangulos), el cual en cada entrada
		#posee los indices de las coordenadas que conforma cada triangulo
		self.color = []
		for i in range(len(self.tri.simplices)):
			#primer vertice
			puntico1 = self.coordenadas[self.tri.simplices[i,0]]
			#segundo vertice
			puntico2 = self.coordenadas[self.tri.simplices[i,1]]
			#tercer vertice
			puntico3 = self.coordenadas[self.tri.simplices[i,2]]
			#centroide del triangulo, siempre cae adentro, super propiedad
			puntico = [int((puntico1[0]+puntico2[0]+puntico3[0])/3),int((puntico1[1]+puntico2[1]+puntico3[1])/3)]
			self.centroide.append(puntico)
			#obtengo el color del pixel
			red, green, blue = self.rgb_im.getpixel( ( puntico[0], puntico[1] ))
			#lo meto en color
			self.color.append( [int(red),int(green),int(blue)] )

		#dibujamos las aristas de los triangulos
		self.TriDrawColor()
		#Llenamos el canvas, yayyy
		self.FloodFillBFS()
		#self.FillBlanks()

		self.canvas.save("Result.jpg")
		self.imgFinal = Image.open("Result.jpg")
		self.fitness = self.Fitness(imgOriginal)

def orden(indexes,coso):
	aux = []
	for i in range(len(indexes)):
		aux.append(coso[indexes[i]])

	return aux

def altMutation(generacion,threshold):

	
	
	tau = 1.0/((float(generacion+1)**(1.0/2.0)))
	sigma = threshold*exp(tau*np.random.normal(0,1))
	if abs(sigma-0.1)<0.000001:
		sigma = 0.000001
	if sigma >0:
		return sigma
	else:
		return abs(threshold)

def AltRecombination(Parents):
	'''
	Whole arithmetic recombination
	Retorna 2 Hijos
	'''

	alpha = np.random.uniform(0,1)
	Children = [alpha*Parents[0]+(1-alpha)*Parents[1],alpha*Parents[1]+(1-alpha)*Parents[0]]

	return Children

#PEDRO IS THE NAMING GOD
#tengamos en cuenta que una biyeccion creciente del intervalo (0,1)-R es:
# y = tan ( (2x-1 )*pi/2)
#por lo tanto, su inversa sera:
# x = atan(y)/pi +1/2
#la cual tambien es creciente.
def GetReal(num,maxi,mini):
	return (maxi-mini)*num+mini
	#print("real: ",num)
	#return tan((2*num-1)*np.pi/2)

	'''
	if (num - num**2)>=0.001:
		return (2*num-1)/(num - num**2)
	else:
		return 0.001
	'''
def GetSmol(num,maxi,mini):
	return (num-mini)/(maxi-mini)
	#return atan(num)/np.pi +0.5
	#print("num: ",num)
	'''
	if num<=0.001:
		return 0.001
	re = (num-2)/(2*num)
	im = (num**2 +4)**(0.5)/(2*num)
	if re>im:
		return re-im
	else:
		return re+im
	'''
def ptInTriangle(p, p0, p1, p2):

	dX = p[0]-p2[0]
	dY = p[1]-p2[1]
	dX21 = p2[0]-p1[0]
	dY12 = p1[1]-p2[1]
	D = dY12*(p0[0]-p2[0]) + dX21*(p0[1]-p2[1])
	s = dY12*dX + dX21*dY
	t = (p2[1]-p0[1])*dX + (p0[0]-p2[0])*dY
	if (D<0):
		return ((s<=0) and (t<=0) and (s+t>=D) )
	else: 
		return ((s>=0) and (t>=0) and (s+t<=D))

def generate_config_file():
	path = os.getcwd()
	longitud_path = len(path)
	with open("configuration.txt","w+") as f:
		f.write("Ruta de la imagen="+path[:longitud_path-4]+"image_set/bestgirl.jpg\n")
		f.write("Ruta de la carpeta donde se guardaran las imagenes="+path[:longitud_path-4]+"Results/\n")
		f.write("Numero de generaciones=10\n")
		f.write("Porcentaje de que ocurra una mutacion=50\n")
		f.write("Cantidad de individuos=10\n")
		f.write("Porcentaje de reproduccion=100\n")
def check_load_config_file():
	path = os.getcwd()
	data = []
	if os.path.exists(path+"configuration.txt"):
		print("i exists")
		with open("configuration.txt","+r") as f:
			for lines in f:
				data.append(lines)
	else:
		generate_config_file()
		with open("configuration.txt","+r") as f:
			for lines in f:
				data.append(lines)
	data[0] = data[0][18:len(data[0])-1] #ruta de origen de la imagen
	data[1] = data[1][51:len(data[1])-1] #destino de guardado
	data[2] = int(data[2][23:len(data[2])-1]) #numero de generaciones
	data[3] = int(data[3][38:len(data[3])-1]) #porcentaje de mutacion
	data[4] = int(data[4][23:len(data[4])-1]) #cantidad de individuos
	data[5] = int(data[5][27:len(data[5])-1]) #porcentaje de reproduccion
	if not(os.path.exists(data[0])):
		raise Exception("Imagen no encontrada, verifique el campo de nuevo.")
	if not(os.path.exists(data[1])):
		raise Exception("Ruta de guardado no encontrada, verifique el campo de nuevo.")
	if data[2]<=0:
		raise Exception("Las generaciones no pueden ser negativas")
	if not (0<=data[3]<=100):
		raise Exception("El porcentaje de mutacion debe de estar entre 0 y 100")
	if data[4]<=0:
		raise Exception("La cantidad de individuos debe ser mayor que 0")
	if not (0<=data[5]<=100):
		raise Exception("El porcentaje de reproduccion debe de estar entre 0 y 100")
	return data