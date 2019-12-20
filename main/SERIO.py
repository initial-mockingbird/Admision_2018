import numpy as np
from scipy.spatial import *
from scipy.misc import *
import imageio
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from PIL import *
from math import *
from random import *
import FuncionesSeria
import cv2
import copy
import os


#nombre del archivo a manipular
nombre = input("Ingrese la ruta de la imagen: ")
#abrimos una imagen
im = Image.open(nombre)
#ruta destino:
ruta_destino = input("Ingrese la ruta de la carpeta donde se guardaran los resultados: ")
#creamos un canvas totalmente blanco del mismo tamano de la imagen
canvas = Image.new('RGB', im.size, (255,255,255,0))
#creamos una superficie sobre el canvas blanco en la cual dibujaremos el resultado
Surface = ImageDraw.Draw(canvas)

#tamano de la imagen
width, height = im.size

#Datos que deberia pedir la interfaz:
#suggested: 10
generacion=int(input("Ingrese generacion: "))
porcentaje_mut=int(input("Ingrese porcentaje de mutacion: "))
#suggested:10
poblacion = int(input("Ingrese la poblacion: "))
porcentaje_re=int(input("Ingrese porcentaje de reproduccion: "))

#Aqui se guardan los individuos los cuales van a ser una super clase
individuos = []

#Aqui se guardaran los fitness para hacerles sort:
auxFitness = []
#Aqui se van a guardar las coordenadas a triangular
#n = min(width,height)

for i in range (poblacion):
	#blur 
	b = np.random.uniform(0.1,1.5)
	#rate, Percentage of vertices/total pixel count we want in the output.
	r = np.random.uniform(0.1,0.5)
	#points, A cap of maximun number of vertices
	#n = np.random.uniform(2500)
	n = np.random.uniform(100,width+height)

	#Nueva imagen para pintar
	canvas = Image.new('RGB', im.size, (255,255,255,0))
	while True:
		#Threshold: minimal strength of potential vertices.
		E = np.random.uniform(1,100)
		#coordenadas del objeto:
		coordenadas = []


		#Grayscale image for preprocessing:
		gray = im.convert('L')
		#gray.show()

		#blurred image:
		blurr = gray.filter(ImageFilter.BoxBlur(b))
		#blurr.show()

		#preprocesamos aun mas la imagen para encontrar los puntos limite
		edges = blurr.filter(ImageFilter.FIND_EDGES)
		#edges = edges.filter(ImageFilter.EDGE_ENHANCE)

		#como PIL no da las coordenadas, vamos a guardar los resultados en una imagen
		edges.save("Result.jpeg")
		#edges.show()

		#cv2 si nos ofrece una herramienta que suelta las coordenadas de los puntos, por lo tanto
		#abrimos la imagen con cv2
		img = cv2.imread("Result.jpeg")
		#creamos un objeto de tipo fast con threshold E, para detectar los puntos limite
		fast = cv2.FastFeatureDetector_create(int(E))
		#objeto que guarda las coordenadas de los puntos limite
		kp = fast.detect(img,None)

		if len(kp)>0:
			break


	#guardamos todos los puntos limite en un arreglo de candidatos
	Candidates = np.array(  [ [kp[k].pt[0],kp[k].pt[1]]  for k  in range(len(kp)) ]   ) 

	#sacamos una cantidad razonable de puntos limite, y los metemos en coordenadas
	for i in range(min(int(n),int(r*width*height))):
	#for i in range(int(n)):
		coordenadas.append(list(map(int,Candidates[np.random.randint(len(Candidates))])))

	#tenemos que colocar las esquinas del canvas
	coordenadas.append([0,0])
	coordenadas.append([0,height-1])
	coordenadas.append([width-1,0])
	coordenadas.append([width-1,height-1])
	individuos.append(FuncionesSeria.Individuo(b,r,n,E,coordenadas,im,canvas,width,height) )




for i in range(len(individuos)):
	auxFitness.append(individuos[i].fitness)
indexes = sorted(range(len(individuos)), key=lambda k: auxFitness[k])
#indexes.reverse()
individuos = FuncionesSeria.orden(indexes,individuos)




print("---------------------------------------------------------")
for i in range(len(individuos)):
	print("individuo "+str(i)+": by fit ",individuos[i].fitness)
	print("--------------------------------------")
	print("individuo blur: ",individuos[i].blur)
	print("Individuo rate: ",individuos[i].rate)
	print("individuo max point: ",individuos[i].maxpoints)
	print("individuo threshold: ",individuos[i].threshold)
	print("individuo fitness: ",individuos[i].fitness)

	individuos[i].TriDrawColor()
	individuos[i].FloodFillBFS()
	individuos[i].displayImg(ruta_destino)
	os.rename(ruta_destino+"/Result.jpg", ruta_destino+"/ResultInd"+str(i)+".jpg")
	#input()
print("---------------------------------------------------------")


for genact in range(generacion):

		#Intentamos recombinarlos:
		#Los individuos se recombinan si pasan un check de probabilidad porcentaje_re
		if(np.random.randint(1,porcentaje_re)>=np.random.randint(1,102-porcentaje_re)):
			while(True):
				#Creo un arreglo de hijos: 
				Children = []
				#Creo un arreglo que va a contener los fitness de los hijos:
				auxFitness = []
				#Agarro un individuo top5
				indexP1 = np.random.randint(5)
				#Agarro un individuo despues del 1er cuarto.
				indexP2 = np.random.randint(len(individuos))


				#pendiente, altrecombination devuelve una lista, tengo que arreglar eso.
				Nblur1 = FuncionesSeria.GetSmol(individuos[indexP1].blur,5,0)
				Nrate1 = FuncionesSeria.GetSmol(individuos[indexP1].rate,5,0)
				Nmaxpoints1 = FuncionesSeria.GetSmol(individuos[indexP1].maxpoints,3000,1)
				Nthreshold1= FuncionesSeria.GetSmol(individuos[indexP1].threshold,250,0)

				Nblur2 = FuncionesSeria.GetSmol(individuos[indexP2].blur,5,0)
				Nrate2 = FuncionesSeria.GetSmol(individuos[indexP2].rate,5,0)
				Nmaxpoints2 = FuncionesSeria.GetSmol(individuos[indexP2].maxpoints,3000,1)
				Nthreshold2 = FuncionesSeria.GetSmol(individuos[indexP2].threshold,250,0)
				#Saco 2 hijos para cada atributo
				Cblur = FuncionesSeria.AltRecombination([Nblur1,Nblur2])
				Crate= FuncionesSeria.AltRecombination([Nrate1,Nrate2])
				Cmaxpoints = FuncionesSeria.AltRecombination([Nmaxpoints1,Nmaxpoints2])
				Cthreshold = FuncionesSeria.AltRecombination([Nthreshold1,Nthreshold2])

				kp = []
				#verifico que los chamines no me generen una imagen no valida
				for i in range(2):

					#coordenadas del objeto:
					coordenadas = []
				
					
					#Grayscale image for preprocessing:
					gray = im.convert('L')
					#gray.show()

					#blurred image:
					blurr = gray.filter(ImageFilter.BoxBlur(Cblur[i]))
					#blurr.show()

					#preprocesamos aun mas la imagen para encontrar los puntos limite
					edges = blurr.filter(ImageFilter.FIND_EDGES)
					#edges = edges.filter(ImageFilter.EDGE_ENHANCE)

					#como PIL no da las coordenadas, vamos a guardar los resultados en una imagen
					edges.save("Result.jpeg")
					#edges.show()

					#cv2 si nos ofrece una herramienta que suelta las coordenadas de los puntos, por lo tanto
					#abrimos la imagen con cv2
					img = cv2.imread("Result.jpeg")
					#creamos un objeto de tipo fast con threshold E, para detectar los puntos limite
					fast = cv2.FastFeatureDetector_create(int(FuncionesSeria.GetReal(Cthreshold[i],250,0)))
					#objeto que guarda las coordenadas de los puntos limite
					kp.append(fast.detect(img,None))
					#print("kp in for: ",kp)
					'''
					print("E-original: ",individuos[indexP1].threshold, " ",individuos[indexP2].threshold)
					print("E- normalized: ",Cthreshold[i])
					print("E int: ",int(FuncionesSeria.GetReal(Cthreshold[i])))
					print("E real",FuncionesSeria.GetReal(Cthreshold[i]))
					input()

				print("-----------------------------------")
				print("kp: ",len(kp[0]))
				print("kp2: ",len(kp[1]))
				#print("kp[0]: ",kp[0])
				print("kp: ",kp)
				print("-----------------------------------------")
				print("kp[0]: ",kp[0])
				print("-----------------------------------")
				print("kp[0][-1]: ",kp[0][-1])
				print("-----------------------------------")
				'''
				#Si la imagen es valida, rompo el while, y si no, lo intento de nuevo.
				if (len(kp[0])>0 and (len(kp[1])>0)):
					break

			for i in range(2):

				canvas = Image.new('RGB', im.size, (255,255,255,0))
				#guardamos todos los puntos limite en un arreglo de candidatos
				Candidates = np.array(  [ [kp[k][j].pt[0],kp[k][j].pt[1]]  for k  in range(len(kp)) for j in range(len(kp[k])) ]   ) 

				#sacamos una cantidad razonable de puntos limite, y los metemos en coordenadas
				for i in range(min(int(Cmaxpoints[i]),int(Crate[i]*width*height))):
				#for i in range(n):
					coordenadas.append(list(map(int,Candidates[np.random.randint(len(Candidates))])))

				#tenemos que colocar las esquinas del canvas
				coordenadas.append([0,0])
				coordenadas.append([0,height-1])
				coordenadas.append([width-1,0])
				coordenadas.append([width-1,height-1])
				Children.append(FuncionesSeria.Individuo(FuncionesSeria.GetReal(Cblur[i],5,0),FuncionesSeria.GetReal(Crate[i],5,0),int(FuncionesSeria.GetReal(Cmaxpoints[i],3000,1)),int(FuncionesSeria.GetReal(Cthreshold[i],250,0)),coordenadas,im,canvas,width,height) )

			#Ordenamos los hijos
			auxFitness = []
			for i in range(len(Children)):
				auxFitness.append(Children[i].fitness)
			indexes = sorted(range(len(Children)), key=lambda k: auxFitness[k])
	
			Children= FuncionesSeria.orden(indexes,Children)

			if Children[0].fitness<individuos[indexP1].fitness:

				individuos[indexP1] = Children[0]
				#print("Children0 maxpoint: ",Children[0].maxpoints)
				if Children[1].fitness<individuos[indexP2].fitness:
					individuos[indexP2] = Children[1]

			elif Children[0].fitness<individuos[indexP2].fitness:
					individuos[indexP2] = Children[0]

			#Ordenamos a los individuos
			auxFitness = []
			for i in range(len(individuos)):
				auxFitness.append(individuos[i].fitness)
			indexes = sorted(range(len(individuos)), key=lambda k: auxFitness[k])
			individuos = FuncionesSeria.orden(indexes,individuos)

		#Intento mutar a cada individuo.
		for j in range(len(individuos)):
			#normalizamos cada valor:
			#tengamos en cuenta que una biyeccion creciente del intervalo (0,1)-R es:
			# y = tan ( (2x-1 )*pi/2)
			#por lo tanto, su inversa sera:
			# x = atan(y)/pi +1/2
			#la cual tambien es creciente.
			


			#Aplicamos Ahora el mapa de (0,1)->R a cada resultado
			if np.random.randint(porcentaje_mut)>np.random.randint(102-porcentaje_mut):
				'''
				print("individual maxpoint: ",individuos[j].maxpoints)
				print("Nmaxpoints smol: ",Nmaxpoints)
				print("Nmaxpoints inverse: ",FuncionesSeria.GetReal(Nmaxpoints,3000,1))
				'''
				while True:
					Nblur = FuncionesSeria.GetSmol(individuos[j].blur,5,0)
					Nrate = FuncionesSeria.GetSmol(individuos[j].rate,5,0)
					Nmaxpoints = FuncionesSeria.GetSmol(individuos[j].maxpoints,3000,1)
					Nthreshold = FuncionesSeria.GetSmol(individuos[j].threshold,250,0)
					Nblur = FuncionesSeria.GetReal(FuncionesSeria.altMutation(genact,Nblur),5,0)
					Nrate = FuncionesSeria.GetReal(FuncionesSeria.altMutation(genact,Nrate),5,0)
					Nmaxpoints =int(FuncionesSeria.GetReal(FuncionesSeria.altMutation(genact,Nmaxpoints),3000,1 ) )
					Nthreshold = (int(FuncionesSeria.GetReal(FuncionesSeria.altMutation(genact,Nthreshold),250,0 ) ) )
					#Nmaxpoints =FuncionesSeria.GetReal(FuncionesSeria.altMutation(genact,Nmaxpoints) ) 
					#Nmaxpoints =FuncionesSeria.altMutation(genact,Nmaxpoints) 
					#print("Maxpoints smol after mutation: ",Nmaxpoints)
					#Nmaxpoints = FuncionesSeria.GetReal(Nmaxpoints,3000,1)
					#coordenadas del objeto:
					coordenadas = []
					

					#Grayscale image for preprocessing:
					gray = im.convert('L')
					#gray.show()

					#blurred image:
					blurr = gray.filter(ImageFilter.BoxBlur(Nblur))
					#blurr.show()

					#preprocesamos aun mas la imagen para encontrar los puntos limite
					edges = blurr.filter(ImageFilter.FIND_EDGES)
					#edges = edges.filter(ImageFilter.EDGE_ENHANCE)

					#como PIL no da las coordenadas, vamos a guardar los resultados en una imagen
					edges.save("Result.jpeg")
					#edges.show()

					#cv2 si nos ofrece una herramienta que suelta las coordenadas de los puntos, por lo tanto
					#abrimos la imagen con cv2
					img = cv2.imread("Result.jpeg")
					#creamos un objeto de tipo fast con threshold E, para detectar los puntos limite
					fast = cv2.FastFeatureDetector_create(Nthreshold)
					#objeto que guarda las coordenadas de los puntos limite
					kp = fast.detect(img,None)

					'''
					print("kp: ",kp)
					print("-----------------------------------------")
					print("kp[0]: ",kp[0])
					'''
					#Si la imagen es valida, rompo el while, y si no, lo intento de nuevo.
					if len(kp)>0:
						'''
						print("Nblur: ",Nblur)
						print("Nrate: ",Nrate)
						print("Nmaxpoints: ",Nmaxpoints)
						print("Nthreshold: ",Nthreshold)
						print("genact: ",genact)
						print("---------------------------------------------------------")
						input()
						'''
						break
				'''
				print("Nmaxpoints after mutation: ",Nmaxpoints)
				print("generacion: ",genact)
				input()
				Nmaxpoints = int(Nmaxpoints)
				'''
				#guardamos todos los puntos limite en un arreglo de candidatos
				Candidates = np.array(  [ [kp[k].pt[0],kp[k].pt[1]]  for k  in range(len(kp)) ]   ) 

				#sacamos una cantidad razonable de puntos limite, y los metemos en coordenadas
				for i in range(min(int(Nmaxpoints),int(Nrate*width*height))):
				#for i in range(n):
					coordenadas.append(list(map(int,Candidates[np.random.randint(len(Candidates))])))

				#tenemos que colocar las esquinas del canvas
				coordenadas.append([0,0])
				coordenadas.append([0,height-1])
				coordenadas.append([width-1,0])
				coordenadas.append([width-1,height-1])
				canvas = Image.new('RGB', im.size, (255,255,255,0))
				individuos[j] = FuncionesSeria.Individuo(Nblur,Nrate,Nmaxpoints,Nthreshold,coordenadas,im,canvas,width,height)


				for i in range(len(individuos)):
					auxFitness.append(individuos[i].fitness)
				indexes = sorted(range(len(individuos)), key=lambda k: auxFitness[k])
				#indexes.reverse()
				individuos = FuncionesSeria.orden(indexes,individuos)
				#Ordenamos a los individuos
				auxFitness = []
				for i in range(len(individuos)):
					auxFitness.append(individuos[i].fitness)
				indexes = sorted(range(len(individuos)), key=lambda k: auxFitness[k])
				individuos = FuncionesSeria.orden(indexes,individuos)
		'''
		print("---------------------------------------------------------")
		for i in range(len(individuos)):
			print("individuo "+str(i)+": by fit ",individuos[i].fitness)
			print("--------------------------------------")
			print("individuo blur: ",individuos[i].blur)
			print("Individuo rate: ",individuos[i].rate)
			print("individuo max point: ",individuos[i].maxpoints)
			print("individuo threshold: ",individuos[i].threshold)
			print("individuo fitness: ",individuos[i].fitness)
			individuos[i].TriDrawColor()
			individuos[i].FloodFillBFS()
			individuos[i].displayImg()
			os.rename("Result.jpg", "ResultInd"+str(i)+"Gen"+str(genact)+".jpg")
			print("---------------------------------------------------------")
		'''

print("---------------------------------------------------------")
for i in range(len(individuos)):
	print("individuo "+str(i))
	print("individuo blur: ",individuos[i].blur)
	print("Individuo rate: ",individuos[i].rate)
	print("individuo max point: ",individuos[i].maxpoints)
	print("individuo threshold: ",individuos[i].threshold)
	print("individuo fitness: ",individuos[i].fitness)

	individuos[i].TriDrawColor()
	individuos[i].FloodFillBFS()
	individuos[i].FillBlanks()
	individuos[i].displayImg(ruta_destino)
	os.rename(ruta_destino+"/Result.jpg", ruta_destino+"/ResultFinal"+str(i)+"jpg")
	print("---------------------------------------------------------")
	#input()

