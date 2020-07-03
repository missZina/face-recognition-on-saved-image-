import os
import cv2 
import numpy as np
from matplotlib import pyplot

    
def unFlatten(vector,rows, cols):
    img = []
    cutter = 0
    while(cutter+cols<=rows*cols):
        try:
            img.append(vector[cutter:cutter+cols])
        except:
            img = vector[cutter:cutter+cols]
        cutter+=cols
    img = np.array(img)
    return img
 
 
w=200
h=200
 
# Construct the input matrix
 
face='....../My Drive/dataset FREC/TRAIN'
 
def main():
	for v in face:
		in_matrix = None 
		imgcnt=0
		print ('Read from: ' + v + ' Directory ')
		for f in os.listdir(os.path.join('training/',v)):
		    imgcnt+=1
		    print(f)
		    # Read the image in as a gray level image. 
		    img = cv2.imread(os.path.join('/content/drive/My Drive/dataset FREC/TRAIN/',v, f), cv2.IMREAD_GRAYSCALE)
		    img_resized = cv2.resize(img,(w,h))
 
		    # let's resize them to w * h 
		    vec = img_resized.reshape(w * h)
		    
		    # stack them up to form the matrix
		    try:
		        in_matrix = np.vstack((in_matrix, vec))
		    except:
		        in_matrix = vec
		    
		    # PCA 
		if in_matrix is not None:
		    mean, eigenvectors = cv2.PCACompute(in_matrix, np.mean(in_matrix, axis=0).reshape(1,-1))
 
		img = unFlatten(mean.transpose(), w, h) #Reconstruct mean to represent an image
		cv2.imwrite('......../dataset FREC/TRAIN'+v+'.jpg',img)
 
import os
import cv2 
import numpy as np
 
#Image Size
w=200
h=200
 
#euclidean distance
def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))
 
inputImageToPredict= '.........../dataset FREC/TEST/aabed fahd/aabed fahd4.jpg'
 
 
def main():
	img = cv2.imread(inputImageToPredict, cv2.IMREAD_GRAYSCALE) # get grayscale image
	img_resized = cv2.resize(img,(w,h))
	      
	imgBlurred = cv2.GaussianBlur(img_resized, (5,5), 0)  # blur
 
	imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #threshold
 
 
	#cv2.imshow('Test Image',imgThresh)
  plt.imshow(img)
 plt.title('test image')
  plt.show()
 
	npaFlattenedImage = imgThresh.reshape((1, w * h)) 
	print (npaFlattenedImage.shape)
	mean_to= npaFlattenedImage.mean()
 
	m=[]
 
	person=['aabed fahd','bassem yakhour','haby','hessin fahmi','mustafa kamar','nadine nassib','mahmoud abdelaziz']
	cnt=0
	for v in vowels:
		print ('Read ' + v + ' mean image from directory !')
		f= '........../dataset FREC/TRAIN'+v+'.ipgg'
		print(f)
		
		img2 = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
		img_resized2 = cv2.resize(img2,(w,h))
		imgBlurred2 = cv2.GaussianBlur(img_resized2, (5,5), 0)      
		imgThresh2 = cv2.adaptiveThreshold(imgBlurred2, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		npaFlattenedImage2= imgThresh2.reshape((1, w * h))  
 
		distance = dist(npaFlattenedImage, npaFlattenedImage2)
		#distance vector
		m.append( distance );
 
	#Distance array
	print ('Euclidean Distance Array: ')
	print (m)
	#Min Distance
	print ('Min Distance: ')
	print (min(m))
	#Array Position
	print ('Array Position: ')
	pos=m.index(min(m))
	print (pos)
	#person Recognized
	print ('The Face Recognized Is : ')
	
