from PIL import Image, ImageDraw 
from random import randint		


def stega_encrypt():	
	
	keys = []
	img = Image.open(input("путь к картинке: "))
	draw = ImageDraw.Draw(img)	
	width = img.size[0] 		   		
	height = img.size[1]
	pix = img.load()			
	f = open('keys.txt','w')

	for elem in ([ord(elem) for elem in input("текст для шифрования: ")]):
		key = (randint(1,width-10),randint(1,height-10))		
		g, b = pix[key][1:3]
		draw.point(key, (elem,g , b))														
		f.write(str(key)+'\n')								
	
	print('ключи были записаны в файл')
	img.save("newimage.png", "PNG")
	f.close()
												
stega_encrypt()
