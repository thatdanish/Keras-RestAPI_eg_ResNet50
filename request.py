import requests

url = 'http://localhost:3000/predict'
image_path = 'dog.jfif'

image = open(image_path,'rb').read()
payload = {'image':image}

r = requests.post(url,files=payload).json()

if r['success']:
    for (i,result) in enumerate(r['predictions']):
        print(i,result)

else:
    print('Request Failed')