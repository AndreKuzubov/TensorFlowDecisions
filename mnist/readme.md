## MNIST

Разпознование цифр MNIST с помощью сверточных сетей 

#### На базе перцептрона 
[mnist_perceptron_2layers.py](mnist_perceptron_2layers.py)   

2 слойный перцептрон  

#### На Свертки  
[mnist_conv.py](mnist_conv.py)

[mnist_keras.py](mnist_keras.py) - с использованием библиотеки keras

[mnist_detection.py](mnist_detection.py) - распознование с обнаружением

2х слойная сверка + 2х слойный перцептон:
   
  <b>Запуск:</b>
     
            >> cd <dir>/mnist
            >> python mnist_conv.py
            >> tensorboard --logdir=log/mnist_conv/tmp 
        
     далее пройти по ссылке: [http://localhost:6006](http://localhost:6006)


 

