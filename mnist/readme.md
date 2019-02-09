## MNIST

Разпознование цифр MNIST с помощью сверточных сетей 

#### На базе перцептрона 
[mnist_perceptron_2layers.py](mnist_perceptron_2layers.py)   

2 слойный перцептрон  

#### На Свертки  
[mnist_conv.py](mnist_conv.py)

2х слойная сверка + 2х слойный перцептон:
   
  <b>Запуск:</b>
     
            >> cd <dir>/mnist
            >> python mnist_conv.py
            >> tensorboard --logdir=log/mnist_conv/tmp 
        
     далее пройти по ссылке: [http://localhost:6006](http://localhost:6006)


#### Оценка устойчивости

[mnist_scale_stability.py](mnist_scale_stability.py)

Оценка устойчивости предсказания сверточной нейронной сети для:
    
  - Изображений разных размеров (без масштабирования) 
  - Изображений разных размеров (с масштабирования) 
  - При нескольких объектах на одном изображении
  
  
Реализовано с библиотекой keras  
    
 

