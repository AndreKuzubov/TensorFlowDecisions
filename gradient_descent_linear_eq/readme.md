## Градиентный спуск 

<b>Постановка задачи:</b> Дан набор точек <img src="https://latex.codecogs.com/svg.latex?(x,y)"/>, надо найти коэффициенты <img src="https://latex.codecogs.com/svg.latex?(k,l)" />   для функции прямой типа
<img src="https://latex.codecogs.com/svg.latex?y=kx+l"/>.  


<b>Решим</b> задачу методом градиентного спуска, по ошибке равной средней квадратичной (НКО), тогда функция ошибки f будет равна

<img src="https://latex.codecogs.com/svg.latex? f(k,l) = (kx+l-y)^2"/>.  


Зная некоторые 
<img src="https://latex.codecogs.com/svg.latex?k_i"/> 
и <img src="https://latex.codecogs.com/svg.latex?l_i"/> 
 мы можем найти:
 
<img src="https://latex.codecogs.com/svg.latex?k_{i+1} = k_i - \lambda f'_k (k_i,l_i) "/> <br>
 
<img src="https://latex.codecogs.com/svg.latex?l_{i+1} = l_i -  \lambda f'_l (k_i,l_i)"/> <br>
 
Раскроем частные производные: 

<img src="https://latex.codecogs.com/svg.latex?k_{i+1} = k_i - 2 \lambda x (k_i x + l_i-y) "/> <br>
 
<img src="https://latex.codecogs.com/svg.latex?l_{i+1} = l_i - 2 \lambda  (k_i x + l_i-y)"/> <br>
 



#### Результаты работы 

 ![](log/linear_eq/loss_scalars.png)
    _Изменение ошибки градиентного спуска_
    
    
 ![](log/linear_eq/k_l_scalars.png)
     _Изменение k и l по мере градиентного спуска_

 

 

#### Структура проекта

*[X] linear_eq.py - работа апроксиматора без применения библиотеки tensorflow
     
     <b>Запуск:</b>
            
            >> cd <dir>/gradient_descent_linear_eq    
            >> python linear_eq.py
                 
  
     
     
*[X] linear_eq_tensorflow.py - работа апроксиматора с применением библиотеки tensorflow, с показом результатов через tensorboard 
 
     <b>Запуск:</b>
     
            >> cd <dir>/gradient_descent_linear_eq
            >> python linear_eq_tensorflow.py
            >> tensorboard --logdir=log/linear_eq_tensorflow/tmp 
        
     далее пройти по ссылке: [http://localhost:6006](http://localhost:6006)