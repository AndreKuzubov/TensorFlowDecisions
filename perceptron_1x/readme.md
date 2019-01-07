## Однослойный перцептрон

Однослойный перцептрон можно описать с использованием матричного подхода

<img src="https://latex.codecogs.com/svg.latex?Y=f(\sum{XW}+L)"/>

,где f - функция активации;
x - входной вектор;
w - матрица весов синапсов ИНС;
y - выходной вектор.   

#### Структура проекта

* [X] [binary_number_classification.py](https://github.com/AndreKuzubov/TensorFlowDecisions/blob/dev/perceptron_1x/binary_number_classification.py)  - 
    
    Рассмотрим простейший пример нейронный сети на основе перцептрона
    из входного вектора размерностью 1
    ИНС размерностью 1x4
    выходного вектора размерностью 1

   <b>Постановка задачи:</b>
    обучить ИНС выбирать числа больше 0.8

    <b>Решение:</b>
    Проведем обучение классификации с учителем с использованием градиентного спуска

    Выберем функцию активации - сигмойда 

    <img src="https://latex.codecogs.com/svg.latex?\sigma(x)=\frac{1}{1+e^{-x}}"/>,

    функцию ошибки - квадрат разности 

    <img src="https://latex.codecogs.com/svg.latex?e(W)=(y_{pred}-y)^2"/>

    ,где <img src="https://latex.codecogs.com/svg.latex?Y_{pred}=f(\sum{XW}+L)"/>

    ,а  <img src="https://latex.codecogs.com/svg.latex?Y"/> - фактические результаты

     <b>Запуск:</b>
     
            >> cd <dir>
            >> python perceptron_1x/binary_number_classification.py
            >> tensorboard --logdir=log/binary_number_classification/tmp 
        
     далее пройти по ссылке: [http://localhost:6006](http://localhost:6006)
     
     <b>Результаты:</b>

     ![classifications.gif](https://github.com/AndreKuzubov/TensorFlowDecisions/blob/dev/perceptron_1x/log/binary_number_classification/classifications.gif?raw=true)


## Выводы

- Каждый элемент выходного слоя <img src="https://latex.codecogs.com/svg.latex?Y"/> может 
    выполнять только бинарную операцию классификации. 
    Поэтому при обучении нейронных сетей на каждом последующем слое необходимо проверять, 
    что по завершении обучения диссперсия ошибки для каждого отдельного нейрона не превышала значения 0.3 или около того, при мат. ожидании равном нулю.
- Если вы обучаете однослойный перцептрон, то каждый элемент выходного вектора должен отвечать за бинарную логику.     
    
