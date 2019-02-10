## Обработка изображении 

Знакомство с методами обработки изображения с использованием tensorflow. Рассмотрим способы работы со сверточными нейронными сетями TensorFlow на примере сверток изображения (фильтров). 

Конвертация изображения в требуемую тразмерность:
 [NHWC to NCHW](https://stackoverflow.com/questions/37689423/convert-between-nhwc-and-nchw-in-tensorflow)
 
 
#### Размытие

Наиболее простой способ сделать размытие - Box фильтр
 
 ![](log/source_image.jpg)  ![](log/box_filter.jpg)<br>

#### Затемнение

Сумма элементов ядра Box фильтра должны в сумме = 1. Исли меньше 1, то будет затемненение  
 
 ![](log/source_image.jpg)  ![](log/box_filter_dark.jpg)<br>

#### Освещение

Сумма элементов ядра Box фильтра должны в сумме = 1. Исли больше 1, то будет засвет
 
 ![](log/source_image.jpg)  ![](log/box_filter_light.jpg)<br>
 
 
#### Сохранение размера
 
При применении сверточных сетей можно сохранять размер с помощью флага SAME - 
однако с ним появляется затемненная рамка в изображении  
 
![](log/source_image.jpg)  ![](log/box_filter_same.jpg)<br>
 
#### Черно-белый фильтр
  
![](log/source_image.jpg)  ![](log/wbImage.jpg)<br>

 
#### Обрезка

с использованием keras
  
![](log/source_image.jpg)  ![](log/scroppingImage.jpg)<br>


#### Приведение к размеру масштабированием

с использованием keras
  
![](log/source_image.jpg)  ![](log/scallingImage.jpg)<br>
