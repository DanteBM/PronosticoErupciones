Este documento intenta explicar qué contiene cada csv para que no hayan confusiones con las versiones: 

- df\_test\_constants.csv: Es un arreglo booleano que nos dice si en el conjunto de archivos de prueba hay algún archivo que en lugar de ser una señal es una constante. Lo tiene para cada sensor de cada documento. Creo que finalmente resulta que hay 10 archivos que son una constante. 

- df\_test\_nans.csv: Se tienen cuántos valores nulos por archivo y por sensor hay para los datos de prueba. 

- df\_train\_constants.csv: Contiene lo mismo que "df\_test\_constants.csv" pero para los archivos de entrenamiento. 

- df\_train\_nans.csv: Contiene lo mismo que "df\_test\_nans.csv" pero para los archivos de entrenamiento. 

- sample\_submission.csv: Contiene un ejemplo del archivo que hay que subir para la evaluación de kaggle.

- stats\_per\_file.csv: Contiene datos estadísticos de cada sensor para cada archivo. Lo que se contempló aquí fueron la suma, promedio, desviación estándar, mínimo y máximo.

- stats\_per\_file\_2.csv: Contiene los mismos datos que el archivo "stats\_per\_file.csv" y la cantidad de datos por sensor por archivo, pero además aquí se toman en cuenta los nans. 

- stats\_per\_file\_3.csv: Contiene los mismos datos que el archivo "stats\_per\_file\_2.csv" pero creo que aquí se hizo con los datos estandarizados (sin imputar).

- stats\_per\_file\_signal.csv: Contiene características de las señales de cada archivo y de cada sensor. Las características que se encuentran aquí son tasa de cruce por el cero, número de picos, la anchura del pico más ancho, el promedio de la anchura de los picos, la altura del pico más alto, el promedio de las alturas de los picos, el máximo del periodograma (que es un estimado del RMS) y el promedio del periodograma. Todo esto para el conjunto de entrenamiento de los datos **estandarizados**.

- stats\_per\_file\_signal_test.csv: Contiene los mismos datos que el archivo "stats\_per\_file\_signal.csv" pero para el conjunto de pruebas **estandarizado**. 

- stats\_per\_file\_signal\_test2.csv: Contiene los mismos datos que el archivo "stats\_per\_file\_signal_train.csv" pero para el conjunto de pruebas **estandarizado2**.

- stats\_per\_file\_signal\_test\_estandarizados.csv: Contiene lo mismo que "stats\_per\_file\_signal_test.csv" pero con la corrección de los datos nulos en el periodograma. Se usan los datos **estandarizados** del conjunto de pruebas. 

- stats\_per\_file\_signal\_test\_estandarizados2.csv: Contiene lo mismo que el archivo "stats\_per\_file\_signal\_test\_estandarizados.csv" pero con los datos **estandarizados2**. 

- stats\_per\_file\_signal\_test\_normal.csv: Contiene lo mismo que el archivo "stats\_per\_file\_signal\_test\_estandarizados.csv" pero con los datos sin estandarizar.

- stats\_per\_file\_signal_train2.csv: Contiene los mismos datos que el archivo "- stats\_per\_file\_signal.csv:" pero para los datos **estandarizados2** que son los que se les hizo el proceso de imputación.

- stats\_per\_file\_signal\_train\_estandarizados.csv: Contiene lo mismo que "stats\_per\_file\_signal\_test\_estandarizados.csv" pero para el conjunto de entrenamiento.

- stats\_per\_file\_signal\_train\_estandarizados2.csv: Contiene lo mismo que "stats\_per\_file\_signal\_test\_estandarizados2.csv" pero para el conjunto de entrenamiento.

- stats\_per\_file\_signal\_train\_normal.csv: Contiene lo mismo que "stats\_per\_file\_signal\_test\_normal.csv" pero para el conjunto de entrenamiento.

- submission.csv: Contiene el primer archivo que se subió a kaggle para evaluación. Se hizo ya teniendo en cuenta el proceso de características de señales pero sin tener en cuenta el overfitting.

- submission2.csv: Contiene el segundo archivo que se subió a kaggle para evaluación. Aquí la única diferencia es que se entrenó el modelo con menos datos pensando que podría servir para el overfitting. 

- submission3.csv: Contiene el tercer archivo que se subió a kaggle para evaluación. Se tiene en cuenta aquí lo de _early\_stopping\_rounds_ para tratar de evitar el sobreajuste, se entrena el modelo con todos los datos de entrenamiento y con los datos estandarizados2. 

- submission4.csv: Contiene el cuarto archivo que se subió a kaggle para evaluación. Se tiene en cuenta aquí lo de _early\_stopping\_rounds_ y se hace la validación cruzada con 10 folds para tratar de evitar el sobreajuste, se entrena el modelo con todos los datos de entrenamiento y con los datos **estandarizados2**.

- submission5.csv: Contiene el quinto archivo que se subió a kaggle para evaluación. Lo mismo que en "submission4.csv" pero ahora con los datos **estandarizados**.

- submission6.csv: Contiene el sexto archivo que se subió a kaggle para evaluación. Se usa CV y ESR en los datos sin estandarizar pero ahora se hace el análisis de señales chido, ya se tiene en cuenta esto de sólo utilizar los datos no nulos en el periodograma.

- submission\_estandarizados.csv: Archivo que se subió a kaggle para evaluación, se usa CV y ESR con los datos estandarizados y con la mejora del periodograma.

- submission\_estandarizados2.csv: Contiene lo mismo que "submission\_estandarizados.csv" pero ahora con los datos imputados.

- submission\_normal.csv: Contiene lo mismo que "submission\_estandarizados.csv" pero con los datos sin estandarizar y sin imputar. 

- train.csv: Contiene el "segment\_id" y el "time\_to\_eruption" de los archivos de entrenamiento.

- whole\_stats.csv: Contiene las estadísticas generales de los 10 sensores tomando en cuenta todos los archivos de entrenamiento. Es como un _describe_ general. 
