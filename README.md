# sexismDetector
A intelligent system that is capable of recognizing and detecting sexism in social networks

Para la iteración inicial simplemente hice un clasisficador binario básico con fastText
Introduje algunas mejoras para la clasificación, como retirar caracteres irrelevantas, números e incluso pasar el texto todo a minúscula
Luego, para incrementar la precisión, decidí separar el entrenamiento en inglés y castellano, y llamar al modelo necesario segun el idioma en el que se encuentre el tweet
Esto aumentó la precisión en inglés hasta el 67%, pero en castellano solo alcanza el 54%. Con los dos mezclados llegaba a alcanzar el 63%

python -m nltk.downloader stopwords

Tras instalar unas librerías a mayores y eliminar palabras redundantes en inglés, he conseguido mejorar la precisión hasta el 70%
Quité también enlaces y menciones.
Muy contento con este resultado, pero por ahora el castellano sigue teniendo un mísero 54%
Reduciendo el ratio de entrenamiento al 15% mejora mucho el castellano, llegando al 58%
Creé nueva función para parsear el castellano, elimina caracteres innecesarios y deja el resto. No sabia usar bien regex así que queda algo feo
Aumentó la precisión al 59-63%