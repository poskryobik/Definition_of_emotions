<b>Постановка задачи</b>

Компания владеет сервисом по предоставлению доступа к сериалам. Чтобы завоевать интерес зрителей, им демонстрируют трейлеры — небольшие видеоролики, состоящие из наиболее зрелищных моментов сериалов.

Для определения зрелищности момента в сериале зрители тестовой группы смотрят сериал, а их эмоции записываются на камеру. С ростом числа сериалов, готовых к запуску, отсматривать эмоции зрителей в ручном режиме экономически нецелесообразно. Для этих целей необходимо реализовать нейронную сеть по классификации эмоций, а результаты данной классификации будут использованы для формирования трейлеров.

<b>Требования заказчика к нейронной сети:</b>
1) Время инференса сети на Google Colab не должно превышать 0,33 секунды (3 кадра в секунду).
2) Применить finetuning предобученной на лицах модели ResNet50. 


<b>Ход работы:</b>

1)Подготовка данных. Применение аугментации (искусственного увеличения датасета) для обучения нейронной сети.

 	
	image_gen = ImageDataGenerator(preprocess_input_facenet, 
                               	brightness_range = (0.5,1),
                               	rotation_range= 25,
                               	width_shift_range=0.1,
                               	height_shift_range=0.1,
                               	horizontal_flip=True,
                               	validation_split=0.2,
                               	shear_range = 2)
	
	
2)Finetuning модели ResNet50.
	
	#Загружаем модель
	vggface_model = load_model(/content/resnet50face.h5)
	#Отсекаем ненужные слои и замораживаем веса 
	base_model =  tf.keras.Model([vggface_model.input], vggface_model.get_layer('avg_pool').output)
	base_model.trainable = False
	#Добавляем сверточные слои. Сверточные слои с ядром 1х1 применяются для уменьшения количества каналов, 
	#для минимизации количества весов и возможного переобучения.
	model = tf.keras.Sequential([
		base_model,
		tf.keras.layers.Conv2D(256, (3, 3), dilation_rate=6, padding='same', activation='relu'),
		tf.keras.layers.Conv2D(128, (1, 1), dilation_rate=6, padding='same', activation='relu'),
		tf.keras.layers.Conv2D(128, (3, 3), dilation_rate=6, padding='same', activation='relu'),
		tf.keras.layers.Conv2D(64, (1, 1), dilation_rate=6, padding='same', activation='relu'),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(units=64, activation= tf.keras.activations.relu),
		tf.keras.layers.Dense(units=9, activation= tf.keras.activations.softmax)
	])

			
3)Применение детектора лица из OpenCV для классификации эмоции только на части изображения.
	
	
	#Инициализация объекта класса
	def __init__(self):
		#Загружаем модель
		self.model = load_model('checkpoint_best.h5')
		#Список классов
		self.classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 
				'neutral', 'sad', 'surprise', 'uncertain']
		#Объект камера
		self.cam = cv2.VideoCapture(0)
		#Инициализируем детектор:
		self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	#Получение изображения
	def get_frame(self):
		ret, self.frame = self.cam.read()
		if ret:
			return self.frame
	
<b>Как использовать?</b>
1) Для обучения модели по распознованию эмоций запустить <b>model_1.ipynb</b> (по условию задачи запускается из GoogleColab)</br>
	*Все необходмые ссылки для скачивания определены константами;</br>
	*Все скачанные файлы располагаются в корне проекта.
2) Для применения обученной модели на веб-камере необходимо использовать <b>Camera.ipynb</b> (по условию задачи применяется на локальной машине)</br>
	*Обученную модель разместить в корне проекта;</br>
	*Для вывода видео создается объект класса: <code> obj  = emotion_on_camera() </code>, после вызывается метод <code> obj.return_video()</code>;</br> 
	*Для выхода из режима видео нажать 'q'. </br>
	
