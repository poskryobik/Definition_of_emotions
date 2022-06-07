<b>Постановка задачи</b>

Компания владеет сервисом по предоставлению доступа к сериалам. Чтобы завоевать интерес зрителей, им демонстрируют трейлеры — небольшие видеоролики, состоящие из наиболее зрелищных моментов сериалов.

Для определения зрелищности момента в сериале зрители тестовой группы смотрят сериал, а их эмоции записываются на камеру. С ростом числа сериалов, готовых к запуску, отсматривать эмоции зрителей в ручном режиме экономически нецелесообразно. Для этих целей необходимо реализовать нейронную сеть по классификации эмоций, а результаты данной классификации будут использованы для формирования трейлеров.



<b>Как использовать?</b>
1) Для обучения модели по распознованию эмоций запустить model_1.ipynb (по условию задачи запускается из GoogleColab)
	*Все необходмые ссылки для скачивания определены константами.
	*Все скачанные файлы располагаются в корне проекта
2) Для применения обученной модели на веб камере необходимо использовать Camera.ipynb (по условию задачи применяется на локальной машине)
	*Обученную модель разместить в корне проекта
	*Для вывода видео создается объект класса: obj  = emotion_on_camera(), после вызывается метод obj.return_video()
	*Для выхода из режима видео нажать нажать 'q'
	
