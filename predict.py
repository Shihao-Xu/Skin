from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

def predict_img(img_path):

	# dimensions of our images    -----   are these then grayscale (black and white)?
	img_width, img_height = 128, 128

	# load the model we saved
	model = None
	try:
#		with keras.backend.get_session
		model = load_model('/home/JingyuXu/result/keras_model.h5')
	except:
		return "Load Model Error", 100
	# Get test image ready
	test_image = None
	try:
		test_image = image.load_img(img_path, target_size=(img_width, img_height))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis=0)
		test_image = test_image.reshape(1, img_width, img_height, 3)    # Ambiguity!
	# Should this instead be: test_image.reshape(img_width, img_height, 3) ??
	except:
		return "Get Test Image Error", 200
	result = [2] * 2
	try:
		result[1] = model.predict_proba(test_image, batch_size=1)[0][0]
		label = model.predict_classes(test_image, batch_size=1)[0][0]
		if label == 0:
			result[0] = 'benign'
		elif label == 1:
			result[0] = 'malignant'
		else:
			result[0] = 'unknown'
	except:
		return "Get Predict Error", 300
	print(result)
	return result

#img_path = '/home/JingyuXu/images/train/benign/ISIC_0024307.jpg'
#img_path = '/home/JingyuXu/skin/static/images/test.jpg'
#p = predict_img(img_path)
