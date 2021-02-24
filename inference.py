import os
import sys
import imutils
import math
import tarfile
from six.moves import urllib
import numpy as np
from PIL import Image
import cv2, argparse
import tensorflow as tf



class DeepLabModel(object):
	"""Class to load deeplab model and run inference."""

	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513
	FROZEN_GRAPH_NAME = 'frozen_inference_graph'

	def __init__(self, tarball_path):
		#"""Creates and loads pretrained deeplab model."""
		self.graph = tf.Graph()
		graph_def = None
		# Extract frozen graph from tar archive.
		tar_file = tarfile.open(tarball_path)
		for tar_info in tar_file.getmembers():
			if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
				file_handle = tar_file.extractfile(tar_info)
				graph_def = tf.GraphDef.FromString(file_handle.read())
				break

		tar_file.close()

		if graph_def is None:
			raise RuntimeError('Cannot find inference graph in tar archive.')

		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')

		self.sess = tf.Session(graph=self.graph)

	def run(self, image):
		"""Runs inference on a single image.

		Args:
		  image: A PIL.Image object, raw input image.

		Returns:
		  resized_image: RGB image resized from original input image.
		  seg_map: Segmentation map of `resized_image`.
		"""
		width, height = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
		target_size = (int(resize_ratio * width), int(resize_ratio * height))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		batch_seg_map = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
		seg_map = batch_seg_map[0]
		return resized_image, seg_map



parser = argparse.ArgumentParser(description='Deeplab Segmentation')
parser.add_argument('-i', '--input_dir', type=str, required=True,help='Directory to read Images. (required)')
parser.add_argument('-ht', '--height', type=int, required=True,help='height of person. (required)')
args=parser.parse_args()

MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
	'mobilenetv2_coco_voctrainaug':
		'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
	'mobilenetv2_coco_voctrainval':
		'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
	'xception_coco_voctrainaug':
		'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
	'xception_coco_voctrainval':
		'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = _MODEL_URLS[MODEL_NAME]

model_dir = 'deeplab_model'
if not os.path.exists(model_dir):
  tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
if not os.path.exists(download_path):
  print('downloading model to %s, this might take a while...' % download_path)
  urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], 
			     download_path)
  print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')

#######################################################################################

for img_name in os.listdir(args.input_dir):
    

    image = Image.open(args.input_dir + '/' + img_name)
    
        
    res_im,seg=MODEL.run(image)
    
    seg=cv2.resize(seg.astype(np.uint8),image.size)
    mask_sel=(seg==15).astype(np.float32)
    mask = 255*mask_sel.astype(np.uint8)
    
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   
    
    res = cv2.bitwise_and(img,img,mask = mask)
    kernel = np.ones((3,5),np.uint8)
   
    if 'side' == os.path.splitext(img_name)[0]:
        sidePose = img#res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        sideMask = cv2.erode(mask,kernel,iterations = 1)
        no_erosion = mask
    else:
        print("Error opening Image")
        sys.exit()
        
    
    erosion = cv2.erode(mask,kernel,iterations = 1)
    res_ero = cv2.bitwise_and(img,img,mask = erosion)



# find contours in thresholded image, then grab the largest

cnts = cv2.findContours(no_erosion.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)


# determine the most extreme points along the contour
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])


extBotNew = extBot[0], extBot[1] -5
body_cut = mask[extBotNew[1],:] 

body_pix_index = np.where(body_cut == 255)
point = (body_pix_index[0][0], extBotNew[1])


height_pixel = math.hypot(extTop[0]-extBot[0], extTop[1]-extBot[1])
foot_length = math.hypot(extBot[0]-extRight[0], extBot[1]-extRight[1])

extRightNew = extRight[0] , point[1]

foot_length = (args.height/height_pixel) * foot_length

cv2.arrowedLine( sidePose, tuple(point), tuple(extRightNew) , (0,0,255), 2)

# show the output image
cv2.imshow("res", sidePose)

print('length of feet in cm = ', foot_length)
cv2.waitKey(0)
cv2.destroyAllWindows
cv2.imwrite("result/side.png",sidePose)