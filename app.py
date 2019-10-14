from flask import Flask, request, render_template
app = Flask(__name__)
from commons import get_tensor
from inference import get_flower_name

#library for image detection
from imageai.Detection import ObjectDetection
import os
execution_path = os.getcwd()


@app.route('/', methods=['GET', 'POST'])
def hello_world():
	if request.method == 'GET':
		return render_template('index.html', value='hi')


	if request.method == 'POST':
		print(request.files)
		if 'file' not in request.files:
			print('file not uploaded')
			return
		file = request.files['file']
		image = file.read()
		category, flower_name = get_flower_name(image_bytes=image)
		get_flower_name(image_bytes=image)
		tensor = get_tensor(image_bytes=image)
		print(get_tensor(image_bytes=image))

		detector = ObjectDetection()
		detector.setModelTypeAsYOLOv3()
		detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
		detector.loadModel()
		detections = detector.detectObjectsFromImage(input_image=file, output_image_path="c:\\Users\\Watson\\Desktop\\Project_Insta - ImageAI\\deploy_example\\example code\\static\\image2new.jpg", minimum_percentage_probability=30)

		for eachObject in detections:
			print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
			print("--------------------------------")



		return render_template('result.html', object_0 =detections[0]['name'], percentage_probability_0=round(detections[0]['percentage_probability']), object_1 =detections[1]['name'], percentage_probability_1=round(detections[1]['percentage_probability']))
				#return render_template('result.html', flower=flower_name, category=category)

# return render_template('result.html', object_1 =detections[1]['name'], percentage_probability_1=detections[1]['percentage_probability'], object_2 =detections[2]['name'], percentage_probability_2=detections[2]['percentage_probability'])
		

if __name__ == '__main__':
	app.run(debug=True)




# detector = ObjectDetection()
# detector.setModelTypeAsYOLOv3()
# detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
# detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image2.jpg"), output_image_path=os.path.join(execution_path , "image2new.jpg"), minimum_percentage_probability=30)

# for eachObject in detections:
#     print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
#     print("--------------------------------")