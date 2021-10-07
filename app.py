from flask import Flask,request,render_template
import os

#importing custom packages
from xray_prediction.prediction import Prediction
from xray_train.training import Training

#creating app
app = Flask(__name__)

#creating objects
predict = Prediction()

#index page
@app.route('/')
def index():
    return render_template('index.html',alert = None,style="display:None;")

@app.route('/train',methods=['POST'])
def training():
    if request.method == 'POST':
        if request.json:
            data = request.json
            train = Training(batch_size=data["batch_size"],epochs=data["epochs"],
                             num_workers=data["num_workers"],learning_rate=data["learning_rate"],
                             momentum=data["momentum"],dataset_path = data["dataset_path"],
                             model_save_folder=data["model_save_folder"])
            train_acc, valid_acc = train.train_model()
            return f"Training Accuracy: {train_acc} \n Validation Accuracy: {valid_acc}"


@app.route("/predict",methods=['POST'])
def prediction():
    if request.method == 'POST':
        if request.files:
            image = request.files["image"]
            path= image.filename
            image.save(path)
            out = predict.predict(path)
            os.remove(path)
            if out=="covid":
                alert = 0
            else:
                alert = 1
            return render_template('index.html',alert = alert,style="font-weight: bold;")
        
        if request.json:
            data = request.json
            path = data["image_path"]
            out = predict.predict(path)
            if out=="covid":
                return "THE PERSON HAS COVID"
            else:
                return "THE PERSON DOES NOT HAVE COVID"

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=PORT)