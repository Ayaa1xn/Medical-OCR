from flask import Flask, render_template, request
import numpy as np
from ChatGPTApi import ChatGPTApi 
import pickle

diabetes_model = pickle.load(open('models/diabetes.pkl', 'rb'))
cancer_model = pickle.load(open('models/cancer.pkl', 'rb'))
heart_model = pickle.load(open('models/heart.pkl', 'rb'))
liver_model = pickle.load(open('models/liver.pkl', 'rb'))
kidney_model = pickle.load(open('models/kidney.pkl', 'rb'))

app = Flask(__name__)


UPLOAD_FOLDER = './static/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/summarizer')
def summarizer():
    return render_template('upload_report.html',type="Report Analyzer")


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/diabetes", methods=['GET','POST'])
def diabetes():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET','POST'])
def cancer():
    return render_template('cancer.html')

@app.route("/heart", methods=['GET','POST'])
def heart():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET','POST'])
def kidney():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET','POST'])
def liver():
    return render_template('liver.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if(len([float(x) for x in request.form.values()])==8):
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])
            
            data = np.array([[preg,glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = diabetes_model.predict(data)
            
            return render_template('predict.html', prediction=my_prediction)
        elif(len([float(x) for x in request.form.values()])==10):
            Age = int(request.form['Age'])
            Total_Bilirubin = float(request.form['Total_Bilirubin'])
            Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
            Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
            Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
            Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
            Total_Protiens = float(request.form['Total_Protiens'])
            Albumin = float(request.form['Albumin'])
            Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])
            Gender_Male = int(request.form['Gender_Male'])

            data = np.array([[Age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Gender_Male]])
            my_prediction = liver_model.predict(data)
            return render_template('predict.html', prediction=my_prediction)

        elif(len([float(x) for x in request.form.values()])==13):
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
            data1 = np.array(data).reshape(1,-1)
            my_prediction = heart_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)

        elif(len([float(x) for x in request.form.values()])==18):
            age = float(int(request.form['age']))
            bp = float(request.form['bp'])
            al = float(request.form['al'])
            su = float(request.form['su'])
            rbc = int(request.form['rbc'])
            pc = int(request.form['pc'])
            pcc = int(request.form['pcc'])
            ba = int(request.form['ba'])
            bgr = float(request.form['bgr'])
            bu = float(request.form['bu'])
            sc = float(request.form['sc'])
            pot = float(request.form['pot'])
            wc = int(request.form['wc'])
            htn = int(request.form['htn'])
            dm = int(request.form['dm'])
            cad = int(request.form['cad'])
            pe = int(request.form['pe'])
            ane = int(request.form['ane'])

            data = [age,bp,al,su,rbc,pc,pcc,ba,bgr,bu,sc,pot,wc,htn,dm,cad,pe,ane]
            data1 = np.array(data).reshape(1,-1)
            my_prediction = kidney_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)

        elif(len([float(x) for x in request.form.values()])==26):
            radius_mean = float(request.form['radius_mean'])
            texture_mean = float(request.form['texture_mean'])
            perimeter_mean = float(request.form['perimeter_mean'])
            area_mean = float(request.form['area_mean'])
            smoothness_mean = float(request.form['smoothness_mean'])
            compactness_mean = float(request.form['compactness_mean'])
            concavity_mean = float(request.form['concavity_mean'])
            concave_points_mean = float(request.form['concave points_mean'])
            symmetry_mean = float(request.form['symmetry_mean'])
            radius_se = float(request.form['radius_se'])
            perimeter_se = float(request.form['perimeter_se'])
            area_se = float(request.form['area_se'])
            compactness_se = float(request.form['compactness_se'])
            concavity_se = float(request.form['concavity_se'])
            concave_points_se = float(request.form['concave points_se'])
            fractal_dimension_se = float(request.form['fractal_dimension_se'])
            radius_worst = float(request.form['radius_worst'])
            texture_worst = float(request.form['texture_worst'])
            perimeter_worst = float(request.form['perimeter_worst'])
            area_worst = float(request.form['area_worst'])
            smoothness_worst = float(request.form['smoothness_worst'])
            compactness_worst = float(request.form['compactness_worst'])
            concavity_worst = float(request.form['concavity_worst'])
            concave_points_worst = float(request.form['concave points_worst'])
            symmetry_worst = float(request.form['symmetry_worst'])
            fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

            data = [radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,radius_se,perimeter_se,area_se,compactness_se,concavity_se,concave_points_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]
            data1 = np.array(data).reshape(1,-1)
            my_prediction  = cancer_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)


 
@app.route('/upload')
def upload():
    type = request.args.get('type')

    return render_template('upload.html',type=type)



from flask import Flask, render_template, request, redirect, flash
from keras.models import model_from_json
import os
import cv2

import tensorflow.keras.utils as image
import tensorflow as tf
import skimage
def load_model(weight,config):

    # Load the model architecture from JSON file
    json_file = open(config, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model weights from h5 file
    loaded_model.load_weights(weight)

    return loaded_model



def predict_img(img_path, loaded_model, categories):

    img = cv2.imread(img_path)
    img = cv2.resize(img, (90, 90))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    preds = loaded_model.predict(img,batch_size=None, verbose=0)
    labels = np.argmax(preds, axis=-1)    
    print("\nPREDICTION : "+categories[labels[0]])

    probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
    all_percentages =   probs * 100
    perc  = max(all_percentages[0])

    # return str(perc)[0:5] +" % probablity "+" of " + categories[labels[0]] 

    return categories[labels[0]]


################   OLD SKIN

# json_file = open('models/skin/custom_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# skin_cancer_model = model_from_json(loaded_model_json)
# skin_cancer_model.load_weights("models/skin/custom_best_model.h5")

# def skin_cancer(file_path):
    
#     categorie = ['Acne and Rosacea','Eczema','Melanoma Skin Cancer Nevi and Moles','Psoriasis pictures Lichen Planus and related diseases','Tinea Ringworm Candidiasis and other Fungal Infections']

#     img_a = np.asarray(Image.open(file_path).convert("RGB"))
#     image_resize = cv2.resize(img_a,(128,128))

#     img = image.load_img(file_path, target_size=(128, 128))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)

#     preds = skin_cancer_model.predict(x,batch_size=None, verbose=1)
#     labels = np.argmax(preds, axis=-1)    
#     print("\nPREDICTION : "+categorie[labels[0]])
#     probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
#     all_percentages =   probs * 100
#     perc  = max(all_percentages[0])
    
#     return str(perc)[0:5] +" % probablity of " + categorie[labels[0]]

######################

json_file = open(r"models\skin\skin_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
skin_cancer_model = model_from_json(loaded_model_json)
skin_cancer_model.load_weights(r"models\skin\skin_weight.h5")

def skin_cancer(file_path):
    
    categorie =['Skin - vasc',
                'Skin - nv',
                'Skin - mel',
                'Skin - df',
                'Skin - bkl',
                'Skin - bcc',
                'Skin - akiec']
 

    img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    img = skimage.transform.resize(img, (64, 64, 3))
    # img = image.load_img(photo, target_size=(64, 64))
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    preds = skin_cancer_model.predict(x,batch_size=None, verbose=1)
    labels = np.argmax(preds, axis=-1)
    print("\nPREDICTION : "+categorie[labels[0]])
    probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
    all_percentages =   probs * 100
    perc  = max(all_percentages[0])

    return categorie[labels[0]]

    # return str(perc)[0:5] +" % " + categorie[labels[0]]




def Decoder(pred):
    probability = pred
    if probability[0] > 0.5:
        return str(probability[0] * 100)[:4],'Suffering from Cancer'
    else:
        return str((1 - probability[0]) * 100)[:4], 'Not Suffering from Cancer'


def breast_cancer(path):    
    ######### DL BASED Models
    resnet = tf.keras.models.load_model('models/breast/resNet_32_3.h5')
    vgg_16 = tf.keras.models.load_model('models/breast/vgg_32_3.h5')
    vgg_19 = tf.keras.models.load_model('models/breast/vgg19_32_3.h5')
    #############################


    img = cv2.imread(path)
    img = cv2.resize(img,(50,50))
    inputs  = np.asarray([img]) / 255
    
    predicted_res = resnet.predict(inputs, verbose=1)
    pred_res = [0 if x <=0.5 else 1 for x in predicted_res]

    predicted_vgg_19 = vgg_19.predict(inputs, verbose=1)
    pred_vgg_19 = [0 if x <=0.5 else 1 for x in predicted_vgg_19]

    predicted_vgg_16 = vgg_16.predict(inputs, verbose=1)
    pred_vgg_16 = [0 if x <=0.5 else 1 for x in predicted_vgg_16]
    print('\n Res:',pred_res , '\n VGG 19:', pred_vgg_19,'\n VGG 16:', pred_vgg_16)  
    
    predicted_res = np.reshape(predicted_res, -1)
    predicted_vgg_19 = np.reshape(predicted_vgg_19, -1)
    predicted_vgg_16 = np.reshape(predicted_vgg_16, -1)

    ensemble_results = 0.33* (predicted_res*0.88 + predicted_vgg_19*0.74 + predicted_vgg_16*0.83)
    decoder_result = Decoder(ensemble_results)
    ensemble_results = [0 if x <=0.5 else 1 for x in ensemble_results]
    
    print('\n Ensemble_results : ', ensemble_results)

    return decoder_result[0]+" % Probability of Patient is "+decoder_result[1]





json_file = open(r"models\tumor\brain_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
brain_cancer_model = model_from_json(loaded_model_json)
brain_cancer_model.load_weights(r"models\tumor\brain_weight.h5")

def brain_tumor(file_path):
    
    categorie = ['Brain - glioma Brain Tumor',
                'Brain - pituitary Brain Tumor',
                'Brain - No  Brain Tumor',
                'Brain - meningioma Brain Tumor',
                'Brain - Alzheimer VeryMildDemented',
                'Brain - Alzheimer NonDemented',
                'Brain - Alzheimer ModerateDemented',
                'Brain - Alzheimer MildDemented']
                

    img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    img = skimage.transform.resize(img, (64, 64, 3))
    # img = image.load_img(photo, target_size=(64, 64))
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    preds = brain_cancer_model.predict(x,batch_size=None, verbose=1)
    labels = np.argmax(preds, axis=-1)
    print("\nPREDICTION : "+categorie[labels[0]])
    probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
    all_percentages =   probs * 100
    perc  = max(all_percentages[0])

    return categorie[labels[0]]

    # return str(perc)[0:5] +" % " + categorie[labels[0]]






@app.route('/prediction', methods=['POST', 'GET'])
def prediction():

    type = request.args.get('type')
    print(type)

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            if '.pdf' in file.filename:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename.split('.')[0] + '.pdf'))
                savedFilePath = './static/assets/images/' + file.filename.split('.')[0] + '.pdf'
 
            elif '.txt' in file.filename:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename.split('.')[0] + '.txt'))
                savedFilePath = './static/assets/images/' + file.filename.split('.')[0] + '.txt'
            else:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename.split('.')[0] + '.jpg'))
                savedFilePath = './static/assets/images/' + file.filename.split('.')[0] + '.jpg'


    algorithm_1_name = ""
    algorithm_1_result = ""

    algorithm_2_name = ""
    algorithm_2_result = ""

    algorithm_3_name = ""
    algorithm_3_result = ""

    early_symptoms = []
    precautions = []
    



    if 'lung' in type.lower():
        all_categories =['Normal','Adenocarcinomas Cell  (Stage 1)','Squamous Cell (Stage 2)','Large Cell (Stage 3 or 4 )']
        cnn_gru_pred =  predict_img('./static/assets/images/' + file.filename.split('.')[0] + '.jpg',load_model("models\lung\cnn_gru_best_model.h5","models\lung\cnn_gru_model.json"),all_categories)
        algorithm_1_name = "CNN + GRU  ALGORITHM (90%)"
        algorithm_1_result = cnn_gru_pred
        cnn_lstm_pred =  predict_img('./static/assets/images/' + file.filename.split('.')[0] + '.jpg',load_model("models\lung\cnn_lstm_best_model.h5","models\lung\cnn_lstm_model.json"),all_categories)
        algorithm_2_name = "CNN + LSTM   ALGORITHM (90%)"
        algorithm_2_result = cnn_lstm_pred
        early_symptoms.append("Persistent cough: A chronic cough that lasts for several weeks, especially if it worsens over time or produces blood.")
        early_symptoms.append("Shortness of breath: Difficulty breathing or feeling breathless, even with minimal physical activity.")
        early_symptoms.append("Chest pain: Dull, aching, or sharp pain in the chest, which may worsen with deep breathing, coughing, or laughing.")
        early_symptoms.append("Unexplained weight loss: A significant and unexplained loss of weight, often accompanied by loss of appetite.")
        early_symptoms.append("Fatigue: Feeling tired or weak even after getting enough rest and sleep.")
        early_symptoms.append("Hoarseness: A persistent or worsening hoarseness or change in voice.")
        early_symptoms.append("Frequent respiratory infections: Repeated occurrences of respiratory infections, such as bronchitis or pneumonia.")
        early_symptoms.append("Wheezing: A whistling or squeaky sound when breathing, particularly during exhalation.")
        early_symptoms.append("Coughing up blood: Coughing up blood or having blood-streaked mucus.")
        early_symptoms.append("Chest discomfort: A feeling of tightness, discomfort, or pressure in the chest.")

        precautions.append("Quit smoking: The most crucial step is to avoid tobacco smoke, including smoking and secondhand smoke.")
        precautions.append("Avoid exposure to carcinogens: Minimize exposure to harmful substances like asbestos, radon, and industrial chemicals, which are known to increase the risk of lung cancer.")
        precautions.append("Test your home for radon: Radon is a radioactive gas that can seep into homes through the soil. Testing and mitigating radon levels can help reduce the risk of lung cancer.")
        precautions.append("Protect against workplace hazards: Follow safety protocols and use protective equipment if you work in industries that involve exposure to carcinogens like asbestos, silica, diesel exhaust, or certain chemicals.")
        precautions.append("Maintain a healthy diet: Consume a balanced diet rich in fruits, vegetables, and whole grains. Some evidence suggests that a diet high in fruits and vegetables may help reduce the risk of lung cancer.")
        precautions.append("Exercise regularly: Engage in regular physical activity to maintain overall health and reduce the risk of various cancers, including lung cancer.")
        precautions.append("Limit alcohol consumption: Excessive alcohol consumption has been linked to an increased risk of developing lung cancer. Moderation is key.")
        precautions.append("Stay protected from air pollution: Minimize exposure to outdoor air pollution and try to improve indoor air quality by reducing exposure to smoke, fumes, and other pollutants.")
        precautions.append("Regular check-ups and screenings: If you have a high risk of developing lung cancer (e.g., due to smoking history or family history), discuss with your healthcare provider about appropriate screening tests such as low-dose CT scans.")
        precautions.append("Stay informed and seek medical advice: Educate yourself about the risk factors, symptoms, and early detection methods of lung cancer. Consult with a healthcare professional for personalized advice and guidance.")


    if 'pneumonia' in type.lower():
        all_categories =['NORMAL', 'PNEUMONIA']
        cnn_gru_pred =  predict_img('./static/assets/images/' + file.filename.split('.')[0] + '.jpg',load_model("models\pnemonia\cnn_gru_best_model.h5","models\pnemonia\cnn_gru_model.json"),all_categories)
        algorithm_1_name = "CNN + GRU   ALGORITHM (97%)"
        algorithm_1_result = cnn_gru_pred
        cnn_lstm_pred =  predict_img('./static/assets/images/' + file.filename.split('.')[0] + '.jpg',load_model("models\pnemonia\cnn_lstm_best_model.h5","models\pnemonia\cnn_lstm_model.json"),all_categories)
        algorithm_2_name = "CNN + LSTM   ALGORITHM (97%)"
        algorithm_2_result = cnn_lstm_pred

        early_symptoms.append("Cough: A persistent cough that may produce phlegm or mucus.")
        early_symptoms.append("Fever: A high temperature, often accompanied by sweating or chills.")
        early_symptoms.append("Shortness of breath: Difficulty breathing or rapid breathing, especially during physical activity.")
        early_symptoms.append("Chest pain: Sharp or stabbing pain in the chest, which may worsen with deep breaths or coughing.")
        early_symptoms.append("Fatigue: Feeling tired or exhausted, often accompanied by weakness or lack of energy.")
        early_symptoms.append("Sudden onset of symptoms: Pneumonia symptoms often appear suddenly, with a rapid progression of illness.")
        early_symptoms.append("Confusion or changes in mental awareness: Especially in older adults or individuals with weakened immune systems.")
        early_symptoms.append("Headache: A persistent or severe headache, often accompanied by body aches.")
        early_symptoms.append("Loss of appetite: A decrease in appetite, leading to reduced food intake.")
        early_symptoms.append("Blue lips or nails: In severe cases, a bluish tint to the lips or nails due to a lack of oxygen.")

        precautions.append("Vaccination: Stay up to date with recommended vaccinations, including the pneumococcal vaccine and influenza vaccine, as they can help prevent certain types of pneumonia.")
        precautions.append("Wash hands frequently: Practice good hand hygiene by washing your hands regularly with soap and water, especially before eating or touching your face.")
        precautions.append("Cover your mouth and nose: Use a tissue or your elbow to cover your mouth and nose when coughing or sneezing to prevent the spread of germs.")
        precautions.append("Avoid close contact: Minimize close contact with individuals who have respiratory infections, as pneumonia can be contagious.")
        precautions.append("Quit smoking: Smoking damages the lungs and weakens the immune system, making individuals more susceptible to respiratory infections, including pneumonia.")
        precautions.append("Maintain a healthy lifestyle: Eat a nutritious diet, exercise regularly, and get enough sleep to keep your immune system strong and reduce the risk of infections.")
        precautions.append("Manage chronic conditions: If you have underlying health conditions such as diabetes or heart disease, work with your healthcare provider to manage them effectively, as these conditions can increase the risk of pneumonia.")
        precautions.append("Avoid exposure to pollutants: Minimize exposure to air pollution, smoke, and harmful chemicals, as they can irritate the lungs and increase the risk of respiratory infections.")
        precautions.append("Seek medical attention: If you experience persistent symptoms or suspect you may have pneumonia, seek prompt medical attention for diagnosis and appropriate treatment.")

 


    if 'tuberculosis' in type.lower():
        all_categories =['Normal', 'Tuberculosis']
        cnn_gru_pred =  predict_img('./static/assets/images/' + file.filename.split('.')[0] + '.jpg',load_model("models\\tuberculosis\cnn_gru_best_model.h5","models\\tuberculosis\cnn_gru_model.json"),all_categories)
        algorithm_1_name = "CNN + GRU   ALGORITHM (98%)"
        algorithm_1_result = cnn_gru_pred
        cnn_lstm_pred =  predict_img('./static/assets/images/' + file.filename.split('.')[0] + '.jpg',load_model("models\\tuberculosis\cnn_lstm_best_model.h5","models\\tuberculosis\cnn_lstm_model.json"),all_categories)
        algorithm_2_name = "CNN + LSTM   ALGORITHM (99%)"
        algorithm_2_result = cnn_lstm_pred

        early_symptoms.append("Persistent cough: A cough that lasts for more than two weeks, often producing phlegm or blood.")
        early_symptoms.append("Fatigue: Feeling tired or weak, even without significant physical exertion.")
        early_symptoms.append("Fever: Mild to high-grade fever, typically occurring in the afternoon or evening.")
        early_symptoms.append("Night sweats: Excessive sweating during sleep, often soaking the sheets or nightclothes.")
        early_symptoms.append("Unintentional weight loss: Losing weight without trying, often accompanied by a decrease in appetite.")
        early_symptoms.append("Loss of appetite: A decreased desire to eat or a significant reduction in food intake.")
        early_symptoms.append("Chest pain: Pain or discomfort in the chest, often exacerbated by deep breathing or coughing.")
        early_symptoms.append("Shortness of breath: Breathlessness or difficulty breathing, particularly during physical activity.")
        early_symptoms.append("Coughing up blood: Coughing up blood or blood-streaked sputum.")
        early_symptoms.append("Swollen lymph nodes: Enlarged and tender lymph nodes, typically in the neck or underarms.")


        precautions.append("Vaccination: Ensure that you have received the Bacillus Calmette-Guérin (BCG) vaccine, which can help reduce the risk of severe forms of tuberculosis in children.")
        precautions.append("Maintain good hygiene: Practice proper respiratory hygiene by covering your mouth and nose with a tissue or your elbow when coughing or sneezing. Dispose of used tissues properly.")
        precautions.append("Avoid close contact: Limit close contact with individuals who have active TB until they have received appropriate treatment and are no longer contagious.")
        precautions.append("Promote ventilation: Ensure proper ventilation in living spaces, workplaces, and public transportation to reduce the risk of exposure to airborne bacteria.")
        precautions.append("Screening and early detection: If you are at high risk for TB (e.g., close contact with infected individuals), consider getting screened regularly for TB infection. Early detection allows for timely treatment and prevents the spread of the disease.")
        precautions.append("Follow prescribed treatment: If diagnosed with latent TB infection or active TB disease, adhere to the prescribed treatment regimen and complete the full course of medication to prevent the development of drug-resistant strains of TB.")
        precautions.append("Promote immune health: Maintain a healthy lifestyle, including a balanced diet, regular exercise, adequate sleep, and stress management, to support a strong immune system.")
        precautions.append("Protect against HIV infection: TB and HIV often occur together. Protect yourself against HIV infection by practicing safe sex, avoiding needle sharing, and getting tested regularly.")
        precautions.append("Seek medical attention: If you experience persistent symptoms or suspect you may have TB, seek prompt medical attention for proper diagnosis and appropriate treatment.")

        
    if 'skin' in type.lower():
        cnn_pred =  skin_cancer('./static/assets/images/' + file.filename.split('.')[0] + '.jpg')
        algorithm_3_name = "CNN   ALGORITHM (73%)"
        algorithm_3_result = cnn_pred
        
        early_symptoms.append("Changes in moles: Look out for moles that change in size, shape, color, or texture. This includes moles that become larger, develop an irregular border, exhibit multiple colors, or become raised or itchy.")
        early_symptoms.append("New growths: Pay attention to the appearance of new growths on your skin, such as a lump, bump, or sore that doesn't heal within a few weeks.")
        early_symptoms.append("Scaly or rough patches: Notice any scaly, rough, or crusty patches of skin that don't resolve or improve with moisturizers or over-the-counter creams.")
        early_symptoms.append("Bleeding or oozing: Be aware of any moles, sores, or growths that bleed, ooze, or won't stop bleeding with gentle pressure.")
        early_symptoms.append("Itching or tenderness: Experience persistent itching, tenderness, or pain in a specific area of your skin.")
        early_symptoms.append("Change in sensation: Observe any changes in the sensation of your skin, such as increased sensitivity or loss of sensation in a particular spot.")
        early_symptoms.append("Spots that look different: Take note of any spots on your skin that look different from the surrounding area, including spots that appear shiny, pearly, translucent, or waxy.")
        early_symptoms.append("Moles with irregular borders: Monitor moles that have irregular or notched borders rather than smooth and well-defined edges.")
        early_symptoms.append("Moles with uneven color: Pay attention to moles that have uneven colors or contain shades of black, brown, red, white, or blue.")
        early_symptoms.append("Moles larger than a pencil eraser: Be cautious of moles larger in diameter than a pencil eraser (approximately 6mm or more).")

        precautions = []

        precautions.append("Sun protection: Protect your skin from harmful UV radiation by wearing sunscreen with a high SPF, seeking shade during peak sun hours, and wearing protective clothing, including hats and sunglasses.")
        precautions.append("Avoid tanning beds: Minimize or avoid the use of tanning beds, as they emit harmful UV radiation that can increase the risk of skin cancer.")
        precautions.append("Perform regular skin self-exams: Check your skin regularly for any changes, including new moles, changes in existing moles, or other suspicious growths.")
        precautions.append("Seek shade: When spending time outdoors, seek shade, especially during the sun's peak hours (typically between 10 am and 4 pm).")
        precautions.append("Wear protective clothing: Cover your skin with clothing that provides adequate sun protection, such as long-sleeved shirts, long pants, and wide-brimmed hats.")
        precautions.append("Stay informed about your skin type: Understand your skin type and its sensitivity to sun exposure. Take appropriate precautions based on your skin's needs.")
        precautions.append("Avoid excessive sun exposure: Limit your time in direct sunlight, especially during peak hours. Take breaks indoors or seek shade to reduce your overall sun exposure.")
        precautions.append("Avoid indoor tanning: Completely avoid the use of indoor tanning beds, as they expose your skin to harmful UV radiation.")
        precautions.append("Regular professional skin examinations: Schedule regular skin examinations with a dermatologist or healthcare professional, particularly if you have a family history of skin cancer or are at higher risk.")
        precautions.append("Stay educated: Learn about the signs and symptoms of skin cancer and stay informed about the latest recommendations for prevention and early detection.")
        



    if 'brain' in type.lower():
        result =  brain_tumor('./static/assets/images/' + file.filename.split('.')[0] + '.jpg')
        algorithm_3_name = "CNN ALGORITHM (84%)"    
        algorithm_3_result = result


        #### Old Dataset

        # result =  predict_tumor('./static/assets/images/' + file.filename.split('.')[0] + '.jpg')
        # algorithm_3_name = "CNN   ALGORITHM (84%)"
        

        # if result[0][0] >0.45:
        #     prob=str(result[0][0]*100)[:4]
        #     algorithm_3_result = prob+ "% probablity of patient having Brain Tumor "
        # else:
        #     prob=str(100-(result[0][0]*100))[:4]
        #     algorithm_3_result = prob+ "% probablity is patient is Normal "

        ############################
            

        early_symptoms.append("Headaches: Persistent or worsening headaches, especially if they are new or different from your usual headaches. Headaches may be accompanied by nausea, vomiting, or changes in vision.")
        early_symptoms.append("Seizures: Seizures that occur without a known cause, particularly if you haven't had seizures before. Seizures can manifest as convulsions, muscle jerking, or unusual sensations.")
        early_symptoms.append("Cognitive changes: Changes in memory, concentration, or thinking abilities. You may experience confusion, difficulty finding words, or trouble remembering things.")
        early_symptoms.append("Personality or mood changes: Noticeable shifts in behavior, mood, or personality. This can include increased irritability, depression, anxiety, or unexplained emotional changes.")
        early_symptoms.append("Vision or hearing problems: Changes in vision, such as blurred vision, double vision, or loss of peripheral vision. You may also experience ringing in the ears or hearing loss.")
        early_symptoms.append("Weakness or numbness: Weakness or numbness in the arms or legs, often on one side of the body. You may experience difficulty with coordination or balance.")
        early_symptoms.append("Difficulty speaking: Problems with speech or language abilities, such as slurred speech, difficulty finding the right words, or trouble understanding others.")
        early_symptoms.append("Fatigue: Persistent fatigue or unexplained tiredness, even with adequate rest and sleep.")
        early_symptoms.append("Unexplained nausea or vomiting: Experiencing nausea or vomiting that is not related to a gastrointestinal issue or other apparent cause.")
        early_symptoms.append("Changes in sensation: Changes in sensations, such as tingling, tingling, or numbness in certain areas of the body.")

        precautions = []
        precautions.append("Regular check-ups: Attend regular health check-ups to monitor your overall health and discuss any concerns or symptoms you may have.")
        precautions.append("Protect your head: Take precautions to prevent head injuries. Wear appropriate headgear and safety equipment during sports and other activities that carry a risk of head trauma.")
        precautions.append("Reduce exposure to radiation: Limit unnecessary exposure to ionizing radiation, such as from excessive medical imaging scans or occupational radiation exposure.")
        precautions.append("Maintain a healthy lifestyle: Adopt a healthy lifestyle that includes regular exercise, a balanced diet, adequate sleep, stress management, and avoiding known risk factors such as excessive alcohol consumption or smoking.")
        precautions.append("Protect against harmful chemicals: Take precautions to minimize exposure to toxic chemicals, such as those found in certain workplace environments or environmental pollutants.")
        precautions.append("Brain health: Engage in activities that promote brain health, such as puzzles, reading, learning new skills, and maintaining social connections.")
        precautions.append("Monitor and manage existing conditions: If you have any pre-existing medical conditions, work closely with your healthcare provider to manage them effectively.")
        precautions.append("Genetic counseling: If you have a family history of brain tumors or genetic conditions associated with an increased risk of brain tumors, consider seeking genetic counseling to understand your risk and potential preventive measures.")
        precautions.append("Stay informed: Educate yourself about the signs and symptoms of brain tumors and seek medical attention if you experience persistent or concerning symptoms.")


        
    if 'breast' in type.lower():
        algorithm_3_result =  breast_cancer('./static/assets/images/' + file.filename.split('.')[0] + '.jpg')
        algorithm_3_name = "VGG19 + VGG16 + RESNET   ALGORITHM (88%)"
        
        early_symptoms.append("Lump or thickening: Noticeable lumps or thickening in the breast or underarm area. These lumps may feel different from the surrounding breast tissue.")
        early_symptoms.append("Changes in breast size or shape: Any changes in breast size, shape, or symmetry, including swelling, shrinkage, or asymmetry between the breasts.")
        early_symptoms.append("Nipple changes: Changes in the appearance of the nipple, such as inversion (when the nipple turns inward), redness, scaling, or discharge (other than breast milk).")
        early_symptoms.append("Skin changes: Dimpling, puckering, or changes in the texture or appearance of the breast skin, such as redness, scaliness, or thickening.")
        early_symptoms.append("Breast or nipple pain: Persistent pain or discomfort in the breast or nipple that is not related to the menstrual cycle or injury.")
        early_symptoms.append("Nipple retraction: A nipple that is pulled inward or appears flattened or indented, especially if it is a recent change.")
        early_symptoms.append("Swollen lymph nodes: Enlarged lymph nodes in the armpit or around the collarbone. These may be felt as lumps or swelling under the arm.")
        early_symptoms.append("Breast asymmetry: Noticeable changes in the size or shape of one breast compared to the other.")
        early_symptoms.append("Breast skin irritation: Unexplained redness, soreness, or rash on the breast or nipple area.")
        early_symptoms.append("Persistent breast changes: Any persistent changes in the breast that seem different from your normal breast tissue.")

        precautions.append("Breast self-exams: Perform regular breast self-examinations to familiarize yourself with your breasts and to detect any changes or abnormalities. Consult with your healthcare provider on the proper technique and frequency.")
        precautions.append("Clinical breast exams: Schedule regular clinical breast examinations by a healthcare professional as part of your routine check-ups.")
        precautions.append("Mammograms: Follow the recommended guidelines for mammograms based on your age and risk factors. Mammograms are important for early detection, particularly for women above the age of 40.")
        precautions.append("Know your family history: Be aware of your family history of breast cancer and discuss it with your healthcare provider. Individuals with a family history of breast cancer may require additional screening or genetic testing.")
        precautions.append("Maintain a healthy lifestyle: Adopt a healthy lifestyle that includes regular exercise, a balanced diet, maintaining a healthy weight, limiting alcohol consumption, and avoiding smoking.")
        precautions.append("Breastfeeding: If you have the opportunity, consider breastfeeding your babies, as it may have a protective effect against breast cancer.")
        precautions.append("Hormone therapy: If you are taking or considering hormone replacement therapy (HRT) or oral contraceptives, discuss the risks and benefits with your healthcare provider, as these may have an impact on breast cancer risk.")
        precautions.append("Breast awareness: Stay aware of the normal look and feel of your breasts and promptly report any changes or concerns to your healthcare provider.")
        precautions.append("Stay informed: Educate yourself about the risk factors, signs, and symptoms of breast cancer. Stay up-to-date on the latest guidelines and recommendations for breast cancer screening and prevention.")


    if 'analyzer' in type.lower():
        if '.pdf' in savedFilePath:
            outText = convertPdfText(savedFilePath)
            print(outText)

        if '.png' in savedFilePath or '.jpg' in savedFilePath or '.jpeg' in savedFilePath:
            outText = getTextFromImage(savedFilePath)

        if '.txt' in savedFilePath:
            outText = open(savedFilePath, 'r').read()

        # Construct the enhanced prompt with HTML formatting
        gpt_prompt = f"""
        I have a doctor's prescription or medical report and I would like to know more about the details mentioned in it. 
        Here is the text extracted from the document:
        '{outText}'

        Please provide detailed information in the following table format:

        <table style="width:100%; border-collapse: collapse; border: 1px solid black;">
            <tr style="background-color: #f2f2f2;">
                <th style="text-align: left; padding: 8px; border: 1px solid black;">Medicines</th>
                <td style="padding: 8px; border: 1px solid black;">Mentioned in the prescription, including their uses, dosages, and possible side effects.</td>
            </tr>
            <tr>
                <th style="text-align: left; padding: 8px; border: 1px solid black;">Health Conditions</th>
                <td style="padding: 8px; border: 1px solid black;">Any health conditions or symptoms noted in the document, and their implications according to you based on document.</td>
            </tr>
            <tr style="background-color: #f2f2f2;">
                <th style="text-align: left; padding: 8px; border: 1px solid black;">Recommendations</th>
                <td style="padding: 8px; border: 1px solid black;">Any recommendations or instructions according to you based on document.</td>
            </tr>
            <tr>
                <th style="text-align: left; padding: 8px; border: 1px solid black;">General Advice</th>
                <td style="padding: 8px; border: 1px solid black;">General advice related to the patient's health according to you based on document.</td>
            </tr>
            <tr style="background-color: #f2f2f2;">
                <th style="text-align: left; padding: 8px; border: 1px solid black;">Additional Information</th>
                <td style="padding: 8px; border: 1px solid black;">Any other relevant information that can help in understanding the document better.</td>
            </tr>
            <tr>
                <th style="text-align: left; padding: 8px; border: 1px solid black;">Test Report Summary</th>
                <td style="padding: 8px; border: 1px solid black;">If it is a test report, provide a summarization of all test values.</td>
            </tr>
            <tr style="background-color: #f2f2f2;">
                <th style="text-align: left; padding: 8px; border: 1px solid black;">Preventions</th>
                <td style="padding: 8px; border: 1px solid black;">Suggestions for preventions you should take according to you based on document.</td>
            </tr>
            <tr>
                <th style="text-align: left; padding: 8px; border: 1px solid black;">Precautions</th>
                <td style="padding: 8px; border: 1px solid black;">Suggestions for precautions the patient should take according to you based on document.</td>
            </tr>
            <tr style="background-color: #f2f2f2;">
                <th style="text-align: left; padding: 8px; border: 1px solid black;">Diet Recommendations</th>
                <td style="padding: 8px; border: 1px solid black;">Recommendations for what to eat according to you based on document.</td>
            </tr>
            <tr>
                <th style="text-align: left; padding: 8px; border: 1px solid black;">Avoid</th>
                <td style="padding: 8px; border: 1px solid black;">Recommendations for anything to avoid according to you based on document.</td>
            </tr>
            <tr style="background-color: #f2f2f2;">
                <th style="text-align: left; padding: 8px; border: 1px solid black;">Specific Parameters and Values</th>
                <td style="padding: 8px; border: 1px solid black;">
                    If the document contains specific parameters like blood cell counts, white cell counts, etc., please extract and display their values here.
                </td>
            </tr>
            <tr>
                <th style="text-align: left; padding: 8px; border: 1px solid black;">Diseases</th>
                <td style="padding: 8px; border: 1px solid black;">
                    If the document indicates any existing diseases or potential future symptoms based on the prescription or report, please provide details here.
                </td>
            </tr>
        </table>

        The doctor has explained everything, but I want to double-check and get more comprehensive information about the prescription or report from you. 
        Please provide a detailed explanation in the table format specified above with all block is mandatory and just give me table content as output only nothing else.
        """

        print("\n\n\n\n\gpt_prompt: " + gpt_prompt)

        result_html_output = ChatGPTApi.run(gpt_prompt)

        print("\n\n\n\n\nRES...", result_html_output)

        # Save the result to an HTML file
        html_output_path = r"templates/result.html"
        with open(html_output_path, 'w', encoding='utf-8') as file:
            file.write(result_html_output)
            
        return render_template('report_result.html',result=result_html_output)


    return render_template('results.html',  algorithm_1_name = algorithm_1_name,    algorithm_1_result = algorithm_1_result,
                           algorithm_2_name = algorithm_2_name,algorithm_2_result = algorithm_2_result,type=type,
                           image=file.filename.split('.')[0] + '.jpg',early_symptoms= early_symptoms, precautions=precautions,algorithm_3_name=algorithm_3_name,algorithm_3_result=algorithm_3_result
                           )




########## PDF to text
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

def convertPdfText(path):
    pdffileobj=open(path,'rb')
    output_string = StringIO()
    parser = PDFParser(pdffileobj)

    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)  

    output_string = output_string.getvalue()

    output_string = output_string.replace('\n','<br>')

    return output_string

############################################################


########################   Image to text
import os
try:
 from PIL import Image
except ImportError:
 import Image
import cv2


import easyocr
# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

 

def getTextFromImage(image_path_in_colab):

    # Read the image
    results = reader.readtext(image_path_in_colab)

    # # Load the image using OpenCV
    # image = cv2.imread(image_path_in_colab)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    str_results = ""

    # Loop over the results and draw bounding boxes
    for (bbox, text, prob) in results:
        # (top_left, top_right, bottom_right, bottom_left) = bbox
        # top_left = (int(top_left[0]), int(top_left[1]))
        # bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        # cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        # cv2.putText(image, text, (top_left[0], top_left[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        str_results += " " + text

    # im = cv2.imread(image_path_in_colab, cv2.IMREAD_COLOR)
    # plt.figure(figsize=(15,15))
    # plt.imshow(im)
    # plt.show()

    # image  = Image.open(image_path_in_colab)
    # extract = pytesseract.image_to_string(image)

    print('\n\n\n\n\n\n********************************************\m',
                '*********** Outpur Extracted Text **********\n',
        str_results,'\n********************************************')
    
    return str_results



 
#######################################


 
 

if __name__ == "__main__":
    app.run(debug=False)