import pickle
from flask import Flask, request, Response, render_template
app = Flask(__name__)

phish_model_ls = pickle.load(open('phishing.pkl', 'rb'))

urlError = {
    "Please enter url field"
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',  methods=['POST'])
def predict():

    X_predict = []

    url = request.form.get("EnterYourSite")
    print(url, "0000000000000000000000")
    if url:
        X_predict.append(str(url))
        y_Predict = ''.join(phish_model_ls.predict(X_predict))
        print(y_Predict)
        if y_Predict == 'bad':
            result = "This could potentially be a malicous Site and can cause harm to device"
        else:
            result = "This is not a malicous Site"

        return render_template('index.html', prediction_text = result)

    elif not url:
        return Response(
            response=urlError,
            status=400
        )

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
