import numpy as np
import json
from flask import template_rendered, Flask, request, jsonify, render_template
import requests
import socket

app = Flask(__name__, template_folder='ui' , static_folder='ui/static')

with open('/home/isam/ai_controler/main/dataset.json', 'r') as file:
    data = json.load(file)

inputs = []
outputs = []

for sample in data:
    input_data = [sample["voltage"], sample["current"], sample["temperature"], sample["humidity"], sample["wind_speed"]]
    output_data = sample["power_source"]

    if output_data == "Battery":
        output_data = [1, 0, 0]
    elif output_data == "Solar Panel":
        output_data = [0, 1, 0]
    elif output_data == "Grid":
        output_data = [0, 0, 1]

    inputs.append(input_data)
    outputs.append(output_data)

IP = np.array(inputs)
OP = np.array(outputs)
IP = (IP - np.min(IP, axis=0)) / (np.max(IP, axis=0) - np.min(IP, axis=0))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Dsigmoid(x):
    return x * (1 - x)

def forward_pass(IP, weights, biases):
    activations = [IP]
    layer_input = IP
    for w, b in zip(weights[:-1], biases[:-1]):
        layer_input = sigmoid(np.dot(layer_input, w) + b)
        activations.append(layer_input)
    output_layer_input = np.dot(layer_input, weights[-1]) + biases[-1]
    final_output = sigmoid(output_layer_input)
    activations.append(final_output)
    return activations

def backpropagate(IP, OP, activations, weights, biases, learning_rate=0.1):
    output_error = OP - activations[-1]
    delta = output_error * Dsigmoid(activations[-1])
    deltas = [delta]
    for i in range(len(weights) - 2, -1, -1):
        delta = deltas[-1].dot(weights[i+1].T) * Dsigmoid(activations[i+1])
        deltas.append(delta)
    deltas.reverse()

    for i in range(len(weights)):
        weights[i] += learning_rate * activations[i].T.dot(deltas[i])
        biases[i] += learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    return weights, biases, np.mean(np.abs(output_error))

def train_network(IP, OP, hidden_layers=[5, 5], epochs=50000, learning_rate=0.1):
    np.random.seed(1)
    layer_sizes = [IP.shape[1]] + hidden_layers + [OP.shape[1]]
    weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]
    biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]

    for epoch in range(epochs):
        activations = forward_pass(IP, weights, biases)
        weights, biases, error = backpropagate(IP, OP, activations, weights, biases, learning_rate)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Error: {error * 100}")
    return weights, biases

def predict(IP, weights, biases):
    activations = forward_pass(IP, weights, biases)
    return np.round(activations[-1])

def get_weather():
    url = "https://api.open-meteo.com/v1/forecast?latitude=32.0&longitude=44.0&current=temperature_2m,humidity_2m,wind_speed_10m"
    try:
        response = requests.get(url)
        data = response.json()['current']
        return {
            "temperature": data.get("temperature_2m", 25),
            "humidity": data.get("humidity_2m", 50),
            "wind_speed": data.get("wind_speed_10m", 3)
        }
    except:
        return {"temperature": 25, "humidity": 50, "wind_speed": 3}

hidden_layers = [10, 8, 6, 4]
weights, biases = train_network(IP, OP, hidden_layers)

@app.route('/')
def home():
    return render_template('ui.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    print("Received Data from ESP32:", data)

    weather = get_weather()
    input_data = np.array([
        data['voltage'],
        data['current'],
        weather['temperature'],
        weather['humidity'],
        weather['wind_speed']
    ])
    input_data = (input_data - np.min(IP, axis=0)) / (np.max(IP, axis=0) - np.min(IP, axis=0))

    prediction = predict(np.array([input_data]), weights, biases)
    index = np.argmax(prediction[0])
    source = ['Battery', 'Solar Panel', 'Grid'][index]

    print("Prediction:", source)
    return jsonify({'prediction': source})

if __name__ == '__main__':
    ip = socket.gethostbyname(socket.gethostname())
    print(f"Visit from other devices at: http://{ip}:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)