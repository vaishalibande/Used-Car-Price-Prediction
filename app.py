from flask import Flask,render_template,url_for,request,jsonify
from utils import get_used_car_price_prediction
import config
import numpy as np

app = Flask(__name__)

def get_model_names():
    # Replace this with your actual logic to fetch model names
    model_names = ['Forte', 'Silverado 1500', 'RAV4', 'Civic', 'Accord', 'GLC',
       '5 Series', 'Wrangler', 'Macan', 'Cascada', '3 Series',
       'Grand Cherokee', 'C-Class', 'Odyssey', '7 Series', 'K5', 'Kicks',
       'CR-V', 'Pacifica', 'Tiguan', 'Cayenne', 'Sentra', 'Camry',
       'Malibu', 'Volt', 'Rover Range Rover Sport', 'Highlander',
       'Optima', 'Charger', 'Corolla', 'Tacoma', '4Runner', 'Mustang',
       'Pilot', 'Rogue Sport', 'QX60', 'XC60', 'Explorer', 'Equinox',
       'GLE', 'S-Class', 'S90', 'X3', 'GLS', 'Bronco Sport',
       'Outlander Sport', 'XC90', 'RX', 'E-Class', 'Versa', 'Edge',
       '4 Series', 'Sonata', '300', 'Fusion', 'Mazda3', 'GLA', 'Blazer',
       'Altima', 'CX-9', 'S2000', 'Palisade', 'Transit Connect Van',
       'S60', 'Traverse', 'RDX', 'Quattroporte', 'Accent',
       'Rover Range Rover', 'Gladiator', 'Ranger', 'Tahoe', 'Camaro',
       'Soul', 'Outback', 'F-150', 'Rover Range Rover Velar', 'HR-V',
       'CLA', 'Fiesta', 'Venza', 'GLB', 'i3', 'Bronco', 'WRX', 'Jetta',
       'X5', 'IS', 'Transit Passenger Wagon', 'Q5', 'GranTurismo', 'Q3',
       'RC', 'Focus', 'Maxima', 'Elantra', 'S3', 'Challenger', 'Rogue',
       'Seltos', 'Compass', 'A-Class', 'allroad', 'Pathfinder', 'NX',
       'LS', 'F-PACE', 'Atlas', 'Canyon', 'Grand Caravan', 'X1', 'CC',
       'Kona', 'Frontier', 'Prius Prime', 'Forester', 'Encore', 'MDX',
       'XT5', 'Cherokee', 'Crosstrek', '1500 Classic', 'ES', 'CX-5',
       'Passat', 'Q50', 'A4', 'Romeo Giulia', 'C-HR',
       'Rover Range Rover Evoque', 'Tucson', 'Rover Discovery', 'Beetle',
       'Yaris iA', 'CX-30', 'Q60', 'XF', 'Sienna', 'Tundra', 'Passport',
       'Colorado', '1500', 'XT6', 'Escape', 'UX', 'Spark', 'Sportage',
       'Ascent', 'Transit Cargo Van', 'Enclave', 'Prius', 'Taos', 'A3',
       'Mazda6', 'Econoline Cargo Van', 'E-PACE', 'A5', '124 Spider',
       'Impreza', 'Corolla Cross', 'Nautilus', 'Corvette', 'Fit', 'M6',
       'M5', 'Sedona', 'Santa Fe', 'Titan', '200', 'Martin DB9', 'Legacy',
       'M-Class', 'Trailblazer', 'SLK', 'SL', 'CTS', 'Journey',
       'Escalade', 'Expedition', 'Metris Cargo Van', 'Clubman', 'A7',
       'ILX', 'CR-Z', '911', 'Yukon', 'Corolla Hatchback',
       'Grand Cherokee WK', 'A8', 'CLS', 'BRZ', 'Levante', 'GX', 'Armada',
       'Sierra 1500', 'Silverado 2500', 'RAV4 Prime', 'Romeo Stelvio',
       '6 Series', 'V60', 'GL', 'A6', 'Explorer Sport Trac', 'G80',
       'Suburban', 'EcoSport', '500', 'QX55', 'Countryman',
       'Super Duty F-250', 'Cruze', 'XT4', 'Acadia', 'Silverado 1500 LTD',
       'RS 5', 'Routan', 'Cayman', 'Town & Country', 'Prius v', 'Trax',
       'Q7', 'Genesis', 'Murano', 'Rover Discovery Sport', 'SLC',
       '2 Series', 'Crosstour', '718 Cayman', 'QX50', 'XC40', 'Golf GTI',
       'tC', 'CLK', 'Cruze Limited', 'Q70', 'Integra', 'Taurus', 'Quest',
       'Monte Carlo', 'Golf', 'Avalon', 'S80', 'Mirai',
       'NV200 Compact Cargo', 'Prius c', 'Express Cargo Van',
       'B9 Tribeca', 'Encore GX', 'Impala Limited', 'Outlander', 'NEXO',
       'Flying Spur', 'S7', 'GLK', 'GR Corolla', 'Hardtop', '2500', 'MKC',
       'Super Duty F-550 Chassis Cab', 'Rover LR4', 'CT', 'Telluride',
       'ProMaster City Cargo Van', 'Impala', 'ATS-V',
       'ProMaster Cargo Van', 'FR-S', 'TTS', 'CL', 'Solstice',
       'Golf Alltrack', 'CT4', 'Santa Fe Sport', 'Durango', 'Sorento',
       'Ghibli', 'G70', 'R-Class', 'Terrain', 'Z3', 'CX-3', 'Aura', 'TLX',
       'Rio', 'Patriot', 'Boxster', 'MKX', 'CX-90', 'xB', 'XJ',
       'Thunderbird', 'Ram 1500', 'Super Duty F-350', 'Atlas Cross Sport']
    return model_names

def get_make_options():
    # Replace this with your actual logic to fetch make options
    make_options = ['Kia', 'Chevrolet', 'Toyota', 'Honda', 'Mercedes-Benz', 'BMW',
       'Jeep', 'Porsche', 'Buick', 'Nissan', 'Chrysler', 'Volkswagen',
       'Land', 'Dodge', 'Ford', 'INFINITI', 'Volvo', 'Mitsubishi',
       'Lexus', 'Hyundai', 'Mazda', 'Acura', 'Maserati', 'Subaru', 'Audi',
       'Jaguar', 'GMC', 'Cadillac', 'Ram', 'Alfa', 'FIAT', 'Lincoln',
       'Aston', 'MINI', 'Genesis', 'Scion', 'Bentley', 'Pontiac',
       'Saturn']
    return make_options

def get_exterior_color_options():
    # Replace this with your actual logic to fetch exterior color options
    exterior_color_options = ['Gray', 'White', 'Silver', 'Blue', 'Black', 'Red', 'Orange',
       'Green', 'Tan', 'Unknown', 'Gold', 'Brown', 'Yellow', 'Purple']
    return exterior_color_options

def get_interior_color_options():
    # Replace this with your actual logic to fetch interior color options
    interior_color_options = ['Black', 'Unknown', 'Beige', 'Brown', 'Gray', 'Red', 'White',
       'Blue', 'Yellow']
    return interior_color_options


# Define the home route to render the index.html template
@app.route('/')
def home():
    make_options = get_make_options()
    exterior_color_options = get_exterior_color_options()
    interior_color_options = get_interior_color_options()
    model_names = get_model_names()
    return render_template('index.html', model_names=model_names,make_options=make_options,
                           exterior_color_options=exterior_color_options,
                           interior_color_options=interior_color_options)
    

import pandas as pd

# ...

# Endpoint to get used car price prediction
@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        # Get data from request
        data = request.json

        # Ensure all required columns are present
        required_cols = ['make', 'model', 'year', 'miles', 'exterior-color', 'interior-color',
                         'accidents-reported', 'num-of-owners']
        if not all(col in data for col in required_cols):
            return jsonify({"error": "Missing required columns"}), 400

        # Convert data to a pandas DataFrame for prediction
        df = pd.DataFrame([data])

        # Get used car price prediction
        used_car_price_prediction = get_used_car_price_prediction(df)

        # Return a JSON response with the predicted price
        return jsonify({"used_car_price_prediction": used_car_price_prediction})

    except Exception as e:
        # Log the exception details
        print(f"Error predicting price: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

        

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=config.PORT_NUMBER,debug=False)

