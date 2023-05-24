from flask import Flask, request, jsonify, render_template
import pickle
from app.defs import *
import joblib
from datetime import datetime, time
app = Flask(__name__)



@app.route('/')
def home():
    # return "HELLO WORLD!!"
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the machine learning model

    model = joblib.load('rf_clf.pkl')

    # Get input data from the form
    origin = request.form.get('origin')
    destination = request.form.get('destination')
    airline = request.form.get('airline')
    flight_date = request.form.get('flight-date')
    departure_time = request.form.get('departure-time')
    arrival_time = request.form.get('arrival-time')
    print(">>>>>>>>>>>origin", origin)
    print(">>>>>>>>>>>destination", destination)
    print(">>>>>>>>>>>airline", airline)
    print(">>>>>>>>>>>flight_date", flight_date)
    print(">>>>>>>>>>>departure_time", departure_time)
    print(">>>>>>>>>>>arrival_time", arrival_time)
    OP_CARRIER_mapping = {'9E': 0, 'AA': 1, 'AS': 2, 'B6': 3, 'DL': 4, 'F9': 5, 'G4': 6, 'HA': 7, 'MQ': 8, 'NK': 9, 'OH': 10, 'OO': 11, 'QX': 12, 'UA': 13, 'WN': 14, 'YV': 15, 'YX': 16}
    ORIGIN_mapping = {'ABE': 0, 'ABI': 1, 'ABQ': 2, 'ABR': 3, 'ABY': 4, 'ACK': 5, 'ACT': 6, 'ACV': 7, 'ACY': 8, 'ADK': 9, 'ADQ': 10, 'AEX': 11, 'AGS': 12, 'AKN': 13, 'ALB': 14, 'ALO': 15, 'ALS': 16, 'ALW': 17, 'AMA': 18, 'ANC': 19, 'APN': 20, 'ASE': 21, 'ATL': 22, 'ATW': 23, 'ATY': 24, 'AUS': 25, 'AVL': 26, 'AVP': 27, 'AZA': 28, 'AZO': 29, 'BDL': 30, 'BET': 31, 'BFF': 32, 'BFL': 33, 'BGM': 34, 'BGR': 35, 'BHM': 36, 'BIL': 37, 'BIS': 38, 'BJI': 39, 'BKG': 40, 'BLI': 41, 'BLV': 42, 'BMI': 43, 'BNA': 44, 'BOI': 45, 'BOS': 46, 'BPT': 47, 'BQK': 48, 'BQN': 49, 'BRD': 50, 'BRO': 51, 'BRW': 52, 'BTM': 53, 'BTR': 54, 'BTV': 55, 'BUF': 56, 'BUR': 57, 'BWI': 58, 'BZN': 59, 'CAE': 60, 'CAK': 61, 'CDB': 62, 'CDC': 63, 'CDV': 64, 'CGI': 65, 'CHA': 66, 'CHO': 67, 'CHS': 68, 'CID': 69, 'CIU': 70, 'CKB': 71, 'CLE': 72, 'CLL': 73, 'CLT': 74, 'CMH': 75, 'CMI': 76, 'CMX': 77, 'CNY': 78, 'COD': 79, 'COS': 80, 'COU': 81, 'CPR': 82, 'CRP': 83, 'CRW': 84, 'CSG': 85, 'CVG': 86, 'CWA': 87, 'DAB': 88, 'DAL': 89, 'DAY': 90, 'DBQ': 91, 'DCA': 92, 'DDC': 93, 'DEC': 94, 'DEN': 95, 'DFW': 96, 'DHN': 97, 'DLG': 98, 'DLH': 99, 'DRO': 100, 'DRT': 101, 'DSM': 102, 'DTW': 103, 'DVL': 104, 'EAR': 105, 'EAT': 106, 'EAU': 107, 'ECP': 108, 'EGE': 109, 'EKO': 110, 'ELM': 111, 'ELP': 112, 'ERI': 113, 'ESC': 114, 'EUG': 115, 'EVV': 116, 'EWN': 117, 'EWR': 118, 'EYW': 119, 'FAI': 120, 'FAR': 121, 'FAT': 122, 'FAY': 123, 'FCA': 124, 'FLG': 125, 'FLL': 126, 'FNT': 127, 'FOD': 128, 'FSD': 129, 'FSM': 130, 'FWA': 131, 'GCC': 132, 'GCK': 133, 'GEG': 134, 'GFK': 135, 'GGG': 136, 'GJT': 137, 'GNV': 138, 'GPT': 139, 'GRB': 140, 'GRI': 141, 'GRK': 142, 'GRR': 143, 'GSO': 144, 'GSP': 145, 'GST': 146, 'GTF': 147, 'GTR': 148, 'GUC': 149, 'GUM': 150, 'HDN': 151, 'HGR': 152, 'HHH': 153, 'HIB': 154, 'HLN': 155, 'HNL': 156, 'HOB': 157, 'HOU': 158, 'HPN': 159, 'HRL': 160, 'HSV': 161, 'HTS': 162, 'HVN': 163, 'HYA': 164, 'HYS': 165, 'IAD': 166, 'IAG': 167, 'IAH': 168, 'ICT': 169, 'IDA': 170, 'ILG': 171, 'ILM': 172, 'IMT': 173, 'IND': 174, 'INL': 175, 'ISP': 176, 'ITH': 177, 'ITO': 178, 'JAC': 179, 'JAN': 180, 'JAX': 181, 'JFK': 182, 'JLN': 183, 'JMS': 184, 'JNU': 185, 'JST': 186, 'KOA': 187, 'KTN': 188, 'LAN': 189, 'LAR': 190, 'LAS': 191, 'LAW': 192, 'LAX': 193, 'LBB': 194, 'LBE': 195, 'LBF': 196, 'LBL': 197, 'LCH': 198, 'LCK': 199, 'LEX': 200, 'LFT': 201, 'LGA': 202, 'LGB': 203, 'LIH': 204, 'LIT': 205, 'LNK': 206, 'LRD': 207, 'LSE': 208, 'LWB': 209, 'LWS': 210, 'MAF': 211, 'MBS': 212, 'MCI': 213, 'MCO': 214, 'MCW': 215, 'MDT': 216, 'MDW': 217, 'MEI': 218, 'MEM': 219, 'MFE': 220, 'MFR': 221, 'MGM': 222, 'MHK': 223, 'MHT': 224, 'MIA': 225, 'MKE': 226, 'MKG': 227, 'MLB': 228, 'MLI': 229, 'MLU': 230, 'MOB': 231, 'MOT': 232, 'MQT': 233, 'MRY': 234, 'MSN': 235, 'MSO': 236, 'MSP': 237, 'MSY': 238, 'MTJ': 239, 'MVY': 240, 'MYR': 241, 'OAJ': 242, 'OAK': 243, 'OGD': 244, 'OGG': 245, 'OGS': 246, 'OKC': 247, 'OMA': 248, 'OME': 249, 'ONT': 250, 'ORD': 251, 'ORF': 252, 'OTH': 253, 'OTZ': 254, 'OWB': 255, 'PAE': 256, 'PAH': 257, 'PBG': 258, 'PBI': 259, 'PDX': 260, 'PGD': 261, 'PHF': 262, 'PHL': 263, 'PHX': 264, 'PIA': 265, 'PIB': 266, 'PIE': 267, 'PIH': 268, 'PIR': 269, 'PIT': 270, 'PLN': 271, 'PNS': 272, 'PRC': 273, 'PSC': 274, 'PSE': 275, 'PSG': 276, 'PSM': 277, 'PSP': 278, 'PUB': 279, 'PUW': 280, 'PVD': 281, 'PVU': 282, 'PWM': 283, 'RAP': 284, 'RDD': 285, 'RDM': 286, 'RDU': 287, 'RFD': 288, 'RHI': 289, 'RIC': 290, 'RIW': 291, 'RKS': 292, 'RNO': 293, 'ROA': 294, 'ROC': 295, 'ROW': 296, 'RST': 297, 'RSW': 298, 'SAF': 299, 'SAN': 300, 'SAT': 301, 'SAV': 302, 'SBA': 303, 'SBN': 304, 'SBP': 305, 'SCC': 306, 'SCE': 307, 'SCK': 308, 'SDF': 309, 'SEA': 310, 'SFB': 311, 'SFO': 312, 'SGF': 313, 'SGU': 314, 'SHD': 315, 'SHR': 316, 'SHV': 317, 'SIT': 318, 'SJC': 319, 'SJT': 320, 'SJU': 321, 'SLC': 322, 'SLN': 323, 'SMF': 324, 'SMX': 325, 'SNA': 326, 'SPI': 327, 'SPN': 328, 'SPS': 329, 'SRQ': 330, 'STC': 331, 'STL': 332, 'STS': 333, 'STT': 334, 'STX': 335, 'SUN': 336, 'SUX': 337, 'SWF': 338, 'SWO': 339, 'SYR': 340, 'TLH': 341, 'TOL': 342, 'TPA': 343, 'TRI': 344, 'TTN': 345, 'TUL': 346, 'TUS': 347, 'TVC': 348, 'TWF': 349, 'TXK': 350, 'TYR': 351, 'TYS': 352, 'USA': 353, 'VCT': 354, 'VEL': 355, 'VLD': 356, 'VPS': 357, 'WRG': 358, 'WYS': 359, 'XNA': 360, 'XWA': 361, 'YAK': 362, 'YKM': 363, 'YUM': 364}
    DEST_mapping = {'ABE': 0, 'ABI': 1, 'ABQ': 2, 'ABR': 3, 'ABY': 4, 'ACK': 5, 'ACT': 6, 'ACV': 7, 'ACY': 8, 'ADK': 9, 'ADQ': 10, 'AEX': 11, 'AGS': 12, 'AKN': 13, 'ALB': 14, 'ALO': 15, 'ALS': 16, 'ALW': 17, 'AMA': 18, 'ANC': 19, 'APN': 20, 'ASE': 21, 'ATL': 22, 'ATW': 23, 'ATY': 24, 'AUS': 25, 'AVL': 26, 'AVP': 27, 'AZA': 28, 'AZO': 29, 'BDL': 30, 'BET': 31, 'BFF': 32, 'BFL': 33, 'BGM': 34, 'BGR': 35, 'BHM': 36, 'BIL': 37, 'BIS': 38, 'BJI': 39, 'BKG': 40, 'BLI': 41, 'BLV': 42, 'BMI': 43, 'BNA': 44, 'BOI': 45, 'BOS': 46, 'BPT': 47, 'BQK': 48, 'BQN': 49, 'BRD': 50, 'BRO': 51, 'BRW': 52, 'BTM': 53, 'BTR': 54, 'BTV': 55, 'BUF': 56, 'BUR': 57, 'BWI': 58, 'BZN': 59, 'CAE': 60, 'CAK': 61, 'CDB': 62, 'CDC': 63, 'CDV': 64, 'CGI': 65, 'CHA': 66, 'CHO': 67, 'CHS': 68, 'CID': 69, 'CIU': 70, 'CKB': 71, 'CLE': 72, 'CLL': 73, 'CLT': 74, 'CMH': 75, 'CMI': 76, 'CMX': 77, 'CNY': 78, 'COD': 79, 'COS': 80, 'COU': 81, 'CPR': 82, 'CRP': 83, 'CRW': 84, 'CSG': 85, 'CVG': 86, 'CWA': 87, 'DAB': 88, 'DAL': 89, 'DAY': 90, 'DBQ': 91, 'DCA': 92, 'DDC': 93, 'DEC': 94, 'DEN': 95, 'DFW': 96, 'DHN': 97, 'DLG': 98, 'DLH': 99, 'DRO': 100, 'DRT': 101, 'DSM': 102, 'DTW': 103, 'DVL': 104, 'EAR': 105, 'EAT': 106, 'EAU': 107, 'ECP': 108, 'EGE': 109, 'EKO': 110, 'ELM': 111, 'ELP': 112, 'ERI': 113, 'ESC': 114, 'EUG': 115, 'EVV': 116, 'EWN': 117, 'EWR': 118, 'EYW': 119, 'FAI': 120, 'FAR': 121, 'FAT': 122, 'FAY': 123, 'FCA': 124, 'FLG': 125, 'FLL': 126, 'FNT': 127, 'FOD': 128, 'FSD': 129, 'FSM': 130, 'FWA': 131, 'GCC': 132, 'GCK': 133, 'GEG': 134, 'GFK': 135, 'GGG': 136, 'GJT': 137, 'GNV': 138, 'GPT': 139, 'GRB': 140, 'GRI': 141, 'GRK': 142, 'GRR': 143, 'GSO': 144, 'GSP': 145, 'GST': 146, 'GTF': 147, 'GTR': 148, 'GUC': 149, 'GUM': 150, 'HDN': 151, 'HGR': 152, 'HHH': 153, 'HIB': 154, 'HLN': 155, 'HNL': 156, 'HOB': 157, 'HOU': 158, 'HPN': 159, 'HRL': 160, 'HSV': 161, 'HTS': 162, 'HVN': 163, 'HYA': 164, 'HYS': 165, 'IAD': 166, 'IAG': 167, 'IAH': 168, 'ICT': 169, 'IDA': 170, 'ILG': 171, 'ILM': 172, 'IMT': 173, 'IND': 174, 'INL': 175, 'ISP': 176, 'ITH': 177, 'ITO': 178, 'JAC': 179, 'JAN': 180, 'JAX': 181, 'JFK': 182, 'JLN': 183, 'JMS': 184, 'JNU': 185, 'JST': 186, 'KOA': 187, 'KTN': 188, 'LAN': 189, 'LAR': 190, 'LAS': 191, 'LAW': 192, 'LAX': 193, 'LBB': 194, 'LBE': 195, 'LBF': 196, 'LBL': 197, 'LCH': 198, 'LCK': 199, 'LEX': 200, 'LFT': 201, 'LGA': 202, 'LGB': 203, 'LIH': 204, 'LIT': 205, 'LNK': 206, 'LRD': 207, 'LSE': 208, 'LWB': 209, 'LWS': 210, 'MAF': 211, 'MBS': 212, 'MCI': 213, 'MCO': 214, 'MCW': 215, 'MDT': 216, 'MDW': 217, 'MEI': 218, 'MEM': 219, 'MFE': 220, 'MFR': 221, 'MGM': 222, 'MHK': 223, 'MHT': 224, 'MIA': 225, 'MKE': 226, 'MKG': 227, 'MLB': 228, 'MLI': 229, 'MLU': 230, 'MOB': 231, 'MOT': 232, 'MQT': 233, 'MRY': 234, 'MSN': 235, 'MSO': 236, 'MSP': 237, 'MSY': 238, 'MTJ': 239, 'MVY': 240, 'MYR': 241, 'OAJ': 242, 'OAK': 243, 'OGD': 244, 'OGG': 245, 'OGS': 246, 'OKC': 247, 'OMA': 248, 'OME': 249, 'ONT': 250, 'ORD': 251, 'ORF': 252, 'OTH': 253, 'OTZ': 254, 'OWB': 255, 'PAE': 256, 'PAH': 257, 'PBG': 258, 'PBI': 259, 'PDX': 260, 'PGD': 261, 'PHF': 262, 'PHL': 263, 'PHX': 264, 'PIA': 265, 'PIB': 266, 'PIE': 267, 'PIH': 268, 'PIR': 269, 'PIT': 270, 'PLN': 271, 'PNS': 272, 'PRC': 273, 'PSC': 274, 'PSE': 275, 'PSG': 276, 'PSM': 277, 'PSP': 278, 'PUB': 279, 'PUW': 280, 'PVD': 281, 'PVU': 282, 'PWM': 283, 'RAP': 284, 'RDD': 285, 'RDM': 286, 'RDU': 287, 'RFD': 288, 'RHI': 289, 'RIC': 290, 'RIW': 291, 'RKS': 292, 'RNO': 293, 'ROA': 294, 'ROC': 295, 'ROW': 296, 'RST': 297, 'RSW': 298, 'SAF': 299, 'SAN': 300, 'SAT': 301, 'SAV': 302, 'SBA': 303, 'SBN': 304, 'SBP': 305, 'SCC': 306, 'SCE': 307, 'SCK': 308, 'SDF': 309, 'SEA': 310, 'SFB': 311, 'SFO': 312, 'SGF': 313, 'SGU': 314, 'SHD': 315, 'SHR': 316, 'SHV': 317, 'SIT': 318, 'SJC': 319, 'SJT': 320, 'SJU': 321, 'SLC': 322, 'SLN': 323, 'SMF': 324, 'SMX': 325, 'SNA': 326, 'SPI': 327, 'SPN': 328, 'SPS': 329, 'SRQ': 330, 'STC': 331, 'STL': 332, 'STS': 333, 'STT': 334, 'STX': 335, 'SUN': 336, 'SUX': 337, 'SWF': 338, 'SWO': 339, 'SYR': 340, 'TLH': 341, 'TOL': 342, 'TPA': 343, 'TRI': 344, 'TTN': 345, 'TUL': 346, 'TUS': 347, 'TVC': 348, 'TWF': 349, 'TXK': 350, 'TYR': 351, 'TYS': 352, 'USA': 353, 'VCT': 354, 'VEL': 355, 'VLD': 356, 'VPS': 357, 'WRG': 358, 'WYS': 359, 'XNA': 360, 'XWA': 361, 'YAK': 362, 'YKM': 363, 'YUM': 364}   
    flight_date_parsed = datetime.strptime(flight_date, '%m/%d/%Y').date()
    DAY_OF_WEEK = flight_date_parsed.weekday()
    month = flight_date_parsed.month
    DAY_OF_MONTH = flight_date_parsed.day
    car_popularity = label_carrier_train(airline)
    OP_CARRIER = OP_CARRIER_mapping[airline]
    ORIGIN = ORIGIN_mapping[origin]
    DEST = DEST_mapping[destination]
    departure_time_object = time.fromisoformat(departure_time)
    DEP_TIME = 0
    if len(str(departure_time_object.minute))==2:
        DEP_TIME = int(str(departure_time_object.hour)+str(departure_time_object.minute))
    elif len(str(departure_time_object.minute))==1:
        DEP_TIME = int(str(departure_time_object.hour)+'0'+str(departure_time_object.minute))
    else:
        DEP_TIME = int(str(departure_time_object.hour)+'00')

    ARR_TIME = 0
    arrival_time_object = time.fromisoformat(arrival_time)
    if len(str(arrival_time_object.minute))==2:
        ARR_TIME = int(str(arrival_time_object.hour)+str(arrival_time_object.minute))
    elif len(str(arrival_time_object.minute))==1:
        ARR_TIME = int(str(arrival_time_object.hour)+'0'+str(arrival_time_object.minute))
    else:
        ARR_TIME = int(str(arrival_time_object.hour)+'00')
    is_weekend = 0
    if DAY_OF_WEEK==5 or DAY_OF_WEEK==6:
        is_weekend = 1
    # Convert input data to a feature vector
    features = [month,DAY_OF_MONTH,OP_CARRIER,ORIGIN,DEST,DEP_TIME,ARR_TIME,DAY_OF_WEEK,car_popularity, is_weekend]
    print(">>>>>>>>>>>features", features)
    # Make a prediction using the machine learning model
    prediction = model.predict([features]).tolist()[0]
    print(">>>>>>>>>>prediction", prediction)
    print(">>>>>>>>>>type(prediction)", type(prediction))

    # Return the prediction as JSON
    return render_template('result.html', output=prediction)
    # return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)