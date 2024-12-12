from flask import Flask
from flask_cors import CORS
from server.routes.route import prediction

def create_server():
    server = Flask(__name__)
    CORS(server)

    server.register_blueprint(prediction)
    # Additional configurations can be added here 
    return server