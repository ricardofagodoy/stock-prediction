from flask import Flask
from flask_restful import Api, Resource, reqparse

# Static index page
app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def root():
    return app.send_static_file('index.html')

# Api for ticks
api = Api(app)

ticks = [
    ['04:00', 8, 10, 12, 19],
    ['05:00', 8, 13, 18, 19],
    ['06:00', 8, 21, 12, 39],
    ['07:00', 8, 10, 12, 19],
    ['08:00', 8, 10, 12, 19],
    ['09:00', 8, 10, 12, 19],
    ['10:00', 8, 10, 12, 19],
    ['11:00', 8, 10, 12, 19],
    ['12:00', 8, 10, 12, 19],
    ['13:00', 8, 10, 12, 19],
    ['14:00', 8, 10, 12, 19],
    ['15:00', 15, 15, 22, 23]
]

class Tick(Resource):
    def get(self):
        return ticks, 200

api.add_resource(Tick, '/tick')
app.run(debug=True)