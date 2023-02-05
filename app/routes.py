from app import app


@app.route('/train')
@app.route('/predict')
def index():
    return "Hello, World!"
