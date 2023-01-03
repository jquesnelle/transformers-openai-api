from flask import Flask
from waitress import serve


def run_server(app: Flask):
    if app.config['ENV'] == 'production':
        serve(app, host=app.config.get('HOST', '0.0.0.0'), port=app.config.get(
            'PORT', 5000), url_scheme=app.config.get('URL_SCHEME', 'http'))
    else:
        app.run(host=app.config.get('HOST', '127.0.0.1'))
