from flask import Flask


def run_server(app: Flask):
    app.run(
        host=app.config.get('HOST', '127.0.0.1'),
        port=app.config.get('PORT', 5000),
        debug=app.config.get('ENV', 'production') != 'production'
    )
