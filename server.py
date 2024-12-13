from server import create_server

app = create_server()

if __name__ == "__main__":
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

    app.run(debug=True, use_reloader=False)  # Avoid double reloads in debug mode
