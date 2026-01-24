import subprocess
from src.settings import settings
from urllib.parse import urlparse


def serve():
    """
    Start the MLflow server using settings from the environment.
    """
    # Parse the tracking URI to find the port
    uri = settings.MLFLOW_TRACKING_URI
    parsed_uri = urlparse(uri)

    # Default port if not found in URI
    port = parsed_uri.port if parsed_uri.port else 5050
    host = parsed_uri.hostname if parsed_uri.hostname else "127.0.0.1"

    # We use the tracking URI itself as the backend-store-uri if it's sqlite
    # If it's a file URI or directory, we handle it accordingly.
    backend_uri = uri

    print(f"Starting MLflow server on {host}:{port}")
    print(f"Backend store URI: {backend_uri}")

    cmd = [
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        backend_uri,
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping MLflow server...")
    except Exception as e:
        print(f"Error starting MLflow server: {e}")


if __name__ == "__main__":
    serve()
