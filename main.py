import os
import sys

# Unset problematic SSL environment variables if they exist
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

# Now import and run the application
from app.interface import iface

if __name__ == '__main__':
    print("Python executable:", sys.executable)
    iface.launch(server_name="0.0.0.0", server_port=8080)