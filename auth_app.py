from flask import Flask, request, abort
import socket
import re

app = Flask(__name__)

ALLOWED_SUFFIX_PATTERN = r'^.*\.amazon\.com$'

@app.route('/')
def index():
    dns_suffix = socket.getfqdn(request.remote_addr)
    if re.match(ALLOWED_SUFFIX_PATTERN, dns_suffix):
        return "Access granted"
    else:
        abort(403, "Access denied")

if __name__ == '__main__':
    app.run(port=5001)
