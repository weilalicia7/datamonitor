#!/usr/bin/env bash
#
# Generate a self-signed TLS certificate for the local docker-compose stack.
# Output: nginx/certs/{fullchain,privkey}.pem
#
# USE ONLY FOR LOCAL DEV.  Production must provision real certificates from
# a trusted CA (Let's Encrypt, AWS ACM, internal CA) and mount them read-only.

set -euo pipefail

CERT_DIR="$(dirname "$0")/../nginx/certs"
mkdir -p "$CERT_DIR"

if [ -f "$CERT_DIR/fullchain.pem" ] && [ -f "$CERT_DIR/privkey.pem" ]; then
    echo "TLS cert already present under $CERT_DIR — skipping."
    exit 0
fi

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$CERT_DIR/privkey.pem" \
    -out    "$CERT_DIR/fullchain.pem" \
    -subj "/C=GB/ST=Wales/L=Cardiff/O=SACT-Scheduler-Local/OU=Dev/CN=localhost" \
    2> /dev/null

chmod 600 "$CERT_DIR/privkey.pem"
chmod 644 "$CERT_DIR/fullchain.pem"

echo "Self-signed TLS cert written to $CERT_DIR/"
echo "  fullchain: $CERT_DIR/fullchain.pem"
echo "  privkey:   $CERT_DIR/privkey.pem"
