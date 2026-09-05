# Local TLS fixture

These DER files are a self-signed localhost certificate and an intentionally
public test-only PKCS#8 private key. They are used only by the in-process TLS
fixture server. They are not production credentials, release-signing keys,
or a production trust root.

Generated with OpenSSL on 2026-09-06, valid for 36500 days, SAN DNS:localhost,
basicConstraints CA:FALSE, extendedKeyUsage serverAuth. Tests explicitly
trust this certificate through a private test-only client configuration.
The normal downloader uses rustls's standard public certificate roots and
must reject this fixture certificate.

No external server, openssl installation at test time, or model weights are
required. The fixture binds an ephemeral loopback port and closes on drop.
