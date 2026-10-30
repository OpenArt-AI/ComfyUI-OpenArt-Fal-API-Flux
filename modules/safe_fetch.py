"""SSRF-safe download of result images returned by the fal API.

The image URLs this node downloads come from a third-party (fal) response,
so they are treated as untrusted:

* only https, default port, no userinfo;
* host must be a fal result host (fal.media / fal.ai / fal.run and their
  subdomains, or storage.googleapis.com which fal also serves results from);
* the host is resolved once, every resolved address must be public, and the
  TCP connection is made to that validated IP (TLS SNI / certificate checks
  still use the hostname) so DNS rebinding cannot swap in an internal IP;
* redirects are not followed automatically: each hop is re-validated with the
  same rules, at most MAX_REDIRECTS hops;
* the body is capped at MAX_DOWNLOAD_BYTES.
"""
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import urllib3

ALLOWED_HOST_SUFFIXES = ("fal.media", "fal.ai", "fal.run")
ALLOWED_EXACT_HOSTS = ("storage.googleapis.com",)
MAX_REDIRECTS = 3
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
TIMEOUT = urllib3.Timeout(connect=10, read=60)


class UnsafeURLError(ValueError):
    pass


def _host_allowed(host: str) -> bool:
    host = host.lower().rstrip(".")
    if host in ALLOWED_EXACT_HOSTS:
        return True
    return any(host == s or host.endswith("." + s) for s in ALLOWED_HOST_SUFFIXES)


def is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def validate_url(url: str):
    """Return (host, port, path_with_query) for an allowed URL or raise."""
    if not isinstance(url, str):
        raise UnsafeURLError("URL must be a string")
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise UnsafeURLError(f"scheme not allowed: {parts.scheme!r}")
    if parts.username is not None or parts.password is not None:
        raise UnsafeURLError("userinfo not allowed")
    host = parts.hostname
    if not host:
        raise UnsafeURLError("missing host")
    try:
        port = parts.port
    except ValueError as e:
        raise UnsafeURLError(f"bad port: {e}") from e
    if port not in (None, 443):
        raise UnsafeURLError(f"port not allowed: {port}")
    try:
        ipaddress.ip_address(host)
        raise UnsafeURLError("IP-literal hosts are not allowed")
    except ValueError:
        pass
    if not _host_allowed(host):
        raise UnsafeURLError(f"host not allowed: {host}")
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    return host, 443, path


def resolve_public(host: str, port: int) -> str:
    """Resolve once; every address must be public. Returns the IP to dial."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise UnsafeURLError(f"cannot resolve {host}: {e}") from e
    addrs = [info[4][0] for info in infos]
    if not addrs:
        raise UnsafeURLError(f"no addresses for {host}")
    for a in addrs:
        if not is_public_address(a):
            raise UnsafeURLError(f"{host} resolves to non-public address")
    return addrs[0]


def _fetch_once(url: str):
    host, port, path = validate_url(url)
    ip = resolve_public(host, port)
    pool = urllib3.HTTPSConnectionPool(
        ip,
        port=port,
        server_hostname=host,
        assert_hostname=host,
        cert_reqs="CERT_REQUIRED",
        timeout=TIMEOUT,
        retries=False,
        maxsize=1,
    )
    try:
        resp = pool.urlopen(
            "GET",
            path,
            headers={"Host": host, "User-Agent": "ComfyUI-OpenArt-Fal-API-Flux"},
            redirect=False,
            retries=False,
            preload_content=False,
        )
        try:
            if resp.status in (301, 302, 303, 307, 308):
                return resp.status, resp.headers.get("Location"), None
            if resp.status >= 400:
                raise IOError(f"HTTP {resp.status} downloading image")
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_DOWNLOAD_BYTES:
                raise IOError("image too large")
            buf = bytearray()
            for chunk in resp.stream(1024 * 1024):
                buf += chunk
                if len(buf) > MAX_DOWNLOAD_BYTES:
                    raise IOError("image too large")
            return resp.status, None, bytes(buf)
        finally:
            resp.release_conn()
    finally:
        pool.close()


def safe_get_image_bytes(url: str) -> bytes:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        status, location, body = _fetch_once(current)
        if body is not None:
            return body
        if not location:
            raise IOError(f"redirect {status} without Location")
        current = urljoin(current, location)
    raise IOError("too many redirects")
