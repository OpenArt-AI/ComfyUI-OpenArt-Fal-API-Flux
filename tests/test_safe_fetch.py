import importlib.util
import os
import socket
import unittest
from unittest import mock

_P = os.path.join(os.path.dirname(__file__), "..", "modules", "safe_fetch.py")
_spec = importlib.util.spec_from_file_location("safe_fetch", _P)
sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sf)


def _dns(mapping):
    def fake(host, port, *a, **k):
        if host not in mapping:
            raise socket.gaierror("nx")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in mapping[host]]
    return fake


class FakeResp:
    def __init__(self, status, headers=None, body=b""):
        self.status = status
        self.headers = headers or {}
        self._body = body

    def stream(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def release_conn(self):
        pass


class FakePool:
    calls = []
    responses = []

    def __init__(self, host, **kw):
        FakePool.calls.append((host, kw))

    def urlopen(self, method, path, headers=None, **kw):
        assert kw["redirect"] is False
        FakePool.calls[-1] = FakePool.calls[-1] + (path, headers)
        return FakePool.responses.pop(0)

    def close(self):
        pass


class ValidateTests(unittest.TestCase):
    def test_rejects(self):
        for url in [
            "http://fal.media/x.png",
            "https://169.254.169.254/latest/meta-data",
            "https://127.0.0.1:8188/x",
            "https://evil.com/x.png",
            "https://fal.media.evil.com/x.png",
            "https://evilfal.media/x.png",
            "https://user:pw@fal.media/x.png",
            "https://fal.media:8188/x.png",
            "file:///etc/passwd",
            "gs://bucket/x",
            "https://[::1]/x",
            None,
        ]:
            with self.assertRaises(sf.UnsafeURLError, msg=url):
                sf.validate_url(url)

    def test_accepts(self):
        for url in [
            "https://fal.media/files/a.png",
            "https://v3.fal.media/files/a.png",
            "https://storage.googleapis.com/falserverless/a.png",
            "https://fal.media:443/files/a.png?x=1",
        ]:
            sf.validate_url(url)

    def test_private_dns_rejected(self):
        for ip in ["10.0.0.1", "127.0.0.1", "169.254.169.254", "100.64.0.1", "::1", "::ffff:10.0.0.1", "fd00::1"]:
            with mock.patch.object(sf.socket, "getaddrinfo", _dns({"v3.fal.media": ["1.1.1.1", ip]})):
                with self.assertRaises(sf.UnsafeURLError):
                    sf.resolve_public("v3.fal.media", 443)


class FetchTests(unittest.TestCase):
    def setUp(self):
        FakePool.calls = []
        FakePool.responses = []
        self.p = mock.patch.object(sf.urllib3, "HTTPSConnectionPool", FakePool)
        self.p.start()

    def tearDown(self):
        self.p.stop()

    def test_pins_validated_ip_and_sni(self):
        FakePool.responses = [FakeResp(200, {}, b"PNGDATA")]
        with mock.patch.object(sf.socket, "getaddrinfo", _dns({"v3.fal.media": ["34.1.2.3"]})):
            self.assertEqual(sf.safe_get_image_bytes("https://v3.fal.media/files/a.png?t=1"), b"PNGDATA")
        host, kw, path, headers = FakePool.calls[0]
        self.assertEqual(host, "34.1.2.3")
        self.assertEqual(kw["server_hostname"], "v3.fal.media")
        self.assertEqual(kw["assert_hostname"], "v3.fal.media")
        self.assertEqual(path, "/files/a.png?t=1")
        self.assertEqual(headers["Host"], "v3.fal.media")

    def test_redirect_to_internal_blocked(self):
        for loc in ["http://169.254.169.254/latest/", "https://127.0.0.1:8188/view", "https://metadata.google.internal/"]:
            FakePool.responses = [FakeResp(302, {"Location": loc})]
            with mock.patch.object(sf.socket, "getaddrinfo", _dns({"fal.media": ["34.1.2.3"]})):
                with self.assertRaises(sf.UnsafeURLError):
                    sf.safe_get_image_bytes("https://fal.media/a.png")

    def test_redirect_to_allowed_host_rebinding_blocked(self):
        FakePool.responses = [FakeResp(302, {"Location": "https://v2.fal.media/b.png"})]
        with mock.patch.object(sf.socket, "getaddrinfo", _dns({"fal.media": ["34.1.2.3"], "v2.fal.media": ["10.1.1.1"]})):
            with self.assertRaises(sf.UnsafeURLError):
                sf.safe_get_image_bytes("https://fal.media/a.png")

    def test_redirect_followed_when_valid(self):
        FakePool.responses = [FakeResp(301, {"Location": "/c.png"}), FakeResp(200, {}, b"ok")]
        with mock.patch.object(sf.socket, "getaddrinfo", _dns({"fal.media": ["34.1.2.3"]})):
            self.assertEqual(sf.safe_get_image_bytes("https://fal.media/a.png"), b"ok")
        self.assertEqual(FakePool.calls[1][2], "/c.png")

    def test_redirect_loop_capped(self):
        FakePool.responses = [FakeResp(302, {"Location": "/a.png"}) for _ in range(10)]
        with mock.patch.object(sf.socket, "getaddrinfo", _dns({"fal.media": ["34.1.2.3"]})):
            with self.assertRaises(IOError):
                sf.safe_get_image_bytes("https://fal.media/a.png")
        self.assertEqual(len(FakePool.calls), sf.MAX_REDIRECTS + 1)

    def test_size_cap(self):
        FakePool.responses = [FakeResp(200, {"Content-Length": str(sf.MAX_DOWNLOAD_BYTES + 1)}, b"x")]
        with mock.patch.object(sf.socket, "getaddrinfo", _dns({"fal.media": ["34.1.2.3"]})):
            with self.assertRaises(IOError):
                sf.safe_get_image_bytes("https://fal.media/a.png")


if __name__ == "__main__":
    unittest.main()
