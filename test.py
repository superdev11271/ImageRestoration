"""Smoke-test a running Image Restoration server against a real image.

    python test.py test_images/a.png
    python test.py test_images/a.png --url http://127.0.0.1:9000

Checks `GET /api/health/`, `POST /api/restore/` with the given image, and that a
non-image upload is rejected. Exits non-zero if any check fails. The result is
written next to the input image as `test_images/a_out.png`, unless `-o` says otherwise.

Needs `requests`, a test-only dependency that is deliberately not in
requirements.txt so it stays out of the Docker image: `pip install requests`.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import requests

DEFAULT_URL = 'http://127.0.0.1:8080'
OUTPUT_SUFFIX = '_out.png'
HEALTH_ROUTE = '/api/health/'
ROUTE = '/api/restore/'
TIMEOUT = 300


def check(name, ok, seconds, detail=''):
    """Print one PASS/FAIL line with its response time and return ok, so run() can
    count the failures."""
    print(f'[{"PASS" if ok else "FAIL"}] {name} -- {seconds * 1000:.0f}ms' + (f' -- {detail}' if detail else ''))
    return ok


def timed(method, url, **kwargs):
    """Send one request and return (response, seconds). The clock covers the whole
    round trip including the response body, which is most of the time on an image."""
    start = time.perf_counter()
    response = requests.request(method, url, timeout=TIMEOUT, **kwargs)
    return response, time.perf_counter() - start


def run(args, source):
    """Run the three checks against the server and return them as a list of bools."""
    results = []
    form = {}

    response, seconds = timed('GET', args.url + HEALTH_ROUTE)
    body = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
    results.append(check('health', response.status_code == 200 and body.get('status') == 'ok',
                         seconds, str(body)))

    with open(args.image, 'rb') as handle:
        response, seconds = timed(
            'POST', args.url + ROUTE,
            files={'image': (os.path.basename(args.image), handle, 'image/png')}, data=form)
    is_png = response.status_code == 200 and response.headers.get('content-type') == 'image/png'
    output = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR) if is_png else None
    results.append(check('restore', output is not None, seconds,
                         f'{source.shape} -> {output.shape}' if output is not None
                         else f'{response.status_code} {response.text[:80]}'))
    if output is not None:
        cv2.imwrite(args.output, output)
        print(f'       wrote {args.output}')

    response, seconds = timed(
        'POST', args.url + ROUTE,
        files={'image': ('not_an_image.png', b'not an image', 'image/png')}, data=form)
    results.append(check('rejects a non-image', response.status_code == 400, seconds,
                         f'{response.status_code} {response.text[:80]}'))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='path to the image to send')
    parser.add_argument('--url', type=str, default=DEFAULT_URL, help='base url of the running server')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='where to write the returned png (default: the input path with '
                             f'{OUTPUT_SUFFIX} in place of its extension)')
    args = parser.parse_args()

    source = cv2.imread(args.image)
    if source is None:
        parser.error(f'cannot read {args.image} as an image')
    if args.output is None:
        # sits next to the input: test/cat.jpg -> test/cat_out.png
        args.output = os.path.splitext(args.image)[0] + OUTPUT_SUFFIX

    try:
        results = run(args, source)
    except requests.ConnectionError:
        print(f'[FAIL] cannot reach {args.url} -- is the server running?')
        return 1

    print(f'{sum(results)}/{len(results)} checks passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
