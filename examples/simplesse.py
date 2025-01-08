# https://github.com/overcat/requests-sse

import json
import time
import logging
from datetime import datetime
import requests
from requests_sse import EventSource, InvalidStatusCodeError, InvalidContentTypeError

url = "https://idrac-R650-9Z38ZZZ.example.com/redfish/v1/SSE?$filter=EventFormatType%20eq%20MetricReport"

logger = logging.getLogger("requests_sse")
logger.setLevel(logging.DEBUG)

def handle_event(event_source):
    for event in event_source:
        #print(event)
        print(datetime.now())
        print(type(event))
        j = json.loads(event.data)
        print(json.dumps(j, indent=4))

with EventSource(url, max_connect_retry=180, timeout=60, auth=("root", "L@bT3@m>C0vid"), verify=False) as event_source:
    try:
        handle_event(event_source)
    except (requests.RequestException, InvalidStatusCodeError):
        # When an iDRAC is reset or becomes unreachable retry connection
        # ERROR: failed with wrong response status: 500
        while True:
            try: 
                time.sleep(60)
                logger.debug("DEBUG: RECONNECTING")
                reconnect_event_source = event_source.connect(retry=180)
                handle_event(reconnect_event_source)
                break
            except (requests.RequestException, InvalidStatusCodeError):
                pass