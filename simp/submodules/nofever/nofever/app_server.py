import queue
import flask
from utils import DebugPrint, CONFIG_NOFEVER_SETTINGS

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)

SETTINGS = CONFIG_NOFEVER_SETTINGS['main']

IP_ADDRESS = SETTINGS['IP_ADDRESS']
PORT = SETTINGS['PORT']

app = flask.Flask(__name__)


class MessageAnnouncer:
    def __init__(self):
        self.listeners = []

    def listen(self):
        self.listeners.append(queue.Queue(maxsize=5))
        return self.listeners[-1]

    def announce(self, msg):
        # We go in reverse order because we might have to delete an element, which will shift the
        # indices backward
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg)
            except queue.Full:
                del self.listeners[i]


announcer = MessageAnnouncer()


def format_sse(data: str, event=None) -> str:
    """Formats a string and an event name in order to follow the event stream convention.
    >>> format_sse(data=json.dumps({'abc': 123}), event='Jackson 5')
    'event: Jackson 5\\ndata: {"abc": 123}\\n\\n'
    """
    msg = f'data: {data}\n\n'
    if event is not None:
        msg = f'event: {event}\n{msg}'
    return msg


@app.route('/detection_idle')
def detection_idle():
    msg = format_sse(data='detection_idle')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/detection_active')
def detection_active():
    msg = format_sse(data='detection_active')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/hardware_not_loaded')
def hardware_not_loaded():
    msg = format_sse(data='hardware_not_loaded')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/new_cycle')
def new_cycle():
    msg = format_sse(data='new_cycle')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/movement_begin')
def movement_begin():
    msg = format_sse(data='movement_begin')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/movement_end')
def movement_end():
    msg = format_sse(data='movement_end')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/measurement_begin')
def measurement_begin():
    msg = format_sse(data='measurement_begin')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/measurement_end')
def measurement_end():
    msg = format_sse(data='measurement_end')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/temperature_measured_good')
def temperature_measured_good():
    msg = format_sse(data='temperature_measured_good')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/temperature_measured_slightly_high')
def temperature_measured_slightly_high():
    msg = format_sse(data='temperature_measured_slightly_high')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/temperature_measured_bad')
def temperature_measured_bad():
    msg = format_sse(data='temperature_measured_bad')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/temperature_measured_wrong')
def temperature_measured_wrong():
    msg = format_sse(data='temperature_measured_wrong')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/temperature_measured_under_twice')
def temperature_measured_under_twice():
    msg = format_sse(data='temperature_measured_under_twice')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/temperature_measured_over_twice')
def temperature_measured_over_twice():
    msg = format_sse(data='temperature_measured_over_twice')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/mask_on')
def mask_on():
    msg = format_sse(data='mask_on')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/mask_off')
def mask_off():
    msg = format_sse(data='mask_off')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/mask_wrong')
def mask_wrong():
    msg = format_sse(data='mask_wrong')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/ping')
def ping():
    msg = format_sse(data='pong')
    announcer.announce(msg=msg)
    return {}, 200


@app.route('/listen', methods=['GET'])
def listen():

    def stream():
        messages = announcer.listen()  # returns a queue.Queue
        while True:
            msg = messages.get()  # blocks until a new message arrives
            yield msg

    return flask.Response(stream(), mimetype='text/event-stream')


DEBUG('Starting server at http://{0}:{1}'.format(IP_ADDRESS, PORT))
app.run(host=IP_ADDRESS, port=PORT, debug=True, threaded=True)
