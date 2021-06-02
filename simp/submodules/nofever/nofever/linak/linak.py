# Application to let your desk dance.
# Copyright (C) 2018 Lukas Schreiner <dev@lschreiner.de>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program. If not, see <https://www.gnu.org/licenses/>.
from usb.core import find as finddev
from ctypes import sizeof
import time
import usb1

from utils import LOG, CONFIG_NOFEVER_SETTINGS, DebugPrint

DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)

SETTINGS = CONFIG_NOFEVER_SETTINGS['linak']

REQ_INIT = 0x0303
REQ_GET_STATUS = 0x0304
REQ_MOVE = 0x0305
REQ_GET_EXT = 0x0309

TYPE_SET_CI = 0x21
TYPE_GET_CI = 0xA1

HID_REPORT_GET = 0x01
HID_REPORT_SET = 0x09

STATUS_REPORT = 4			# Adress to get Data/Report
STATUS_REPORT_LENGTH = 64 	# Amount of bytes of the status package
NRB_STATUS_REPORT = 56

SET_OPERATION_MODE = 3  # Setting operation mode to 3 so we can communicate through USB2LIN
GET_DATA = 4  			# Get data/status
CONTROL_CBD = 5  		# CBD is what they call thier control boxes


LINAK_TIMEOUT = 1000  # checking if we can decrease to 300 ms

MOVE_DOWNWARDS = 32767 	# 0x7fff / "Running the table downwards"
MOVE_UPWARDS = 32768  	# 0x8000	/ "Running the table upwards"
STOP_MOVEMENT = 32769  	# 0x8001	/ "For no activation"


class Status(object):
    positionLost = True
    antiColision = True
    overloadDown = True
    overloadUp = True
    unknown = 4

    @classmethod
    def fromBuf(sr, buf):
        no_connection_cnt = 0
        no_connection_max = 10
        self = sr()
        attr = ['positionLost', 'antiColision', 'overloadDown', 'overloadUp']
        bitlist = '{:0>8s}'.format(bin(int(buf, base=16)).lstrip('0b'))
        for i in range(0, 4):
            setattr(self, attr[i], True if bitlist[i] == '1' else False)
        self.unknown = int(buf[1:], 16)
        if "11100000" in bitlist:
            # DEBUG("Moving Down")
            pass
        elif "00010000" in bitlist:
            # DEBUG("Moving Up")
            pass
        elif "00000000" in bitlist:
            # abcde = 1
            pass
        elif "11110000" in bitlist:
            # DEBUG("Collision detected")
            pass
        elif "00000001" in bitlist:
            DEBUG("Resetting Linak in Action")
            time_start = time.time()
            while "00000001" in bitlist:
                time.sleep(0.2)
                if time.time() - time_start >= 10:
                    raise ConnectionError('Linak stuck at "Resetting Linak in action".')
                time.sleep(0.2)
                bitlist = '{:0>8s}'.format(bin(int(buf, base=16)).lstrip('0b'))
        else:
            no_connection_cnt += 1
            DEBUG('no_connection_cnt = {}'.format(no_connection_cnt))
            if no_connection_cnt >= no_connection_max:
                raise ConnectionError('No connection to Linak')
            DEBUG('bitlist = {}'.format(bitlist))
            DEBUG('self.unknown = {}'.format(self.unknown))
        return self


class StatusPositionSpeed(object):
    pos = None
    status = None
    speed = 0

    @classmethod
    def fromBuf(sr, buf):
        self = sr()
        self.pos = int(buf[2:4] + buf[:2], 16)
        self.status = Status.fromBuf(buf[4:6])
        self.speed = int(buf[6:8], 16)

        return self


class ValidFlags(object):
    ID00_Ref1_pos_stat_speed = True
    ID01_Ref2_pos_stat_speed = True
    ID02_Ref3_pos_stat_speed = True
    ID03_Ref4_pos_stat_speed = True
    ID10_Ref1_controlInput = True
    ID11_Ref2_controlInput = True
    ID12_Ref3_controlInput = True
    ID13_Ref4_controlInput = True
    ID04_Ref5_pos_stat_speed = True
    ID28_Diagnostic = True
    ID05_Ref6_pos_stat_speed = True
    ID37_Handset1command = True
    ID38_Handset2command = True
    ID06_Ref7_pos_stat_speed = True
    ID07_Ref8_pos_stat_speed = True
    unknown = True

    @classmethod
    def fromBuf(sr, buf):
        self = sr()
        attr = ['ID00_Ref1_pos_stat_speed',
                'ID01_Ref2_pos_stat_speed',
                'ID02_Ref3_pos_stat_speed',
                'ID03_Ref4_pos_stat_speed',
                'ID10_Ref1_controlInput',
                'ID11_Ref2_controlInput',
                'ID12_Ref3_controlInput',
                'ID13_Ref4_controlInput',
                'ID04_Ref5_pos_stat_speed',
                'ID28_Diagnostic',
                'ID05_Ref6_pos_stat_speed',
                'ID37_Handset1command',
                'ID38_Handset2command',
                'ID06_Ref7_pos_stat_speed',
                'ID07_Ref8_pos_stat_speed',
                'unknown']
        bitlist = '{:0>16s}'.format(bin(int(buf, base=16)).lstrip('0b'))
        for i in range(0, len(bitlist)):
            setattr(self, attr[i], True if bitlist[i] == '1' else False)

        return self


class StatusReport(object):
    featureRaportID = 0
    numberOfBytes = 0
    validFlag = None
    ref1 = None
    ref2 = None
    ref3 = None
    ref4 = None
    ref1cnt = 0
    ref2cnt = 0
    ref3cnt = 0
    ref4cnt = 0
    ref5 = None
    diagnostic = None
    undefined1 = None
    handset1 = 0
    handset2 = 0
    ref6 = None
    ref7 = None
    ref8 = None
    undefined2 = None

    @classmethod
    def fromBuf(sr, buf):
        self = sr()
        raw = buf.hex()
        self.featureRaportID = buf[0]
        self.numberOfBytes = buf[1]
        self.validFlag = ValidFlags.fromBuf(raw[4:8])
        self.ref1 = StatusPositionSpeed.fromBuf(raw[8:8 + 8])
        self.ref2 = StatusPositionSpeed.fromBuf(raw[16:16 + 8])
        self.ref3 = StatusPositionSpeed.fromBuf(raw[24:24 + 8])
        self.ref4 = StatusPositionSpeed.fromBuf(raw[32:32 + 8])
        self.ref1cnt = int(raw[42:44] + raw[40:42], 16)
        self.ref2cnt = int(raw[46:48] + raw[44:46], 16)
        self.ref3cnt = int(raw[50:52] + raw[48:50], 16)
        self.ref4cnt = int(raw[54:56] + raw[52:54], 16)
        self.ref5 = StatusPositionSpeed.fromBuf(raw[56:56 + 8])
        self.diagnostic = raw[64:64 + 16]
        self.undefined1 = raw[80:84]
        self.handset1 = int(raw[86:88] + raw[84:86], 16)
        self.handset2 = int(raw[88:90] + raw[86:88], 16)
        self.ref6 = StatusPositionSpeed.fromBuf(raw[90:90 + 8])
        self.ref7 = StatusPositionSpeed.fromBuf(raw[98:98 + 8])
        self.ref8 = StatusPositionSpeed.fromBuf(raw[106:106 + 8])
        self.undefined2 = raw[114:]

        return self


class LinakController(object):
    # Vendor ID. Usually the same for all Linak columns. Check by running 'lsusb' in UNIX terminal.
    LINAK_VENDOR_ID = 0x12d3
    # Product ID. Usually the same for all Linak columns. Check by running 'lsusb' in UNIX terminal.
    LINAK_PRODUCT_ID = 0x0002

    LINAK_MOVING = False
    LINAK_HEIGHT = 0
    LINAK_HEIGHT_ABSOLUTE = 0
    HEIGHT = 0
    _handle = None
    _ctx = None

    NOFEVER_ABSOLUTE_MINIMUM_HEIGHT = SETTINGS.getint('NOFEVER_ABSOLUTE_MINIMUM_HEIGHT')
    MAX_DISTANCE_abs = SETTINGS.getint('MAX_DISTANCE_abs')
    MAX_DISTANCE_rel = MAX_DISTANCE_abs - NOFEVER_ABSOLUTE_MINIMUM_HEIGHT
    MIN_DISTANCE_abs = SETTINGS.getint('MIN_DISTANCE_abs')
    MIN_DISTANCE_rel = MIN_DISTANCE_abs - NOFEVER_ABSOLUTE_MINIMUM_HEIGHT

    FOOT_HEIGHT = SETTINGS.getint('FOOT_HEIGHT')
    NECK_HEIGHT = SETTINGS.getint('NECK_HEIGHT')
    HEAD_HEIGHT = SETTINGS.getint('HEAD_HEIGHT')
    HORN_HEIGHT = SETTINGS.getint('HORN_HEIGHT')
    STATIC_MIN_HEIGHT = FOOT_HEIGHT + NECK_HEIGHT + HEAD_HEIGHT + HORN_HEIGHT

    def __init__(self, vendor_id=LINAK_VENDOR_ID, product_id=LINAK_PRODUCT_ID):
        self._ctx = usb1.USBContext()
        # self._ctx.setDebug(4)
        self._handle = self._ctx.openByVendorIDAndProductID(
            vendor_id,
            product_id,
            skip_on_error=True,
        )
        if self._handle is None:
            raise Exception('Could not connect to Linak USB device at vendorID {0} and productID {1}.'.format(
                vendor_id, product_id))
        self._handle.setAutoDetachKernelDriver(True)
        self._handle.claimInterface(0)
        self._initDevice()

        try:
            self.getHeight()
        except ConnectionError:
            DEBUG('Linak stuck in "Resetting Linak in Action". Moving it down...')
            for x in range(250):
                self._moveDown()
                time.sleep(0.1)
            time.sleep(1)
            try:
                self.getHeight()
            except ConnectionError:
                DEBUG('Linak still stuck even after _moveDown attempt.')
                raise SystemError('Linak is dead.')

    def __del__(self):
        self.close()

    def close(self):  # NOTE HACK Throws and error when code is closed with ctrl+c
        dev = finddev(idVendor=self.LINAK_VENDOR_ID, idProduct=self.LINAK_PRODUCT_ID)
        dev.reset()  # Resetting USB port
        DEBUG('Linak USB port is rebooted')
        LOG.critical('Linak USB port is rebooted')
        time.sleep(1)
        # if self._handle:
        #     DEBUG('Linak USB interface is released.')
        #     self._handle.releaseInterface(0)
        del(self._handle)
        del(self._ctx)

    def _controlWriteRead(self, request_type, request, value, index, data, timeout=0):
        data, data_buffer = usb1.create_initialised_buffer(data)
        transferred = self._handle._controlTransfer(request_type, request, value, index, data,
                                                    sizeof(data), timeout)
        return transferred, data_buffer[:transferred]

    def _getStatusReport(self):
        buf = bytearray(b'\x00' * STATUS_REPORT_LENGTH)
        buf[0] = STATUS_REPORT
        # DEBUG('> {:s}'.format(buf.hex()))
        x, buf = self._controlWriteRead(TYPE_GET_CI, HID_REPORT_GET, REQ_GET_STATUS, 0, buf, LINAK_TIMEOUT)
        # DEBUG(buf)

        # check if the response match to request!
        if buf[0] != STATUS_REPORT:
            raise Exception('Invalid status report received!')

        return buf

    def _setStatusReport(self):
        buf = bytearray(b'\x00' * STATUS_REPORT_LENGTH)
        buf[0] = SET_OPERATION_MODE
        buf[1] = GET_DATA
        buf[2] = 0
        buf[3] = 251

        x, buf = self._controlWriteRead(TYPE_SET_CI, HID_REPORT_SET, REQ_INIT, 0, buf, LINAK_TIMEOUT)

        if x != STATUS_REPORT_LENGTH:
            raise Exception('Device is not ready yet. Initialization failed in step 1.')

    def _move(self, height):
        buf = bytearray(b'\x00' * STATUS_REPORT_LENGTH)
        buf[0] = CONTROL_CBD

        hHex = '{:04x}'.format(height)
        hHigh = int(hHex[2:], 16)
        hLow = int(hHex[:2], 16)

        buf[1] = hHigh
        buf[2] = hLow
        buf[3] = hHigh
        buf[4] = hLow
        buf[5] = hHigh
        buf[6] = hLow
        buf[7] = hHigh
        buf[8] = hLow

        x, buf = self._controlWriteRead(TYPE_SET_CI, HID_REPORT_SET, REQ_MOVE, 0, buf, LINAK_TIMEOUT)
        return x == STATUS_REPORT_LENGTH

    def _moveDown(self):
        return self._move(MOVE_DOWNWARDS)

    def _moveUp(self):
        return self._move(MOVE_UPWARDS)

    def _moveEnd(self):
        return self._move(STOP_MOVEMENT)

    def _isStatusReportNotReady(self, buf):
        if buf[0] != STATUS_REPORT or buf[1] != NRB_STATUS_REPORT:
            return False

        for i in range(2, STATUS_REPORT_LENGTH - 5):
            if buf[i] != 0:
                return False

        return True

    def _initDevice(self):
        buf = self._getStatusReport()
        if not self._isStatusReportNotReady(buf):
            return
        else:
            DEBUG('Device not ready!')

        self._setStatusReport()
        time.sleep(0.0001)
        if not self._moveEnd():
            raise Exception('Device not ready - initialization failed on step 2 (moveEnd)')

        time.sleep(0.1)

    def getHeight(self):
        buf = self._getStatusReport()
        r = StatusReport.fromBuf(buf)
        self.LINAK_HEIGHT = r.ref1.pos
        self.LINAK_HEIGHT_ABSOLUTE = r.ref1.pos + self.STATIC_MIN_HEIGHT
        return r.ref1.pos  # , r.ref1.pos / 98.0

    def move_threaded(self, target_height, stop):
        self.LINAK_MOVING = True
        a = max_a = 3
        epsilon = 13
        oldH = 0
        self.HEIGHT = target_height
        while True:
            self._move(self.HEIGHT)
            time.sleep(0.2)
            buf = self._getStatusReport()
            r = StatusReport.fromBuf(buf)
            self.LINAK_HEIGHT = r.ref1.pos
            self.LINAK_HEIGHT_ABSOLUTE = r.ref1.pos + self.STATIC_MIN_HEIGHT
            distance = r.ref1cnt - r.ref1.pos
            delta = oldH - r.ref1.pos
            if abs(distance) <= epsilon or abs(delta) <= epsilon or oldH == r.ref1.pos:
                a -= 1
            else:
                a = max_a
            # s = 'Current height: {:d}; target height: {:d}; distance: {:d}'.format(r.ref1.pos, self.HEIGHT, distance)
            # DEBUG(s)
            if a == 0:
                self.LINAK_MOVING = False
                break

            oldH = r.ref1.pos

            if stop():
                self.LINAK_MOVING = False
                break
        # return abs(r.ref1.pos - self.HEIGHT) <= epsilon
        return r.ref1.pos
