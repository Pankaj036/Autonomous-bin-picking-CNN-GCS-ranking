#!/usr/bin/env python3
import socket, struct, time, sys

GRIPPER_IP, GRIPPER_PORT = "192.168.1.105", 502
UNIT_ID    = 0x00
REG_OUTPUT = 0x0000
REG_INPUT  = 0x0000
_tid = 0

def _tid_next():
    global _tid
    _tid = (_tid+1)&0xFFFF
    return _tid

def _mbap(tid, pdu):
    return struct.pack(">HHH", tid, 0, len(pdu)) + pdu

def _drain(s):
    s.settimeout(0.05)
    try:
        while s.recv(256): pass
    except: pass
    s.settimeout(2.0)

def _write(s, vals):
    tid  = _tid_next()
    data = b"".join(struct.pack(">H", v) for v in vals)
    pdu  = struct.pack(">BBHHB", UNIT_ID, 0x10, REG_OUTPUT, len(vals), len(vals)*2) + data
    s.sendall(_mbap(tid, pdu))
    try:
        s.settimeout(1.0)
        s.recv(256)
    except: pass
    s.settimeout(2.0)

def _read_status(s):
    _drain(s)
    tid = _tid_next()
    pdu = struct.pack(">BBHH", UNIT_ID, 0x04, REG_INPUT, 3)
    s.sendall(_mbap(tid, pdu))
    try: return s.recv(256)
    except: return b""

def parse(resp):
    if len(resp) < 15: return None
    if resp[7] & 0x80:
        print("Modbus exception 0x{:02X}".format(resp[8]))
        return None
    d = resp[9:15]
    return {
        "gACT": d[0] & 0x01,
        "gSTA": (d[0]>>4) & 0x03,
        "gFLT": d[2] & 0x0F,
        "raw":  d.hex()
    }

def activate():
    print("Connecting to {}:{} ...".format(GRIPPER_IP, GRIPPER_PORT))
    s = socket.socket()
    s.settimeout(5.0)
    s.connect((GRIPPER_IP, GRIPPER_PORT))
    s.settimeout(2.0)
    print("TCP connected")
    r = _read_status(s)
    st = parse(r)
    if st:
        print("Current: gACT={} gSTA={} gFLT={} raw={}".format(st['gACT'],st['gSTA'],st['gFLT'],st['raw']))
        if st['gACT']==1 and st['gSTA']==3:
            print("Gripper already ACTIVATED and ready!")
            s.close()
            return True
    print("RESET ...")
    _write(s, [0,0,0])
    time.sleep(1.0)
    _drain(s)
    print("ACTIVATE ...")
    _write(s, [0x0100,0,0])
    time.sleep(0.5)
    _drain(s)
    print("Polling for gSTA=3 ...")
    for i in range(60):
        r = _read_status(s)
        st = parse(r)
        if not st:
            print("  [{}] no valid response: {}".format(i+1, r.hex() if r else 'empty'))
        else:
            print("  [{}] gACT={} gSTA={} gFLT={} raw={}".format(i+1,st['gACT'],st['gSTA'],st['gFLT'],st['raw']))
            if st['gFLT']:
                print("  Fault - retrying ...")
                _write(s, [0,0,0])
                time.sleep(1.0)
                _write(s, [0x0100,0,0])
                time.sleep(0.5)
                _drain(s)
            elif st['gACT']==1 and st['gSTA']==3:
                print("Gripper ACTIVATED and ready!")
                s.close()
                return True
        time.sleep(0.3)
    print("TIMED OUT")
    s.close()
    return False

sys.exit(0 if activate() else 1)
