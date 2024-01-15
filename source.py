# read data from serial port and parse it

import serial
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

# init serial port
ser = serial.Serial()
ser.baudrate = 921600
ser.port = '/dev/ttyACM2'
ser.open()

MAX_NUM_OBJECTS = 200;
OBJ_STRUCT_SIZE_BYTES = 10;
MAX_NUM_CLUSTERS = 24;
CLUSTER_STRUCT_SIZE_BYTES = 8;
MAX_NUM_TRACKERS = 24;
TRACKER_STRUCT_SIZE_BYTES = 12;
STATS_SIZE_BYTES = 16;
MMWDEMO_UART_MSG_DETECTED_POINTS    = 1;
MMWDEMO_UART_MSG_CLUSTERS           = 2;
MMWDEMO_UART_MSG_TRACKED_OBJ        = 3;
MMWDEMO_UART_MSG_PARKING_ASSIST     = 4;

bytevec_cp_len = 0
bytevecAccLen = 0
bytevec_cp_max_len = 2**15
bytevec_cp = np.zeros(bytevec_cp_max_len, dtype='uint8')

bytebufferlength = 0

magicok = 0
countok = 0

barker_code = [2, 1, 4, 3, 6, 5, 8, 7]

#get header
def getHeader(bytevec, idx):
    idx = idx + 8
    word = [1, 256, 65536, 16777216]
    header = {}
    # multiply by 256^0, 256^1, 256^2, 256^3 and add resulting in unit32
    header['version'] = int(np.frombuffer(bytevec[idx:idx+4], dtype='uint32'))
    idx = idx + 4
    header['totalPacketLen'] = int(np.frombuffer(bytevec[idx:idx+4], dtype='uint32'))
    idx = idx + 4
    header['platform'] = int(np.frombuffer(bytevec[idx:idx+4], dtype='uint32'))
    idx = idx + 4
    header['frameNumber'] = int(np.frombuffer(bytevec[idx:idx+4], dtype='uint32'))
    idx = idx + 4
    header['timeCpuCycles'] = int(np.frombuffer(bytevec[idx:idx+4], dtype='uint32'))
    idx = idx + 4
    header['numDetectedObj'] = int(np.frombuffer(bytevec[idx:idx+4], dtype='uint32'))
    idx = idx + 4
    header['numTLVs'] = int(np.frombuffer(bytevec[idx:idx+4], dtype='uint32'))
    idx = idx + 4
    header['subFrameNumber'] = int(np.frombuffer(bytevec[idx:idx+4], dtype='uint32'))
    idx = idx + 4
    return header, idx

#get tlv
def getTlv(bytevec, idx):
    tlv = {}
    tlv['type'] = int(np.frombuffer(bytevec[idx:idx+4], dtype='uint32'))
    idx = idx + 4
    tlv['length'] = int(np.frombuffer(bytevec[idx:idx+4], dtype='uint32'))
    idx = idx + 4
    return tlv, idx

def getDetObj(bytevec, idx, tlvlen, detObj):
    detObj['numObj'] = 0
    len_bytevec = len(bytevec)

    if len_bytevec < idx + 4:
        idx = len_bytevec
        return detObj, idx
    
    if tlvlen > 0:
        detObj['numObj'] = int(np.frombuffer(bytevec[idx:idx+2], dtype='int16'))
        idx = idx + 2
        xyzQFormat = 2**int(np.frombuffer(bytevec[idx:idx+2], dtype='int16'))
        idx = idx + 2
        invxyzQFormat = 1.0/float(xyzQFormat)
        if len_bytevec < idx + detObj['numObj']*OBJ_STRUCT_SIZE_BYTES:
            detObj['numObj'] = 0
            idx = len_bytevec
            return detObj, idx
        bytes = bytevec[idx:idx+detObj['numObj']*OBJ_STRUCT_SIZE_BYTES]
        idx = idx + detObj['numObj']*OBJ_STRUCT_SIZE_BYTES

        bytes = np.reshape(bytes, (detObj['numObj'], OBJ_STRUCT_SIZE_BYTES))
        detObj['doppler'] = bytes[:, 0] + bytes[:, 1]*256
        detObj['peakVal'] = bytes[:, 2] + bytes[:, 3]*256

        detObj['x'] = np.array(bytes[:, 4] + bytes[:, 5]*256, dtype='int16')
        detObj['y'] = np.array(bytes[:, 6] + bytes[:, 7]*256, dtype='int16')
        detObj['z'] = np.array(bytes[:, 8] + bytes[:, 9]*256, dtype='int16')

        
        
        detObj['x'][detObj['x'] > 32767] = detObj['x'][detObj['x'] > 32767] - 65536
        detObj['y'][detObj['y'] > 32767] = detObj['y'][detObj['y'] > 32767] - 65536
        detObj['z'][detObj['z'] > 32767] = detObj['z'][detObj['z'] > 32767] - 65536

        detObj['doppler'][detObj['doppler'] > 32767] = detObj['doppler'][detObj['doppler'] > 32767] - 65536

        detObj['x'] = detObj['x']*invxyzQFormat
        detObj['y'] = detObj['y']*invxyzQFormat
        detObj['z'] = -detObj['z']*invxyzQFormat
        detObj['doppler'] = detObj['doppler']*invxyzQFormat

        # print("detObj['x']: ", detObj['x'])
        # print("detObj['y']: ", detObj['y'])
        # print("detObj['z']: ", detObj['z'])

        detObj['range'] = np.sqrt(detObj['x']**2 + detObj['y']**2 + detObj['z']**2)

        # print("detObj['range']: ", detObj['range'])
    
    return detObj, idx

def getCluster(bytevec, idx, tlvlen, cluster):
    len_bytevec = len(bytevec)
    if len_bytevec < idx + 4:
        idx = len_bytevec
        return cluster, idx
    
    if tlvlen > 0:
        cluster['numObj'] = int(np.frombuffer(bytevec[idx:idx+2], dtype='int16'))
        idx = idx + 2
        onebyxyzqformat = 1.0/(2 ** int(np.frombuffer(bytevec[idx:idx+2], dtype='int16')))
        idx = idx + 2
        if len_bytevec < idx + cluster['numObj']*CLUSTER_STRUCT_SIZE_BYTES:
            cluster['numObj'] = 0
            idx = len_bytevec
            return cluster, idx
        
        bytes = bytevec[idx:idx+cluster['numObj']*CLUSTER_STRUCT_SIZE_BYTES]
        idx = idx + cluster['numObj']*CLUSTER_STRUCT_SIZE_BYTES

        bytes = np.reshape(bytes, (cluster['numObj'], CLUSTER_STRUCT_SIZE_BYTES))
        
        x = np.array(bytes[:, 0] + bytes[:, 1]*256, dtype='int16')
        y = np.array(bytes[:, 2] + bytes[:, 3]*256, dtype='int16')

        x[x > 32767] = x[x > 32767] - 65536
        y[y > 32767] = y[y > 32767] - 65536

        x = x*onebyxyzqformat
        y = y*onebyxyzqformat

        x_size = bytes[:, 4] + bytes[:, 5]*256
        y_size = bytes[:, 6] + bytes[:, 7]*256
        
        x_size = x_size*onebyxyzqformat
        y_size = y_size*onebyxyzqformat

        area = x_size*y_size * 4

        x_size[area > 20] = 99999

        x_loc = x + x_size * [-1, 1, 1, -1, -1, 99999]
        y_loc = y + y_size * [-1, -1, 1, 1, -1, 99999]

        cluster['x_loc'] = x_loc
        cluster['y_loc'] = y_loc

    return cluster, idx

def getTracker(bytevec, idx, tlvlen, tracker):

    tracker['numObj'] = 0
    len_bytevec = len(bytevec)

    if len_bytevec < idx + 4:
        idx = len_bytevec
        return tracker, idx
    
    if tlvlen > 0:
        tracker['numObj'] = int(np.frombuffer(bytevec[idx:idx+2], dtype='int16'))
        idx = idx + 2
        xyzQFormat = 2**int(np.frombuffer(bytevec[idx:idx+2], dtype='int16'))
        idx = idx + 2
        invxyzQFormat = 1.0/float(xyzQFormat)

        if len_bytevec < idx + tracker['numObj']*TRACKER_STRUCT_SIZE_BYTES:
            tracker['numObj'] = 0
            idx = len_bytevec
            return tracker, idx
        
        bytes = bytevec[idx:idx+tracker['numObj']*TRACKER_STRUCT_SIZE_BYTES]
        idx = idx + tracker['numObj']*TRACKER_STRUCT_SIZE_BYTES

        bytes = np.reshape(bytes, (tracker['numObj'], TRACKER_STRUCT_SIZE_BYTES))

        tracker['x'] = np.array(bytes[:, 0] + bytes[:, 1]*256, dtype='int16')
        tracker['y'] = np.array(bytes[:, 2] + bytes[:, 3]*256, dtype='int16')

        tracker['x'][tracker['x'] > 32767] = tracker['x'][tracker['x'] > 32767] - 65536
        tracker['y'][tracker['y'] > 32767] = tracker['y'][tracker['y'] > 32767] - 65536

        tracker['x'] = tracker['x']*invxyzQFormat
        tracker['y'] = tracker['y']*invxyzQFormat

        tracker['vx'] = bytes[:, 4] + bytes[:, 5]*256
        tracker['vy'] = bytes[:, 6] + bytes[:, 7]*256

        tracker['vx'][tracker['vx'] > 32767] = tracker['vx'][tracker['vx'] > 32767] - 65536
        tracker['vy'][tracker['vy'] > 32767] = tracker['vy'][tracker['vy'] > 32767] - 65536

        tracker['vx'] = tracker['vx']*invxyzQFormat
        tracker['vy'] = tracker['vy']*invxyzQFormat

        x_size = bytes[:, 8] + bytes[:, 9]*256
        y_size = bytes[:, 10] + bytes[:, 11]*256

        x_size = x_size*invxyzQFormat
        y_size = y_size*invxyzQFormat

        # convert x_size to a numpy array of 1 columns
        x_size = np.array(x_size).reshape(-1, 1)
        y_size = np.array(y_size).reshape(-1, 1)



        x_loc = np.array(tracker['x']).reshape(-1,1) + x_size * np.array([[-1, 1, 1, -1, -1, 99999]])
        y_loc = np.array(tracker['y']).reshape(-1,1) + y_size * np.array([[-1, 1, 1, -1, -1, 99999]])

        tracker['cluster_x_loc'] = x_loc
        tracker['cluster_y_loc'] = y_loc

        tracker['range'] = np.sqrt(tracker['x']**2 + tracker['y']**2)

        print("res: ", tracker['x'] * tracker['vx'] + tracker['y'] * tracker['vy'])
        print("tracker['range']: ", tracker['range'])

    
        tracker['doppler'] = (tracker['x']*tracker['vx'] + tracker['y']*tracker['vy'])/tracker['range']

        # print("tracker['doppler']: ", tracker['doppler'])

    return tracker, idx


    
#get bytes to read from serial port
while True:
    if ser.in_waiting > 0:
        ser_data = ser.read(ser.in_waiting)
        bytebufferlength = len(ser_data)
        if (bytevec_cp_len + bytebufferlength) < bytevec_cp_max_len:
            bytevec_cp[bytevec_cp_len:bytevec_cp_len+bytebufferlength] = np.frombuffer(ser_data, dtype='uint8')
            bytevec_cp_len += bytebufferlength
            bytebufferlength = 0
            # print("bytevec_cp_len: ", bytevec_cp_len)
        else:
            print("Error: bytevec_cp overflow")
            bytebufferlength = 0
            bytevec_cp_len = 0
            bytevec_cp = np.zeros(bytevec_cp_max_len, dtype='uint8')
    
    bytevecStr = bytevec_cp[0:bytevec_cp_len].tostring()
    magicok = 0
    if (bytevec_cp_len>72):
        startIdx = bytevecStr.find(b'\x02\x01\x04\x03\x06\x05\x08\x07')
    else:
        startIdx = -1
    
    if (startIdx != -1):
        if (startIdx >= 0):
            # print("startIdx: ", startIdx)
            magicok = 1
            countok = 0
            bytevecAccLen = bytevec_cp_len - startIdx
            bytevec_cp[0:bytevecAccLen] = bytevec_cp[startIdx:bytevec_cp_len]
            bytevec_cp_len = bytevecAccLen
            # bytevecStr = bytevec_cp[0:bytevec_cp_len].tostring()
            # print("bytevec_cp_len: ", bytevec_cp_len)
            # print("bytevecAccLen: ", bytevecAccLen)
            # print("bytevecStr: ", bytevecStr[0], bytevecStr[1], bytevecStr[2], bytevecStr[3])
        else:
            print("Error: startIdx < 0")
            magicok = 0
            countok = 0
            bytevec_cp_len = 0
            bytevec_cp = np.zeros(bytevec_cp_max_len, dtype='uint8')

    byteVecIdx = 0
    if (magicok == 1):
        # start time
        tStart = time.time()
        # for i in range(len(bytevec_cp)):
        #     print(bytevec_cp[i])
        #get header info
        header, byteVecIdx = getHeader(bytevec_cp, byteVecIdx)
        sfIdx = header['subFrameNumber'] + 1
        if (sfIdx > 2) | (header['numDetectedObj'] > MAX_NUM_OBJECTS):
            continue
        
        detObj = {}
        cluster = {}
        tracker = {}
        detObj['numObj'] = 0
        cluster['numObj'] = 0
        tracker['numObj'] = 0

        # print header info
        # print("header['version']: ", header['version'])
        # print("header['totalPacketLen']: ", header['totalPacketLen'])
        # print("header['numTLVs']: ", header['numTLVs'])
        # #get detected objects
        for tlvidx in range(header['numTLVs']):

            tlv, byteVecIdx = getTlv(bytevec_cp, byteVecIdx)
            # print("tlv['type']: ", tlv['type'])

            if tlv['type'] == MMWDEMO_UART_MSG_DETECTED_POINTS:
                if tlv['length'] >= OBJ_STRUCT_SIZE_BYTES:
                    detObj, byteVecIdx = getDetObj(bytevec_cp, byteVecIdx, tlv['length'], detObj)
            
            elif tlv['type'] == MMWDEMO_UART_MSG_CLUSTERS:
                if tlv['length'] >= CLUSTER_STRUCT_SIZE_BYTES:
                    cluster, byteVecIdx = getCluster(bytevec_cp, byteVecIdx, tlv['length'], cluster)

            elif tlv['type'] == MMWDEMO_UART_MSG_TRACKED_OBJ:
                if tlv['length'] >= TRACKER_STRUCT_SIZE_BYTES:
                    tracker, byteVecIdx = getTracker(bytevec_cp, byteVecIdx, tlv['length'], tracker)
        print("detObj['numObj']: ", detObj['numObj'])
        num = detObj['numObj']
        for i in range(0,num):
            # print("detObj['x'][i]: ", detObj['x'][i])
            # print("detObj['y'][i]: ", detObj['y'][i])
            # # print("detObj['z'][i]: ", detObj['z'][i])
            # print("detObj['range'][i]: ", detObj['range'][i])
            #print x y range in  same line upto 3 decimal places in euqal spacing tab seperated
            # print("x     y     range")
            print("%.3f %.3f %.3f" % (detObj['x'][i], detObj['y'][i], detObj['range'][i]))
        print("******************************")


