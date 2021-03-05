# Test communication between robots
import struct

def main(robot):
    #message = "h"
    timestep = int(robot.getBasicTimeStep())
    format = "4s h d"

    while robot.step(timestep) != -1:
        if robot.getName() == 'optimal':
            message = struct.pack(format, b'babc', 45, 120.08)
            robot.emitter.send(message)
        else:
            receiver = robot.receiver
            if receiver.getQueueLength() > 0:
                data = receiver.getData()
                print(struct.unpack(format, data)[0])
                receiver.nextPacket()