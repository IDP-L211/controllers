# Test communication between robots
import struct

def main(robot):
    #message = 5.123456789123456789
    timestep = int(robot.getBasicTimeStep())
    #format = "d"
    sent = False

    while robot.step(timestep) != -1:
        if robot.check_receiver():
            #print("Received packets")
            print(robot.get_and_decode())
            print(robot.get_other_pos())

        if robot.getName() == 'optimal':
            if sent == False:

                robot.ask_position()

                '''message = struct.pack(format, message)
                robot.emitter.send(message)
                coords = robot.position
                print(coords)
                for i in coords:
                    message = struct.pack(format, i)
                    robot.emitter.send(message)'''

                sent = True
        else:


            '''receiver = robot.receiver
            if receiver.getQueueLength() > 0:
                data = receiver.getData()
                #print(struct.unpack(format, data)[0].decode('utf-8'))
                print(float(struct.unpack(format, data)[0]))
                print(receiver.getDataSize())
                receiver.nextPacket()'''