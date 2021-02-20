from controller import Robot, GPS, Compass, Motor


class IDPCompass(Compass):
    def __init__(self, name, sampling_rate):
        super().__init__(name)
        self.enable(sampling_rate)


class IDPGPS(GPS):
    def __init__(self, name, sampling_rate):
        super().__init__(name)  # default to infinite resolution
        if self.getCoordinateSystem() != 0:
            raise RuntimeWarning('Need to set GPS coordinate system in WorldInfo to local')
        self.enable(sampling_rate)


class IDPMotor(Motor):
    def __init__(self, name):
        super().__init__(name)
        # self.setPosition(float('inf'))
        self.setVelocity(0.0)


class IDPRobot(Robot):
    def __init__(self):
        super().__init__()

        self.timestep = int(self.getBasicTimeStep())  # get the time step of the current world.

        self.left_motor = self.getDevice('wheel1')
        self.right_motor = self.getDevice('wheel2')

        self.gps = self.getDevice('gps')  # or use createGPS() directly
        self.compass = self.getDevice('compass')

    # .getDevice() will call createXXX if the tag name is not in __devices[]
    def createCompass(self, name: str) -> IDPCompass:  # override method to use the custom Compass class
        return IDPCompass(name, self.timestep)

    def createGPS(self, name: str) -> IDPGPS:
        return IDPGPS(name, self.timestep)

    def createMotor(self, name: str) -> IDPMotor:
        return IDPMotor(name)

    @property
    def position(self) -> list:
        return self.gps.getValues()

    @property
    def speed(self) -> float:
        return self.gps.getSpeed()

    @property
    def bearing(self) -> float:
        return self.compass.getValues()[2]

    def drive_to_position(self, target):
        pass
