# Example control_script for robot

def main(idp_controller):

    # Main loop, perform simulation steps until Webots is stopping the controller
    while idp_controller.step():
        pass
