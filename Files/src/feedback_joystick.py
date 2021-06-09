import pygame

pygame.display.init()
pygame.joystick.init()
pygame.joystick.Joystick(0).init()

pygame.event.pump()


class feedbackJoystick:
    def __init__(self):
        self.joystick_axis = None

        self.threshold = 0.1


    def get_h(self):
        x_axis = pygame.joystick.Joystick(0).get_axis(0)
        y_axis = pygame.joystick.Joystick(0).get_axis(1)
        z_axis = pygame.joystick.Joystick(0).get_axis(4)


        if  abs(x_axis) > self.threshold or abs(y_axis) > self.threshold or abs(z_axis) > self.threshold:
            if abs(x_axis) < self.threshold:
                x_axis = 0
            if abs(y_axis) < self.threshold:
                y_axis = 0
            if abs(z_axis) < self.threshold:
                z_axis = 0



            self.joystick_axis = [x_axis, -1 * y_axis, -1 * z_axis, 1]

        else:
            self.joystick_axis = None

        return self.joystick_axis





