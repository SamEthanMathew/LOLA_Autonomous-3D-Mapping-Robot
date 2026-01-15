import curses
import sys
from gpiozero import Motor, PWMOutputDevice


class Vehicle:
    def __init__(self):
        self.FL_motor = Motor(forward=27, backward=22)
        self.FL_pwm = PWMOutputDevice(17)
        self.FL_pwm.value = 0

        self.FR_motor = Motor(forward=24, backward=23)
        self.FR_pwm = PWMOutputDevice(25)
        self.FR_pwm.value = 0

        self.BL_motor = Motor(forward=6, backward=5)
        self.BL_pwm = PWMOutputDevice(13)
        self.BL_pwm.value = 0

        self.BR_motor = Motor(forward=16, backward=26)
        self.BR_pwm = PWMOutputDevice(12)
        self.BR_pwm.value = 0
        

    def forward(self):
        self.FL_motor.forward()
        self.FR_motor.forward()
        self.BL_motor.forward()
        self.BR_motor.forward()

        self.FL_pwm.value = 0.5
        self.FR_pwm.value = 0.5
        self.BL_pwm.value = 0.5
        self.BR_pwm.value = 0.5

    def backward(self):
        self.FL_motor.backward()
        self.FR_motor.backward()
        self.BL_motor.backward()
        self.BR_motor.backward()

        self.FL_pwm.value = 0.5
        self.FR_pwm.value = 0.5
        self.BL_pwm.value = 0.5
        self.BR_pwm.value = 0.5

    def slide_left(self):
        self.FL_motor.backward()
        self.FR_motor.forward()
        self.BL_motor.forward()
        self.BR_motor.backward()

        self.FL_pwm.value = 0.5
        self.FR_pwm.value = 0.5
        self.BL_pwm.value = 0.5
        self.BR_pwm.value = 0.5

    def slide_right(self):
        self.FL_motor.forward()
        self.FR_motor.backward()
        self.BL_motor.backward()
        self.BR_motor.forward()

        self.FL_pwm.value = 0.5
        self.FR_pwm.value = 0.5
        self.BL_pwm.value = 0.5
        self.BR_pwm.value = 0.5

    def turn_left(self):
        self.FL_motor.backward()
        self.FR_motor.forward()
        self.BL_motor.backward()
        self.BR_motor.forward()

        self.FL_pwm.value = 0.5
        self.FR_pwm.value = 0.5
        self.BL_pwm.value = 0.5
        self.BR_pwm.value = 0.5

    def turn_right(self):
        self.FL_motor.forward()
        self.FR_motor.backward()
        self.BL_motor.forward()
        self.BR_motor.backward()

        self.FL_pwm.value = 0.5
        self.FR_pwm.value = 0.5
        self.BL_pwm.value = 0.5
        self.BR_pwm.value = 0.5

    def map_key_to_command(self, key):
        map = {
            curses.KEY_UP: self.forward,
            curses.KEY_DOWN: self.backward,
            curses.KEY_LEFT: self.turn_left,
            curses.KEY_RIGHT: self.turn_right,
            ord('w'): self.forward,
            ord('s'): self.backward,
            ord('a'): self.slide_left,
            ord('d'): self.slide_right,
            ord('q'): self.turn_left,
            ord('e'): self.turn_right
        }
        return map[key]

    def control(self, key):
        return self.map_key_to_command(key)


rpi_vehicle = Vehicle()


def main(window):
    next_key = None

    while True:
        curses.halfdelay(1)
        if next_key is None:
            key = window.getch()
            print(key)
        else:
            key = next_key
            next_key = None
        if key == 99:
            sys.exit()
        if key != -1:
            # KEY PRESSED
            curses.halfdelay(1)
            action = rpi_vehicle.control(key)
            if action:
                action()
            next_key = key
            while next_key == key:
                next_key = window.getch()
            # KEY RELEASED
            rpi_vehicle.FL_motor.stop()
            rpi_vehicle.FR_motor.stop()
            rpi_vehicle.BL_motor.stop()
            rpi_vehicle.BR_motor.stop()


curses.wrapper(main)

# thank you mcdominik
