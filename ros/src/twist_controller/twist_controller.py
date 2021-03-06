from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
        accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        # Emperically determined
        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0. # minimum throttle value
        mx = 0.2 # maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        # For smoothing noisy velocity
        tau = 0.5
        ts = .02
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, linear_vel_target, angular_vel_target, linear_vel_current, dbw_enabled):
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        linear_vel_filt = self.vel_lpf.filt(linear_vel_current)

        steering = self.yaw_controller.get_steering(linear_vel_target, angular_vel_target, linear_vel_filt)

        vel_error = linear_vel_target - linear_vel_filt

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel_target == 0. and linear_vel_filt < 0.1:
            throttle = 0
            brake = 700 # N*m to hold the car (Carla) in place if we are stopped at a light.
        elif throttle < self.brake_deadband and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            # Braking Torque = Vehicle mass * wheel radius * desired acceleration
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
        return throttle, brake, steering
