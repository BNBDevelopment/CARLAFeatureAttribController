"""This file is the main controller file

Here, you will design the controller for your for the adaptive cruise control system.
"""

from mp1_simulator.simulator import Observation


# NOTE: Very important that the class name remains the same
class Controller:
    def __init__(self, target_speed: float, distance_threshold: float):
        self.target_speed = target_speed
        self.distance_threshold = distance_threshold

        self.prev_dist_estimate = None
        self.prev_step_gettingcloser = False

    def run_step(self, obs: Observation, estimate_dist) -> float:
        """This is the main run step of the controller.

        Here, you will have to read in the observatios `obs`, process it, and output an
        acceleration value. The acceleration value must be some value between -10.0 and 10.0.

        Note that the acceleration value is really some control input that is used
        internally to compute the throttle to the car.

        Below is some example code where the car just outputs the control value 10.0
        """

        ego_velocity = obs.velocity
        target_velocity = obs.target_velocity
        dist_to_lead = estimate_dist

        # print(f"ego_velocity: {ego_velocity}")
        # print(f"target_velocity: {target_velocity}")
        # print(f"PREDICTED dist_to_lead: {dist_to_lead}")
        # print(f"ACTUAL dist_to_lead: {obs.distance_to_lead}")

        max_dist_change_threshold = 1
        buffer = 10

        # if dist_to_lead == 30.0: #if we see the default distance value
        #     print(f"ERROR MISSED DIST: {dist_to_lead}")
        #     print(f"ACTUAL dist_to_lead: {obs.distance_to_lead}")

        #Check if the change in estimated values is too high between timesteps
        if self.prev_dist_estimate is not None:
            if abs(dist_to_lead - self.prev_dist_estimate) > max_dist_change_threshold:
                #print(f"MAJOR CHANGE -- dist_to_lead: {dist_to_lead}")
                #print(f"MAJOR CHANGE -- self.prev_dist_estimate: {self.prev_dist_estimate}")

                if self.prev_step_gettingcloser: #was getting closer before, so want the lower bound on estimated distance based
                    dist_to_lead = min(0, self.prev_dist_estimate - max_dist_change_threshold)
                else:  #otherwise we were getting farther away/stying same distance, so estimated distance for speed calcualtion can stay the same
                    dist_to_lead = self.prev_dist_estimate

            if self.prev_dist_estimate > dist_to_lead:
                self.prev_step_gettingcloser = True
            else:
                self.prev_step_gettingcloser = False

        break_dist = (2.2 * ego_velocity) + ((ego_velocity**2) / 20) + buffer

        #print(f"attribs of obj: {dir(obs)}")
        max_acc = 10
        min_acc = -10

        em_stop_dist = break_dist + self.distance_threshold
        #print(f"CALCULATION -- dist_to_lead: {dist_to_lead}")
        #print(f"ACTUAL      -- dist_to_lead: {obs.distance_to_lead}")
        #print(f"ACTUAL      -- ego_velocity: {ego_velocity}")
        #print(f"ACTUAL      -- break_dist: {break_dist}")

        if dist_to_lead < self.distance_threshold:
            actual_target_velocity = 0
            acc_modifier = 10
        elif dist_to_lead < em_stop_dist:
            actual_target_velocity = 0
            acc_modifier = 10
        else:
            actual_target_velocity = target_velocity
            acc_modifier = 10

        if ego_velocity > actual_target_velocity:
            return_val = -acc_modifier
        else:
            return_val = acc_modifier


        if return_val > 0:
            return_val = min(return_val, max_acc)
        else:
            return_val = max(return_val, min_acc)

        #print(f"ACTUAL      -- return_val: {return_val}")
        #print(f"ACTUAL      -- actual_target_velocity: {actual_target_velocity}")
        self.prev_dist_estimate = dist_to_lead
        return return_val
