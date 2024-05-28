import numpy as np
from tqdm import trange
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from flygym.simulation import SingleFlySimulation, Fly
from flygym.examples.common import PreprogrammedSteps
from flygym.examples.cpg_controller import CPGNetwork
from flygym.preprogrammed import get_cpg_biases

from pathlib import Path
import pickle
from flygym.preprogrammed import all_leg_dofs
from flygym.util import get_data_path

_tripod_phase_biases = get_cpg_biases("tripod")
_tripod_coupling_weights = (_tripod_phase_biases > 0) * 10
_default_correction_vectors = {
    "F": np.array([0, 0, 0, -0.02, 0, 0.016, 0]),
    "M": np.array([-0.015, 0, 0, 0.004, 0, 0.01, -0.008]),
    "H": np.array([0, 0, 0, -0.01, 0, 0.005, 0]),
}
_default_correction_rates = {"retraction": (500, 1000 / 3), "stumbling": (2000, 500)}
_contact_sensor_placements = tuple(
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
)


class WalkingFly(Fly):
    def __init__(
        self,
        timestep,
        preprogrammed_steps=None,
        intrinsic_freqs=np.ones(6) * 12,
        intrinsic_amps=np.ones(6) * 1,
        phase_biases=_tripod_phase_biases,
        coupling_weights=_tripod_coupling_weights,
        convergence_coefs=np.ones(6) * 20,
        init_phases=None,
        init_magnitudes=None,
        stumble_segments=("Tibia", "Tarsus1", "Tarsus2"),
        stumbling_force_threshold=-1,
        correction_vectors=_default_correction_vectors,
        correction_rates=_default_correction_rates,
        amplitude_range=(-0.5, 1.5),
        draw_corrections=False,
        contact_sensor_placements=_contact_sensor_placements,
        seed=0,
        **kwargs,
    ):
        # Initialize core NMF simulation
        super().__init__(contact_sensor_placements=contact_sensor_placements, **kwargs)

        if preprogrammed_steps is None:
            preprogrammed_steps = PreprogrammedSteps()

        self.preprogrammed_steps = preprogrammed_steps
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.phase_biases = phase_biases
        self.coupling_weights = coupling_weights
        self.convergence_coefs = convergence_coefs
        self.stumble_segments = stumble_segments
        self.stumbling_force_threshold = stumbling_force_threshold
        self.correction_vectors = correction_vectors
        self.correction_rates = correction_rates
        self.amplitude_range = amplitude_range
        self.draw_corrections = draw_corrections

        self.action_space = spaces.Box(*amplitude_range, shape=(2,))
        # Initialize CPG network
        self.cpg_network = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=intrinsic_freqs,
            intrinsic_amps=intrinsic_amps,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=convergence_coefs,
            seed=seed,
        )
        self.cpg_network.reset(init_phases, init_magnitudes)

        # Initialize variables tracking the correction amount
        self.retraction_correction = np.zeros(6)
        self.stumbling_correction = np.zeros(6)

        # Find stumbling sensors
        self.stumbling_sensors = self._find_stumbling_sensor_indices()

        # ------------------------------------------------------------------------------------------------------------------------
        actuated_joints = all_leg_dofs

    @property
    def timestep(self):
        return self.cpg_network.timestep

    def _find_stumbling_sensor_indices(self):
        stumbling_sensors = {leg: [] for leg in self.preprogrammed_steps.legs}
        for i, sensor_name in enumerate(self.contact_sensor_placements):
            leg = sensor_name.split("/")[1][:2]  # sensor_name: e.g. "Animat/LFTarsus1"
            segment = sensor_name.split("/")[1][2:]
            if segment in self.stumble_segments:
                stumbling_sensors[leg].append(i)
        stumbling_sensors = {k: np.array(v) for k, v in stumbling_sensors.items()}
        if any(
            v.size != len(self.stumble_segments) for v in stumbling_sensors.values()
        ):
            raise RuntimeError(
                "Contact detection must be enabled for all tibia, tarsus1, and tarsus2 "
                "segments for stumbling detection."
            )
        return stumbling_sensors

    def _retraction_rule_find_leg(self, obs):
        """Returns the index of the leg that needs to be retracted, or None
        if none applies."""
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.05:
            leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]
        else:
            leg_to_correct_retraction = None
        return leg_to_correct_retraction

    def _stumbling_rule_check_condition(self, obs, leg):
        """Return True if the leg is stumbling, False otherwise."""
        # update stumbling correction amounts
        contact_forces = obs["contact_forces"][self.stumbling_sensors[leg], :]
        fly_orientation = obs["fly_orientation"]
        # force projection should be negative if against fly orientation
        force_proj = np.dot(contact_forces, fly_orientation)
        return (force_proj < self.stumbling_force_threshold).any()

    def _get_net_correction(self, retraction_correction, stumbling_correction):
        """Retraction correction has priority."""
        if retraction_correction > 0:
            return retraction_correction
        return stumbling_correction

    def _update_correction_amount(
        self, condition, curr_amount, correction_rates, viz_segment
    ):
        """Update correction amount and color code leg segment.

        Parameters
        ----------
        condition : bool
            Whether the correction condition is met.
        curr_amount : float
            Current correction amount.
        correction_rates : Tuple[float, float]
            Correction rates for increment and decrement.
        viz_segment : str
            Name of the segment to color code. If None, no color coding is
            done.

        Returns
        -------
        float
            Updated correction amount.
        """
        if condition:  # lift leg
            increment = correction_rates[0] * self.timestep
            new_amount = curr_amount + increment
            color = (0, 1, 0, 1)
        else:  # condition no longer met, lower leg
            decrement = correction_rates[1] * self.timestep
            new_amount = max(0, curr_amount - decrement)
            color = (1, 0, 0, 1)
        if viz_segment is not None:
            self.change_segment_color(viz_segment, color)
        return new_amount

    def reset(self, sim, seed=None, init_phases=None, init_magnitudes=None, **kwargs):
        obs, info = super().reset(sim, seed=seed, **kwargs)
        self.cpg_network.random_state = np.random.RandomState(seed)
        self.cpg_network.intrinsic_amps = self.intrinsic_amps
        self.cpg_network.intrinsic_freqs = self.intrinsic_freqs
        self.cpg_network.reset(init_phases, init_magnitudes)
        self.retraction_correction = np.zeros(6)
        self.stumbling_correction = np.zeros(6)
        return obs, info

    def pre_step(self, action, sim):
        """Step the simulation forward one timestep.

        Parameters
        ----------
        action : np.ndarray
            Array of shape (2,) containing descending signal encoding turning
        """
        physics = sim.physics

        # update CPG parameters
        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        # get current observation
        obs = super().get_observation(sim)

        # Retraction rule: is any leg stuck in a gap and needing to be retracted?
        leg_to_correct_retraction = self._retraction_rule_find_leg(obs)

        self.cpg_network.step()

        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            # update retraction correction amounts
            self.retraction_correction[i] = self._update_correction_amount(
                condition=(i == leg_to_correct_retraction),
                curr_amount=self.retraction_correction[i],
                correction_rates=self.correction_rates["retraction"],
                viz_segment=f"{leg}Tibia" if self.draw_corrections else None,
            )
            # update stumbling correction amounts
            self.stumbling_correction[i] = self._update_correction_amount(
                condition=self._stumbling_rule_check_condition(obs, leg),
                curr_amount=self.stumbling_correction[i],
                correction_rates=self.correction_rates["stumbling"],
                viz_segment=f"{leg}Femur" if self.draw_corrections else None,
            )
            # get net correction amount
            net_correction = self._get_net_correction(
                self.retraction_correction[i], self.stumbling_correction[i]
            )

            # get target angles from CPGs and apply correction
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            my_joints_angles += net_correction * self.correction_vectors[leg[1]]
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )
            adhesion_onoff.append(my_adhesion_onoff)

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }

        return super().pre_step(action, sim)


class LungingFly(Fly):
    def __init__(
        self,
        timestep,
        preprogrammed_steps=None,
        intrinsic_freqs=np.ones(6) * 12,
        intrinsic_amps=np.ones(6) * 1,
        phase_biases=_tripod_phase_biases,
        coupling_weights=_tripod_coupling_weights,
        convergence_coefs=np.ones(6) * 20,
        init_phases=None,
        init_magnitudes=None,
        stumble_segments=("Tibia", "Tarsus1", "Tarsus2"),
        stumbling_force_threshold=-1,
        correction_vectors=_default_correction_vectors,
        correction_rates=_default_correction_rates,
        amplitude_range=(-0.5, 1.5),
        obj_threshold=0.15,
        draw_corrections=False,
        contact_sensor_placements=_contact_sensor_placements,
        seed=0,
        **kwargs,
    ):
        # Initialize core NMF simulation
        super().__init__(
            contact_sensor_placements=contact_sensor_placements,
            enable_vision=True,
            **kwargs
        )

        if preprogrammed_steps is None:
            preprogrammed_steps = PreprogrammedSteps()


        self.obj_threshold = obj_threshold
        self.coms = np.empty((self.retina.num_ommatidia_per_eye, 2))
        self.visual_inputs_hist = []

        self.preprogrammed_steps = preprogrammed_steps
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.phase_biases = phase_biases
        self.coupling_weights = coupling_weights
        self.convergence_coefs = convergence_coefs
        self.stumble_segments = stumble_segments
        self.stumbling_force_threshold = stumbling_force_threshold
        self.correction_vectors = correction_vectors
        self.correction_rates = correction_rates
        self.amplitude_range = amplitude_range
        self.draw_corrections = draw_corrections

        # Define action and observation spaces

        self.action_space = spaces.Box(*amplitude_range, shape=(3,))    #Add the 3rd dimension to determine wheather start lunging or not 


        # Initialize CPG network
        self.cpg_network = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=intrinsic_freqs,
            intrinsic_amps=intrinsic_amps,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=convergence_coefs,
            seed=seed,
        )
        self.cpg_network.reset(init_phases, init_magnitudes)

        # Initialize variables tracking the correction amount
        self.retraction_correction = np.zeros(6)
        self.stumbling_correction = np.zeros(6)

        # Find stumbling sensors
        self.stumbling_sensors = self._find_stumbling_sensor_indices()

        # Then modify the data in fly0. The data was modifyies by observing the lunging video.
        # np.linspace(statring phase, endphase, # of frames)
        # each linspace list is one action that is actuated linearly
        # data["join name"]=np.concatenate([transitioning, squading, standing, leaning, attacking])

        actuated_joints = all_leg_dofs
        
        self.data_path = get_data_path("flygym", "data")
        self.render_mode="saved"
        self.render_playspeed=0.2, 
        self.draw_contacts=True
        with open(self.data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
            data = pickle.load(f)
        with open(self.data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
            data1=pickle.load(f)

        data["joint_LFCoxa"]=np.concatenate([np.linspace(1, 0, 200),np.linspace(0, 0, 100),np.linspace(0, 0, 100),np.linspace(0, 0, 100),np.linspace(0, 0, 100)]).tolist()
        data["joint_LFCoxa_roll"]=np.concatenate([np.linspace(-0.5, 0.8, 200),np.linspace(0.8, 0.8, 100),np.linspace(0.8, 0.8, 100),np.linspace(0.8, 0.8,100),np.linspace(0.8, 0.3, 100)]).tolist()
        data["joint_LFCoxa_yaw"]=np.concatenate([np.linspace(0.5, 1.5, 200),np.linspace(1.5,1.5, 100),np.linspace(1.5,1.5, 100),np.linspace(1.5,1.5, 100),np.linspace(1.5, 1.5, 100)]).tolist()
        data["joint_LFFemur"]=np.concatenate([np.linspace(-2.8, -1.5, 200),np.linspace(-1.5,-1.5, 100),np.linspace(-1.5,-1.5, 100),np.linspace(-1.5,-1.5,100),np.linspace(-1.5, -1.5, 100)]).tolist()
        data["joint_LFFemur_roll"]=np.concatenate([np.linspace(-1.5, -1.5, 200),np.linspace(-1.5,-1.5, 100),np.linspace(-1.5,-1.5,100),np.linspace(-1.5,-1.5, 100),np.linspace(-1.5,-1.5, 100)]).tolist()
        data["joint_LFTibia"]=np.concatenate([np.linspace(1, -1.17,200),np.linspace(-1.17,-1.1, 100),np.linspace(-1.1,-1.1 ,100),np.linspace(-1.1,-1.1, 100),np.linspace(-1.1, -0.9, 100)]).tolist()
        data["joint_LFTarsus1"]=np.concatenate([np.linspace(-0.5, -0.5, 200),np.linspace(-0.5,-0.21 ,100),np.linspace(-0.2,0, 100),np.linspace(0,0, 100),np.linspace(0, -0.6, 100)]).tolist()

        data["joint_LMCoxa"]=np.concatenate([np.linspace(0.11, 0.11, 200),np.linspace(0.11,0.12, 100),np.linspace(0.12,0.6, 100),np.linspace(0.6,0.8, 100),np.linspace(1, -0.6, 100)]).tolist()
        data["joint_LMCoxa_roll"]=np.concatenate([np.linspace(2, 0.1, 200),np.linspace(0.1,0.1 ,100),np.linspace(0.1,0.1,100),np.linspace(0.1,0.1 ,100),np.linspace(0.1, 0.1, 100)]).tolist()
        data["joint_LMCoxa_yaw"]=np.concatenate([np.linspace(0.16, 0.4, 200),np.linspace(0.4,0.4, 100),np.linspace(0.4,0.4, 100),np.linspace(0.4,0.6, 100),np.linspace(0.6, 0.4, 100)]).tolist()
        data["joint_LMFemur"]=np.concatenate([np.linspace(-1.45, -1.68, 200),np.linspace(-1.68,-0.798 , 100),np.linspace(-0.798,-0.528, 100),np.linspace(-0.528,-0.528,100),np.linspace(-0.428, -0.498, 100)]).tolist()
        data["joint_LMFemur_roll"]=np.concatenate([np.linspace(0.37, 0, 200),np.linspace(0,0 , 100,),np.linspace(0,0, 100),np.linspace(0,0 , 100),np.linspace(0, 0,100)]).tolist()
        data["joint_LMTibia"]=np.concatenate([np.linspace(1.84, 1.74, 200),np.linspace(1.74,0.5, 100),np.linspace(0.5,0.2, 100),np.linspace(0.2,0.2, 100),np.linspace(0.2,1.5, 100)]).tolist()
        data["joint_LMTarsus1"]=np.concatenate([np.linspace(-0.81, -0.2, 200),np.linspace(-0.2,-0.33 ,100),np.linspace(-0.33,-0.1, 100),np.linspace(-0.1,-0.1 ,100),np.linspace(-0.1, -1.2, 100)]).tolist()

        #
        data["joint_LHCoxa"]=np.concatenate([np.linspace(0.4, 0.4, 200),np.linspace(0.4,0.21, 100),np.linspace(0.21,-0.2, 100),np.linspace(-0.2,-0.2, 100),np.linspace(-0.2, 0.6, 100)]).tolist()
        data["joint_LHCoxa_roll"]=np.concatenate([np.linspace(2.5, 2.5, 200),np.linspace(2.5,2.5,100),np.linspace(2.5,2.5, 100),np.linspace(2.5,2.5, 100),np.linspace(2.5, 2.61, 100)]).tolist()
        data["joint_LHCoxa_yaw"]=np.concatenate([np.linspace(0.2,0.2, 200),np.linspace(0.2,0.2,100),np.linspace(0.2,0.2, 100),np.linspace(0.2, 0, 100),np.linspace(0, 0.2, 100)]).tolist()
        data["joint_LHFemur"]=np.concatenate([np.linspace(-1.89, -1.89, 200),np.linspace(-1.89,-1.82 ,100),np.linspace(-1.82,-1.2, 100),np.linspace(-1.2,-1.2 ,100),np.linspace(-1.2, -0.2, 100)]).tolist()
        data["joint_LHFemur_roll"]=np.concatenate([np.linspace(0.18, 0, 200),np.linspace(0,0 , 100),np.linspace(0,0, 100),np.linspace(0,0 , 100),np.linspace(0, 0, 100)]).tolist()
        data["joint_LHTibia"]=np.concatenate([np.linspace(2, 2.15, 200),np.linspace(2.07,2.15, 100),np.linspace(2.15,0.1, 100),np.linspace(0.1,0.1, 100),np.linspace(0.1, 0.1, 100)]).tolist()
        data["joint_LHTarsus1"]=np.concatenate([np.linspace(-0.5, -0.9, 200),np.linspace(-0.9,-1.1, 100),np.linspace(-1.1,-0.2, 100),np.linspace(-0.2,-0.2,100),np.linspace(-0.2, -0.2, 100)]).tolist()

        data["joint_RFCoxa"]=data["joint_LFCoxa"]
        data["joint_RFCoxa_roll"]=[-x for x in data["joint_LFCoxa_roll"]]
        data["joint_RFCoxa_yaw"]=[-x for x in data["joint_LFCoxa_yaw"]]
        data["joint_RFFemur"]=data["joint_LFFemur"]
        data["joint_RFFemur_roll"]=[-x for x in data["joint_LFFemur_roll"]]
        data["joint_RFTibia"]=data["joint_LFTibia"]
        data["joint_RFTarsus1"]=data["joint_LFTarsus1"]

        data["joint_RMCoxa"]=data["joint_LMCoxa"]
        data["joint_RMCoxa_roll"]=[-x for x in data["joint_LMCoxa_roll"]]
        data["joint_RMCoxa_yaw"]=[-x for x in data["joint_LMCoxa_yaw"]]
        data["joint_RMFemur"]=data["joint_LMFemur"]
        data["joint_RMFemur_roll"]=[-x for x in data["joint_LMFemur_roll"]]
        data["joint_RMTibia"]=data["joint_LMTibia"]
        data["joint_RMTarsus1"]=data["joint_LMTarsus1"]

        data["joint_RHCoxa"]=data["joint_LHCoxa"]
        data["joint_RHCoxa_roll"]=[-x for x in data["joint_LHCoxa_roll"]]
        data["joint_RHCoxa_yaw"]=[-x for x in data["joint_LHCoxa_yaw"]]
        data["joint_RHFemur"]=data["joint_LHFemur"]
        data["joint_RHFemur_roll"]=[-x for x in data["joint_LHFemur_roll"]]
        data["joint_RHTibia"]=data["joint_LHTibia"]
        data["joint_RHTarsus1"]=data["joint_LHTarsus1"]



        self.input_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
        self.output_t = np.arange(5000) * 1e-4
        data_block = np.zeros((len(actuated_joints), 5000)) #Set the time of the datablock
        input_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
        output_t = np.arange(5000) * timestep



        # The data for both flies are interpolated
        for i, joint in enumerate(actuated_joints):
            data_block[i, :] = np.interp(output_t, input_t, data[joint])


        self.data_block1 = np.zeros((len(actuated_joints), 5000))
        self.input_t = np.arange(len(data1["joint_LFCoxa"])) * data1["meta"]["timestep"]
        for i, joint in enumerate(actuated_joints):
            self.data_block1[i, :] = np.interp(self.output_t, self.input_t, data1[joint])

        self.data = data_block

        self.i_lunging = 0  # To keep track of where it is in the whole lunging action 

    @property
    def timestep(self):
        return self.cpg_network.timestep

    # Processing the visual input by average weighted pixels
    def average_pixels(self, vision_input):
        averages = np.zeros(len(vision_input))
        for i, ommatidia_readings in enumerate(vision_input):
            averages[i] = np.mean(ommatidia_readings)
        return averages

    def _find_stumbling_sensor_indices(self):
        stumbling_sensors = {leg: [] for leg in self.preprogrammed_steps.legs}
        for i, sensor_name in enumerate(self.contact_sensor_placements):
            leg = sensor_name.split("/")[1][:2]  # sensor_name: e.g. "Animat/LFTarsus1"
            segment = sensor_name.split("/")[1][2:]
            if segment in self.stumble_segments:
                stumbling_sensors[leg].append(i)
        stumbling_sensors = {k: np.array(v) for k, v in stumbling_sensors.items()}
        if any(
            v.size != len(self.stumble_segments) for v in stumbling_sensors.values()
        ):
            raise RuntimeError(
                "Contact detection must be enabled for all tibia, tarsus1, and tarsus2 "
                "segments for stumbling detection."
            )
        return stumbling_sensors

    def _retraction_rule_find_leg(self, obs):
        """Returns the index of the leg that needs to be retracted, or None
        if none applies."""
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.05:
            leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]
        else:
            leg_to_correct_retraction = None
        return leg_to_correct_retraction

    def _stumbling_rule_check_condition(self, obs, leg):
        """Return True if the leg is stumbling, False otherwise."""
        # update stumbling correction amounts
        contact_forces = obs["contact_forces"][self.stumbling_sensors[leg], :]
        fly_orientation = obs["fly_orientation"]
        # force projection should be negative if against fly orientation
        force_proj = np.dot(contact_forces, fly_orientation)
        return (force_proj < self.stumbling_force_threshold).any()

    def _get_net_correction(self, retraction_correction, stumbling_correction):
        """Retraction correction has priority."""
        if retraction_correction > 0:
            return retraction_correction
        return stumbling_correction

    def _update_correction_amount(
        self, condition, curr_amount, correction_rates, viz_segment
    ):
        """Update correction amount and color code leg segment.

        Parameters
        ----------
        condition : bool
            Whether the correction condition is met.
        curr_amount : float
            Current correction amount.
        correction_rates : Tuple[float, float]
            Correction rates for increment and decrement.
        viz_segment : str
            Name of the segment to color code. If None, no color coding is
            done.

        Returns
        -------
        float
            Updated correction amount.
        """
        if condition:  # lift leg
            increment = correction_rates[0] * self.timestep
            new_amount = curr_amount + increment
            color = (0, 1, 0, 1)
        else:  # condition no longer met, lower leg
            decrement = correction_rates[1] * self.timestep
            new_amount = max(0, curr_amount - decrement)
            color = (1, 0, 0, 1)
        if viz_segment is not None:
            self.change_segment_color(viz_segment, color)
        return new_amount

    def reset(self, sim, seed=None, init_phases=None, init_magnitudes=None, **kwargs):
        obs, info = super().reset(sim, seed=seed, **kwargs)
        self.cpg_network.random_state = np.random.RandomState(seed)
        self.cpg_network.intrinsic_amps = self.intrinsic_amps
        self.cpg_network.intrinsic_freqs = self.intrinsic_freqs
        self.cpg_network.reset(init_phases, init_magnitudes)
        self.retraction_correction = np.zeros(6)
        self.stumbling_correction = np.zeros(6)
        return obs, info

    def pre_step(self, action, sim):
        """Step the simulation forward one timestep.

        Parameters
        ----------
        action : np.ndarray
            Array of shape (2,) containing descending signal encoding turning
        """
        physics = sim.physics

        start_lunging = action[2]
        action = action[:2]

        # Update the action based on starting lunging or not
        if not start_lunging:
            # update CPG parameters
            amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
            freqs = self.intrinsic_freqs.copy()
            freqs[:3] *= 1 if action[0] > 0 else -1
            freqs[3:] *= 1 if action[1] > 0 else -1
            self.cpg_network.intrinsic_amps = amps
            self.cpg_network.intrinsic_freqs = freqs

            # get current observation
            obs = super().get_observation(sim)

            # Retraction rule: is any leg stuck in a gap and needing to be retracted?
            leg_to_correct_retraction = self._retraction_rule_find_leg(obs)

            self.cpg_network.step()

            joints_angles = []
            adhesion_onoff = []
            for i, leg in enumerate(self.preprogrammed_steps.legs):
                # update retraction correction amounts
                self.retraction_correction[i] = self._update_correction_amount(
                    condition=(i == leg_to_correct_retraction),
                    curr_amount=self.retraction_correction[i],
                    correction_rates=self.correction_rates["retraction"],
                    viz_segment=f"{leg}Tibia" if self.draw_corrections else None,
                )
                # update stumbling correction amounts
                self.stumbling_correction[i] = self._update_correction_amount(
                    condition=self._stumbling_rule_check_condition(obs, leg),
                    curr_amount=self.stumbling_correction[i],
                    correction_rates=self.correction_rates["stumbling"],
                    viz_segment=f"{leg}Femur" if self.draw_corrections else None,
                )
                # get net correction amount
                net_correction = self._get_net_correction(
                    self.retraction_correction[i], self.stumbling_correction[i]
                )

                # get target angles from CPGs and apply correction
                my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                    leg,
                    self.cpg_network.curr_phases[i],
                    self.cpg_network.curr_magnitudes[i],
                )
                my_joints_angles += net_correction * self.correction_vectors[leg[1]]
                joints_angles.append(my_joints_angles)

                # get adhesion on/off signal
                my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                    leg, self.cpg_network.curr_phases[i]
                )
                adhesion_onoff.append(my_adhesion_onoff)

            action = {
                "joints": np.array(np.concatenate(joints_angles)),
                "adhesion": np.array(adhesion_onoff).astype(int),
            }
        else:
            self.joint_pos_steady = self.data_block1[:, self.i_lunging]
            action = {"joints": self.data[:, self.i_lunging], "adhesion": np.zeros(6)}
            sim.render()
            self.i_lunging += 1

        return super().pre_step(action, sim)

    @staticmethod
    def calc_ipsilateral_speed(deviation, is_found):
        if not is_found:
            return 1.0
        else:
            return np.clip(1 - deviation * 3, 0.4, 1.2)

    def process_visual_observation(self, vision_input):
        features = np.zeros((2, 3))
        for i, ommatidia_readings in enumerate(vision_input):
            is_obj = ommatidia_readings. max(axis=1) < self.obj_threshold
            is_obj_coords = self.coms[is_obj]
            if is_obj_coords.shape[0] > 0:
                features[i, :2] = is_obj_coords.mean(axis=0)
            features[i, 2] = is_obj_coords.shape[0]
        features[:, 0] /= self.retina.nrows  # normalize y_center
        features[:, 1] /= self.retina.ncols  # normalize x_center
        features[:, 2] /= self.retina.num_ommatidia_per_eye  # normalize area
        return features.ravel().astype("float32")
