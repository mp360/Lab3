from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np


def motion_update(particles, odom):
    """ Particle filter motion update
        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*
        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """


    motion_particles = []

    for particle in particles:
        rx, ry, rh = add_odometry_noise(odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA)
        xr,yr = rotate_point(rx, ry, particle.h)
        x = particle.x + xr
        y = particle.y + yr
        h = particle.h + rh
        motion_particles.append(Particle(x, y, h))
    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update
        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)
        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree
                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one
        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles
        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = []
    part = np.array([1.0] * len(particles))

    for i, particle in enumerate(particles):
        particleMarkers = particle.read_markers(grid)
        if len(measured_marker_list) == 0 and len(particleMarkers) == 0:
            part[i] = 1
        elif len(measured_marker_list) == 0 or len(particleMarkers) == 0:
            part[i] = DETECTION_FAILURE_RATE * SPURIOUS_DETECTION_RATE
        else:
            for markers in measured_marker_list:
                distance = 0
                angle = 0

                for marker in particleMarkers:
                    marker0 = marker[0]
                    marker1 = marker[1]
                    marker2 = marker[2]
                    dist = grid_distance(markers[0], markers[1], marker0, marker1)

                    if not distance or distance > dist:
                        distance = dist
                        angle = diff_heading_deg(markers[2], marker2)

                power = - (distance**2)/(2*(MARKER_TRANS_SIGMA**2))
                power -= (angle**2)/(2*(MARKER_ROT_SIGMA**2))
                part[i] *= np.exp(power)

        lengthParticle = max(0, len(particleMarkers) - len(measured_marker_list))
        lengthList = max(0, len(measured_marker_list) - len(particleMarkers))
        part[i] = max(part[i], DETECTION_FAILURE_RATE*SPURIOUS_DETECTION_RATE)
        part[i] *= (DETECTION_FAILURE_RATE ** lengthParticle)
        part[i] *= (SPURIOUS_DETECTION_RATE ** lengthList)

    part = part / sum(part)
    random = .01 * len(particles)
    random = int(np.rint(random))
    indexes = np.random.choice(a=range(0, len(particles)), size=(len(particles) - random),
                               replace=True, p=part).tolist()
    measured_particles[0:random] = Particle.create_random(random, grid)
    measured_particles[random+1:-1] = [particles[i] for i in indexes]

    return measured_particles