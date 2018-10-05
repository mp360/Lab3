#! python3
#   AUTHORS: Manan Patel, Parrish McCall
from grid import *
from utils import *
from setting import *

from particle import Particle
import numpy as np


def motion_update(particles, odom):

    """ Particle filter motion update
        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*
        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
        references: https://github.com/mxie33/CS3630-robotics/tree/1c0d8c1e92d81e7f74f1930b9d2c64a35aab0062
    """


    motion_particles = []
    for prt in particles:
        odomGaussNoiseApplied =  add_odometry_noise(odom,ODOM_HEAD_SIGMA,ODOM_TRANS_SIGMA)
        dx = odomGaussNoiseApplied[0]
        dy = odomGaussNoiseApplied[1]
        dh = odomGaussNoiseApplied[2]
        xRotatedH,yRotatedH = rotate_point(dx,dy,prt.h)
        newY = prt.y + yRotatedH
        newX = prt.x + xRotatedH
        newH = prt.h + dh
        motion_particles.append(Particle(newX,newY,newH))
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
        references: https://github.com/mxie33/CS3630-robotics/tree/1c0d8c1e92d81e7f74f1930b9d2c64a35aab0062

    """
    measured_particles = []
    probSDFailure = SPURIOUS_DETECTION_RATE* DETECTION_FAILURE_RATE
    probFunc = np.array([], dtype=float)
    for i in range(len(particles)):
        probFunc = np.append(probFunc, 1.0)
    # print(len(probFunc))
    # print(len(particles))

    for i,particle in enumerate(particles):
        curXCoord = particle.x
        curYCoord = particle.y
        if grid.is_in(curXCoord,curYCoord):
            parmarkers = particle.read_markers(grid)

            falsePos = max(abs(0),
                abs(len(parmarkers))-len(measured_marker_list))

            lenMeasuredList = len(measured_marker_list)
            lenParMarkerList = len(parmarkers)

            falseNegs = max(abs(0),
                abs(len(measured_marker_list))-len(parmarkers))

            if not lenMeasuredList == 0 and not lenParMarkerList == 0:
                for realMarker in measured_marker_list:
                    angleDiff = None
                    minDist = math.inf
                    for predictedMarker in parmarkers:
                        eucDist = grid_distance(realMarker[0],realMarker[1],
                            predictedMarker[0],predictedMarker[1])

                        if minDist > eucDist:
                            minDist = eucDist
                            angleDiff = abs(diff_heading_deg(realMarker[2],predictedMarker[2]))

                    if angleDiff is not None:
                        gaussPower = -1 *((minDist**2)/(2*(MARKER_TRANS_SIGMA**2)) +
                            (angleDiff**2)/(2*(MARKER_ROT_SIGMA**2)))
                        probFunc[i] = abs(probFunc[i]) * np.exp(gaussPower)
            elif lenMeasuredList == 0 and lenParMarkerList==0:
                j = 4 #DO NOTHING (keep probFnc[i] = 1)
            elif lenMeasuredList==0 or lenParMarkerList==0:
                probFunc[i] *= SPURIOUS_DETECTION_RATE * DETECTION_FAILURE_RATE
            errorCoefficient = (DETECTION_FAILURE_RATE**falsePos) * (SPURIOUS_DETECTION_RATE**falseNegs)
            probFunc[i] = errorCoefficient * max(probFunc[i], probSDFailure)
        else:
            probFunc[i] = 0
    if sum(probFunc) is not 0:
        probFunc = np.true_divide(probFunc, sum(probFunc))

    resamplePercent = 0.014
    resampleThreshold = int(np.rint(len(particles) * resamplePercent))
    partIndices = list(np.random.choice(a=range(abs(len(particles))),
        size=abs(len(particles) - resampleThreshold),replace=True,p=probFunc))

    # print(len(range(randThreshold)))
    half1 = Particle.create_random(resampleThreshold, grid)

    for i in range(resampleThreshold):
        measured_particles += [half1[i]]

    for i in partIndices:
        measured_particles += [particles[i]]

    return measured_particles

