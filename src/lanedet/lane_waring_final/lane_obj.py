import numpy as np
import global_config

CFG = global_config.cfg

class Lane:
    def __init__(self, laneIdx):
        self.age = 0
        self.points = np.zeros([12,2], dtype = int)
        self.image_X = 1920
        self.laneIdx = laneIdx
        self.isInit = False
        self.lostCnt = 0

        self.detectedLeftLane = np.zeros([12,2], dtype = int)
        self.detectedRightLane = np.zeros([12,2], dtype = int)

        self.detectedLostCnt = 0

    def reset(self):
        self.points = np.zeros([12,2], dtype = int)
        self.isInit = False
        self.lostCnt = 0
        self.age = 0

        print('\n{} lane reset'.format(self.laneIdx))
        return True

    def isLost(self):
        self.lostCnt = self.lostCnt+1
        if self.age > 30:
            if self.lostCnt > 12:
                return self.reset()
        elif self.age > 10:
            if self.lostCnt > 8:
                return self.reset()
        elif self.lostCnt > 5:
                return self.reset()
        self.age = self.age+1
        return  False

    def findDetectedLeftRightLane(self, lane_x_coords):
        self.detectedLeftLane = np.zeros([12,2], dtype = int)
        self.detectedRightLane = np.zeros([12,2], dtype = int)

        if len(lane_x_coords) >= 2:
            sorted_lane_x_coords = sorted(lane_x_coords[:2],key=(lambda x : x[5]))

            self.detectedLeftLane[:,0] = sorted_lane_x_coords[0]
            self.detectedRightLane[:,0] = sorted_lane_x_coords[1]
            self.detectedLeftLane[:,1] = self.detectedRightLane[:,1] = np.linspace(CFG.LANE_START_Y, CFG.LANE_END_Y, 12)

        else:
            if lane_x_coords[0][0] < self.image_X/2:
                self.detectedLeftLane[:,0] = lane_x_coords[0]
                self.detectedLeftLane[:,1] = np.linspace(CFG.LANE_START_Y, CFG.LANE_END_Y, 12)
            else:
                self.detectedRightLane[:,0] = lane_x_coords[0]
                self.detectedRightLane[:,1] = np.linspace(CFG.LANE_START_Y, CFG.LANE_END_Y, 12)

    def initLane(self):
        if self.laneIdx == 'left' and not (self.detectedLeftLane == 0).all():
            self.points = self.detectedLeftLane.copy()
        elif self.laneIdx == 'right' and not (self.detectedRightLane == 0).all():
            self.points = self.detectedRightLane.copy()
        else:
            return

        # self.points = self.detectedLeftLane.copy()  if self.laneIdx == 'left' else self.detectedRightLane.copy()
        self.isInit = True
        self.age = self.age+1

    def updateLane(self, fit_params):
        if len(fit_params) == 0:
            self.isLost()
            return

        plot_y = np.linspace(CFG.LANE_START_Y, CFG.LANE_END_Y, 12)
        lane_x_coords = []
        # detectedLeftLaneX = [0] * 12
        # detectedRightLaneX = [0] * 12

        for fit_param in fit_params:
            fitx = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            # if abs(fitx[5] - self.image_X/2) > 550:
            #     continue
            lane_x_coords.append(fitx.astype(np.int))

        lane_x_coords.sort(key=(lambda x : abs(x[5] - self.image_X/2)))
        self.findDetectedLeftRightLane(lane_x_coords)
                
        if not self.isInit:
            self.initLane()
        else:
            detectedPoints = self.points.copy()
            minDist = 500
            bestFit = -1
            for idx in range(len(lane_x_coords)):
                # if lane_x_coords[idx][0] == (self.detectedRightLane[0][0] if self.laneIdx == 'left' else self.detectedLeftLane[0][0]):
                #     continue
                detectedPoints[:,0] = lane_x_coords[idx]
                dist = np.linalg.norm(self.points - detectedPoints)
                print('{}, dist:{}'.format(self.laneIdx,dist))
                if dist < minDist:
                    bestFit = idx
                    minDist = dist

            if bestFit == -1:
                if self.isLost():
                    self.initLane()
            else:
                # if self.age < 10 and lane_x_coords[bestFit][0] != (detectedLeftLane[0][0] if self.laneIdx == 'left' else detectedRightLane[0][0]):
                #     self.initLane(detectedLeftLane, detectedRightLane)
                # else:
                #     self.lostCnt = 0
                #     self.age = self.age+1
                #     detectedPoints[:,0] = lane_x_coords[bestFit]
                #     self.points[:,0] = np.average([self.points[:,0], detectedPoints[:,0]], axis=0, weights=[0.8,0.2])
                self.lostCnt = 0
                self.age = self.age+1
                detectedPoints[:,0] = lane_x_coords[bestFit]
                self.points[:,0] = np.average([self.points[:,0], detectedPoints[:,0]], axis=0, weights=[0.9,0.1])

                if int(lane_x_coords[bestFit][0]) == (self.detectedLeftLane[0][0] if self.laneIdx == 'left' else self.detectedRightLane[0][0]):
                    self.detectedLostCnt = 0
                else:
                    self.detectedLostCnt = self.detectedLostCnt + 1

                if self.detectedLostCnt > 5:
                    self.initLane()

        print('\n{} lane, age:{}, lost cnt:{}, detectedLostCnt:{}'.format(self.laneIdx, self.age, self.lostCnt, self.detectedLostCnt))
