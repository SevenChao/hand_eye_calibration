import os
import sys 

import numpy as np
from dataset_store import Dataset
from transformation_util2 import RT2Transmat,ENUTransformer,Transmat2RT
from scipy.spatial.transform import Rotation as R
from tsmap2 import TSMap, GNSSTransformer
from quaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PoseExtractor(object):
    def __init__(self,bag_name, topic,ts_begin, ts_end, out_path):
        self.bag_name = bag_name
        self.topic = topic
        self.ts_begin = ts_begin
        self.ts_end = ts_end
        self.path = list()

        #open dataset
        try:
            self.ds = Dataset.open(self.bag_name)
        except Exception as e:
            print("Cannot open bag: {}".format(self.bag))
            raise Exception(e)

        #get bag information
        self.vehicle,self.map_name = self._get_info()
        self.map_base = self._get_map_base()

        #set output path
        self.out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path, self.vehicle,self.bag_name)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

    def process(self):
        raise NotImplementedError("Must override process")

    def _get_info(self):
        try:
            vehicle = self.ds.meta.get('vehicle').get('name').encode('ascii')
            map_name = os.path.splitext(self.ds.meta.get('map_file').encode('ascii'))[0]
        except Exception as e:
            print('Cannot extract bag information')
            raise Exception(e)
        return [vehicle, map_name]

    def _get_map_base(self):
        # set map prefix path
        pre_map_path = '/mnt/truenas/scratch/maps/map/'
        map_path = os.path.join(
            pre_map_path, self.map_name, self.map_name+".tsmap")
        try:
            map_handler = TSMap(map_path)
            map_base = GNSSTransformer.get_instance().get_base()
        except Exception as e:
            print ('\nNo such map found: {}'.format(map_path))
            raise Exception(e)
        return map_base 

class CamPoseExtractor(PoseExtractor):
    def __init__(self, bag_name,topic,ts_begin, ts_end, out_path):
        super(CamPoseExtractor,self).__init__(bag_name, topic,ts_begin, ts_end, out_path)
    
    def process(self):
        csv_file_name = os.path.join(self.out_path,"cam{}_pose.csv".format(self.topic[-1]))
        with open(csv_file_name, 'w') as f:
            for ts, msg in self.ds.fetch(self.topic,ts_begin=self.ts_begin,ts_end = self.ts_end):
                time_stamp = msg.header.stamp.to_nsec()/1e9
                transf_q = self._campose_to_quat(msg)
                pose = [time_stamp]+transf_q
                self.path.append(transf_q[:3])
                f.write(
                str(pose[0]) + ', ' +
                str(pose[1]) + ', ' +
                str(pose[2]) + ', ' +
                str(pose[3]) + ', ' +
                str(pose[4]) + ', ' +
                str(pose[5]) + ', ' +
                str(pose[6]) + ', ' +
                str(pose[7]) + '\n'
            )
    def _campose_to_quat(self,cam_pose):
        enu2cam = np.array(cam_pose.mat_enu2cam).reshape(4, 4)
        cam2enu = np.linalg.inv(enu2cam)
        r,t= Transmat2RT(cam2enu) 
        # print("cam:",[t,r])
        #compose RT from matrix
        # r,t = Transmat2RT(cam2enu)
        q = Quaternion.from_rotation_matrix(cam2enu)
        # rot = R.from_euler('xyz',np.asarray(r))
        # q = rot.as_quat()
        transf_q = list(t)+[q.x,q.y,q.z,q.w]
        return transf_q


class IMUPoseExtractor(PoseExtractor):

    def __init__(self, bag_name,topic,ts_begin, ts_end, out_path):
        super(IMUPoseExtractor,self).__init__(bag_name, topic,ts_begin, ts_end, out_path)
        self.base_gnss_trans = ENUTransformer()
        self.base_gnss_trans.SetBase(list(self.map_base))
        self.base_North_GPS = np.array(self.base_gnss_trans.Global2LLH(np.array([[0., 0.], [0., 1.]])))
        self.vehicle_gnss_trans = ENUTransformer()
    
    def process(self):
        csv_file_name = os.path.join(self.out_path,"imu_pose.csv")
        with open(csv_file_name, 'w') as f:
            for ts,msg in self.ds.fetch(self.topic,ts_begin=ts_begin,ts_end=ts_end):
                time_stamp = msg.header2.stamp.to_nsec()/1e9
                enu = self.gps2enu(msg)
                transf_q = self.rotation_to_quaternion(enu)
                pose = [time_stamp] + transf_q
                print("imu: ",pose)
                self.path.append(transf_q[:3])
                f.write(
                    str(pose[0]) + ', ' +
                    str(pose[1]) + ', ' +
                    str(pose[2]) + ', ' +
                    str(pose[3]) + ', ' +
                    str(pose[4]) + ', ' +
                    str(pose[5]) + ', ' +
                    str(pose[6]) + ', ' +
                    str(pose[7]) + '\n'
                )
        
        

    def gps2enu(self, gps):
        try:
            assert gps is not None
        except AssertionError:
            print("Error: Cannot found gps")
            return np.zeros((2, 3)), np.zeros((2, 3))
        
        gps_position = np.array([gps.latitude, gps.longitude, gps.altitude])
        x, y, z = self.base_gnss_trans.LLH2Global(gps_position)
        
        #TODO
        #update altitude
        roll, pitch = np.deg2rad(gps.roll), np.deg2rad(gps.pitch)
        yaw = self._convert_azimuth_to_yaw(gps_position, gps.azimuth)
        enu_pose = [[x, y, z], [roll, pitch, yaw]]
        enu_mat = RT2Transmat(enu_pose[1],enu_pose[0])
        # print ("imu: ",enu_mat)
        return enu_mat

    def _convert_azimuth_to_yaw(self, gps_position, azimuth, correction=False):
        """
        convert GPS azimuth to yaw
        :param azimuth:
        :return:
        """
        if correction:
            angle = self.compute_yaw_diff_btw_map_local(gps_position)
        else:
            angle = 0
        return -(np.deg2rad(azimuth) - angle)

    def compute_yaw_diff_btw_map_local(self,gps_position):
        """
        convert map based yaw to be local yaw
        """
        self.vehicle_gnss_trans.SetBase(list(gps_position))
        pts_enu = np.array(self.vehicle_gnss_trans.LLH2Global(self.base_North_GPS))
        pts_enu = (pts_enu - pts_enu[0])[1]
        angle = np.arctan2(pts_enu[0], pts_enu[1])
        return angle
    
    def rotation_to_quaternion(self, enu_pose):
        """
        convert enu_pose to [x, y, z, q_x, q_y, q_z, q_w]
        """
        q= Quaternion.from_rotation_matrix(enu_pose)
        # rot = R.from_euler('xyz',np.asarray(enu_pose[1]))
        # q = rot.as_quat()
        trans_q = list(enu_pose[:3,3])+[q.x,q.y,q.z,q.w]
        return trans_q



if __name__ == "__main__":
    bag_name = "2020-01-10-10-52-55"
    topics= ["/lps_cam_pose/camera1","/novatel_data/inspvax"]
    ts_begin = "00:02:19"
    # ts_end = 1577412002.6*1e9
    ts_end = "00:06:21"
    out_path = "../../"
    for topic in topics:
        if "lps_cam_pos" in topic:
            cam_pose_extractor = CamPoseExtractor(bag_name,topic,ts_begin,ts_end,out_path)
            cam_pose_extractor.process()
        if "inspvax" in topic:
            imu_pose_extractor = IMUPoseExtractor(bag_name,topic,ts_begin,ts_end,out_path)
            imu_pose_extractor.process()
    
    #plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    x= np.array(cam_pose_extractor.path)[:,0]
    y= np.array(cam_pose_extractor.path)[:,1]
    z= np.array(cam_pose_extractor.path)[:,2]
    ax.plot3D(x,y,z,'red')

    i_x = np.array(imu_pose_extractor.path)[:,0]
    i_y = np.array(imu_pose_extractor.path)[:,1]
    i_z = np.array(imu_pose_extractor.path)[:,2]

    ax.plot3D(i_x,i_y,i_z)
    plt.show()
    
    
    
    #fetch campose

    

