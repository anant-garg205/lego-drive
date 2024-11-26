'''
    Date: January 14, 2024
	Description: Script to measure the metrics .
'''

import numpy as np

class Metrics:

	def __init__(self):
		pass
	

	def calc_l2_dist(self, pt1, pt2):
		dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
		return dist


	def find_closest_point_with_traj(self, point, traj):
		dist_list = [self.calc_l2_dist(point, traj_point) for traj_point in traj]
		return min(dist_list)


	def calc_minADE(self, optim_trajs, non_optim_trajs):
        ''' 
        Function to calculate the average displacement error between
        optimal and not-optimal trajectory
        '''
		def calc_ADE(optim_traj, non_optim_traj):
			displacement = 0
			for point in non_optim_traj:
				min_dist = self.find_closest_point_with_traj(point, optim_traj)
				displacement += min_dist
			ADE = displacement / non_optim_traj.shape[0]
			return ADE
		
		ADE_list = np.array([calc_ADE(optim_traj, non_optim_traj) for optim_traj, non_optim_traj in zip(optim_trajs, non_optim_trajs)])
		return min(ADE_list), np.mean(ADE_list)


	def calc_minFDE(self, optim_trajs, non_optim_trajs):
		'''	
		Function to calculate the Final Displacement Error (FDE)
		defined by L2 dist. between final trajectory point and predicted goal
		'''
		traj_last_locations = [nt[-1] for nt in non_optim_trajs]
		final_locations = [ot[-1] for ot in optim_trajs]
		dist_list = np.array([self.calc_l2_dist(final_location, t_location) for final_location, t_location in zip(final_locations, traj_last_locations)])
		return min(dist_list), np.mean(dist_list)

	
	def calc_success_rate(self, optim_trajs, non_optim_trajs, threshold=2):
		'''
		Function to calculate the success rate of the trajectory
		defined by the final point within 2m of the predicted goal point
		'''
		traj_last_locations = [nt[-1] for nt in non_optim_trajs]
		final_locations = [ot[-1] for ot in optim_trajs]
		dist_list = np.array([self.calc_l2_dist(final_location, t_location) for final_location, t_location in zip(final_locations, traj_last_locations)])
		miss_num = dist_list[dist_list<threshold]
		return len(miss_num) / len(traj_last_locations)


	def calc_lane_cte(self, goal, lane_center):
		'''
		Function to calculate cross track error of the trajectory
		from the ego-lane center
		'''
		min_dist = self.find_closest_point_with_traj(goal, lane_center)
		min_index = lane_center.index(min_dist)
		angle = np.arctan2(lane_center[min_index][1] - goal[1], lane_center[min_index][0] - goal[0])
		lane_cte = min_dist * max(np.sin(angle), np.cos(angle))
		return lane_cte


	def calc_ROC(self, trajectories):
		'''
		Function to calcuate the radius of curvature of the trajectories
		'''
		curvatures = []
		for trajectory in trajectories:
			dx = np.gradient(trajectory[:, 0])
			dy = np.gradient(trajectory[:, 1])
			
			d2x = np.gradient(dx)
			d2y = np.gradient(dy)

			numerator = np.abs(dx * d2y - dy * d2x)
			denominator = (dx**2 + dy**2)**(3/2)

			curvature = numerator / denominator
			curvatures.append(np.mean(curvature))
			
		return np.mean(np.array(curvatures)), np.max(np.array(curvatures))