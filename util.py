
import numpy as np
import os
import time
import pandas as pd

from murty import *
from MSC import *
from BTP import *
from plotting import *

import pdb

def Embryo_graph(n):
    
    adj = np.zeros((n,n),'int')
    
    for i in range(n):
        for j in range(n):

            if i-1 == j:
                adj[i,j]=1

            
            if i-2==j:
                adj[i,j]=1
                
            if i%2:
                if i-3==j:
                    adj[i,j]=1
                
    adj[-1,-2]=1
    
    edge_list = np.transpose(np.nonzero(adj))
                
    return adj, edge_list
    
def Embryo(in_arr, out_arr, edge_list):
    
    m = edge_list.shape[0]
    starts = edge_list[:,0]
    ends = edge_list[:,1]

    in_starts = in_arr[starts]
    in_ends = in_arr[ends]

    out_starts = out_arr[starts]
    out_ends = out_arr[ends]

    in_gaps = in_ends - in_starts
    in_lengths = np.linalg.norm(in_gaps, axis=1)

    out_gaps = out_ends - out_starts
    out_lengths = np.linalg.norm(out_gaps, axis=1)

    cost = np.abs(out_lengths - in_lengths).sum()

    return cost

def Posture(in_arr, out_arr, edge_list, inv_cov):

    m = edge_list.shape[0]
    starts = edge_list[:,0]
    ends = edge_list[:,1]

    in_starts = in_arr[starts]
    in_ends = in_arr[ends]

    out_starts = out_arr[starts]
    out_ends = out_arr[ends]

    in_gaps = in_ends - in_starts
    in_lengths = np.linalg.norm(in_gaps, axis=1)

    out_gaps = out_ends - out_starts
    out_lengths = np.linalg.norm(out_gaps, axis=1)

    diffs = out_lengths - in_lengths

    cost_sq = np.linalg.multi_dot([diffs.T, inv_cov, diffs])
    cost = cost_sq**.5

    return cost

def Movement(in_arr, out_arr, inv_cov):

    # d1 and d0 are ordered. 
    diffs = out_arr - in_arr
    diffs_vec = diffs.flatten()

    cost_sq = np.linalg.multi_dot([diffs_vec.T, inv_cov, diffs_vec])
    cost = cost_sq**.5

    return cost

def PostureMovement(in_arr, out_arr, edge_list, inv_cov1, inv_cov2):

    posture = Posture(in_arr, out_arr, edge_list, inv_cov1)
    movement = Movement(in_arr, out_arr, inv_cov2)

    posture_movement = posture + movement

    return posture_movement

def Intersection_test(data, skinnyFactor=0.55):
    
    numCells = 18
    numSegs = int(np.floor(numCells/2) - 1)
    origALL = np.empty((8*numSegs**2, 3))
    vecDirALL = np.empty((8*numSegs**2, 3))
    relComps = np.transpose(np.abs(np.diff(np.array([np.repeat(np.arange(numSegs), 8*numSegs), np.tile(np.repeat(np.repeat(np.arange(numSegs), 1), numSegs), numSegs)]), axis=0))>=2)[:,0]
    
    coords = data[:numCells].copy()
            
    coordsL = coords[::2]
    coordsR = coords[1::2]
    
    vertCoordsL = coordsL - (coordsL - coordsR)*skinnyFactor / 2
    vertCoordsR = coordsR - (coordsR - coordsL)*skinnyFactor / 2
    vertCoordsLR = np.empty((numCells, 3))
    vertCoordsLR[::2] = vertCoordsL
    vertCoordsLR[1::2] = vertCoordsR
    
    vert0Seq = np.repeat(vertCoordsL[:numSegs], numSegs, axis=0)
    vert1Seq = np.repeat(vertCoordsR[:numSegs], numSegs, axis=0)
    vert2Seq = np.repeat(vertCoordsLR[2:],numSegs/2, axis=0)
    
    vecDirLR = coordsR-coordsL
    vecDirAPL = np.diff(coordsL, axis=0)
    vecDirAPR = np.diff(coordsR, axis=0)
    
    origALL[::2, :] = np.repeat(coordsL[:numSegs, :], 8*numSegs/2, axis=0)
    origALL[1::4, :] = np.repeat(coordsR[:numSegs, :], 8*numSegs/4, axis=0)
    origALL[3::4, :] = np.repeat(coordsL[1:numSegs+1, :], 8*numSegs/4, axis=0)
    
    vecDirALL[::4, :] = np.repeat(vecDirAPL[:numSegs, :], 8*numSegs/4, axis=0)
    vecDirALL[1::4, :] = np.repeat(vecDirAPR[:numSegs, :], 8*numSegs/4, axis=0)
    vecDirALL[2::4, :] = np.repeat(vecDirLR[:numSegs, :], 8*numSegs/4, axis=0)
    vecDirALL[3::4, :] = np.repeat(vecDirLR[1:numSegs+1, :], 8*numSegs/4, axis=0)
    
    vert0ALL = np.tile(vert0Seq, (numSegs, 1))
    vert1ALL = np.tile(vert1Seq, (numSegs, 1))
    vert2ALL = np.tile(vert2Seq, (numSegs, 1))
    
    orig = origALL[relComps,:]
    vecDir = vecDirALL[relComps,:]
    vert0 = vert0ALL[relComps,:]
    vert1 = vert1ALL[relComps,:]
    vert2 = vert2ALL[relComps,:]
        
    edge1 = vert1-vert0
    edge2 = vert2-vert0
    tve = orig-vert0       
    pve = np.cross(vecDir, edge2, axis=1)
    
    det1 = np.sum(edge1*pve,axis=1)
    angOK = np.abs(det1)>0
    
    det1[np.invert(angOK)] = np.inf
                
    u = np.sum(tve*pve, axis=1)/det1
    v = np.inf + np.zeros(u.shape) 
    t = v.copy()
    ok = np.logical_and.reduce((angOK, u>=0, u<=1.0))
                
    if np.invert(np.any(ok)):
        intersections = ok
    
    else:
        qve = np.cross(tve[ok,:], edge1[ok,:], axis=1)
        v[ok] = np.sum(vecDir[ok,:]*qve, axis=1)/det1[ok]
        t[ok] = np.sum(edge2[ok,:]*qve, axis=1)/det1[ok]
        ok = np.logical_and.reduce((ok, v >= 0, u+v <= 1.0))
        intersections = np.logical_and.reduce((ok, t >= 0, t <= 1.0))

    int_cost = 1e6*intersections.sum()

    return int_cost

class MHHT:

    def __init__(self, config):

        # Read in config
        self.Dataset = config['Dataset'] 
        self.Q = config['Q'] 
        self.Detection = config['Detection'] 
        self.Interpolation = config['Interpolation'] 
        self.Interpolation_cost = config['Interpolation_cost'] 
        self.cost_threshold = config['Cost_threshold']
        self.print_interval = config['Print_interval']
        self.d = config['d'] 
        self.Model = config['Model'] 
        self.K = config['K']
        self.N = config['N']
        self.Solver = config['Solver'] 
        self.StartFrame = config['StartFrame'] 
        self.EndFrame = config['EndFrame'] 
        self.InitialFrame = config['InitialFrame']
        self.CurrentFrame = self.StartFrame

        self.out_path = os.path.join(config['Home'], 'Results', self.Dataset)
        self.track_path = os.path.join(self.out_path, 'output')

        os.mkdir(self.out_path) if not os.path.exists(self.out_path) else None

        self.output_name = 'Tracks_' + str(self.StartFrame) + '_' + str(self.EndFrame) + '_' + self.Model + '_' + \
                        str(self.K) + '_' + str(self.N) + '_' + self.Detection + '_' + self.Interpolation + '_' + \
                        str(self.d) + '.npy'
        
        self.n = 21 if self.Q else 19

        # Build graph:
        self.adj, self.edge_list = Embryo_graph(self.n)

        # Initialize arrays
        self.Annotations = np.load(config['Annotation_path'], allow_pickle=True)
        self.Predictions = np.load(config['Prediction_path'], allow_pickle=True)
        self.Tracks = np.zeros((self.EndFrame - self.StartFrame + 1, self.n, 3))
        self.Tracks[0] = self.Annotations[self.StartFrame - self.InitialFrame]
        self.Costs = np.zeros(self.EndFrame - self.StartFrame + 1)

        self.Posture_weights = np.load(config['Posture_weights_path']) 
        self.Movement_weights = np.load(config['Movement_weights_path'])

    def get_unary_cost(self, out_array_interp, C, out_perm):

        costs = C[np.arange(self.n),out_perm]
        costs[costs==self.d] = self.Interpolation_cost

        intersection_cost = Intersection_test(out_array_interp)
        
        unary_cost = costs.sum() + intersection_cost

        return unary_cost

    def get_total_assignment_cost(self, j, C, out_perm):
        
        edge_list = self.edge_list

        in_arr = self.inter_arrays[j]
        out_arr = self.inter_arrays[j+1]

        unary_cost = self.get_unary_cost(out_arr, C, out_perm)

        if self.Model == 'GNN' or self.Model == 'MHT':
            
            total_assignment_cost = unary_cost

        elif self.Model == 'Embryo':
            
            graphical_cost = Embryo(in_arr, out_arr, edge_list)
            
            total_assignment_cost = unary_cost + graphical_cost

        elif self.Model == 'Movement':
            
            graphical_cost = Movement(in_arr, out_arr, self.Movement_weights)
            
            total_assignment_cost = unary_cost + graphical_cost
            
        elif self.Model == 'Posture':
            
            graphical_cost = Posture(in_arr, out_arr, edge_list, self.Posture_weights)
            
            total_assignment_cost = unary_cost + graphical_cost
            
        elif self.Model == 'Posture-Movement':
            
            graphical_cost = PostureMovement(in_arr, out_arr, edge_list, self.Posture_weights, self.Movement_weights)
            
            total_assignment_cost = unary_cost + graphical_cost
                    
        return total_assignment_cost

    def update(self):

        j = 0

        # These will update
        self.n_frames_remaining = self.EndFrame - self.CurrentFrame
        self.N_scan = min([self.N, self.n_frames_remaining])

        self.path = np.zeros((self.N_scan+1, self.n, 3))

        self.final_arrays = np.zeros((self.K, self.n, 3))
        self.inter_arrays = np.zeros((self.N_scan+1, self.n, 3))
        self.inter_arrays[0] = self.Tracks[self.CurrentFrame - self.StartFrame]

        self.final_path = []
        self.cost = []
        self.order = []
        self.final_order = np.zeros(self.n)
        self.best_cost = 1e8

        self.search(j)

        return self.final_arrays[self.final_order[0]], self.best_cost  

    def track(self):

        for t in range(self.StartFrame, self.EndFrame):

            st = time.time()

            tracks, cost = self.update()

            self.CurrentFrame += 1
            self.Tracks[t+1 - self.StartFrame] = tracks
            self.Costs[t+1 - self.StartFrame] = cost

            rt = time.time() - st

            print('Frame', self.CurrentFrame)
            print('Cost', cost)
            print('Runtime:', np.round(rt,2), 'seconds \n')

            np.save(os.path.join(self.out_path, self.output_name), self.Tracks)

    def track_notebook(self):

        print_interval = self.print_interval
        print_frames = np.arange(self.StartFrame + print_interval, self.EndFrame + print_interval, print_interval)

        for t in range(self.StartFrame, self.EndFrame):

            update_progress(self.StartFrame, t+1, self.EndFrame, label="Tracking {} of {}".format(t+1, self.EndFrame))

            st = time.time()

            tracks, cost = self.update()

            rt = time.time() - st

            print('Frame', self.CurrentFrame + 1)
            print('Cost', cost)
            print('Runtime:', np.round(rt,2), 'seconds \n')

            if cost >= self.cost_threshold or t+1 in print_frames:

                plot_3d_overlay(self.Tracks[t - self.StartFrame], tracks, errors={})
                tracks = correction(t+1, self.out_path, self.Tracks[t - self.StartFrame], tracks)

            self.CurrentFrame += 1
            self.Tracks[t+1 - self.StartFrame] = tracks
            self.Costs[t+1 - self.StartFrame] = cost

            # Write out tracks.csv
            this_track_path = os.path.join(self.track_path, 'tracks_' + str(t) + '.csv')
            tracks_df = pd.DataFrame(tracks, columns = ['x','y','z'])
            tracks_df.to_csv(this_track_path)

class MHHT_Murty(MHHT):

    def __init__(self, config):

        MHHT.__init__(self, config)

    def interpolate(self, in_array_aug, out_array, chosen_cols):

        adj = self.adj
        n2 = out_array.shape[0]

        found_nuclei = np.argwhere(chosen_cols < n2)[:,0] # of assigned points, which are actual nuclei
        missing_nuclei = np.argwhere(chosen_cols >= n2)[:,0] # of assigned points, which are missed nuclei
        num_missing = missing_nuclei.shape[0] 
        
        if self.Interpolation == 'Last':
            
            # Use last point. 
            interp_points = {z:in_array_aug[z] for z in missing_nuclei}
                  
        elif self.Interpolation == 'Graph':
            
            interp_points = {}

            for z in missing_nuclei:

                last_pos = in_array_aug[z]

                used_nuclei = np.concatenate([np.nonzero(adj[z,:])[0], np.nonzero(adj[:,z])[0]])
                used_found_nuclei = np.intersect1d(found_nuclei, used_nuclei)
                num_used_found_nuclei = used_found_nuclei.shape[0]

                if num_used_found_nuclei > 0:

                    preds = np.empty((num_used_found_nuclei, 3))

                    for y in range(num_used_found_nuclei):
                            
                        nuc = used_found_nuclei[y]

                        last_nuc = in_array_aug[nuc]
                        this_nuc = out_array[chosen_cols[nuc]]

                        this_pred = this_nuc - (last_nuc - last_pos)

                        preds[y] = this_pred

                    pred_pos = preds.mean(axis=0)
                    
                    interp_points[z] = pred_pos

                elif num_used_found_nuclei == 0:

                    interp_points[z] = in_array_aug[z]
        
        # Update out_array
        out_array_interp = np.empty((self.n, 3))
        for z in range(self.n):
            
            pred_col = chosen_cols[z]
            
            # if missing:
            if pred_col >= n2:
                
                # use interpolated. 
                out_array_interp[z] = interp_points[z]
            
            elif pred_col < n2:
                
                out_array_interp[z] = out_array[pred_col]

        return out_array_interp

    def search(self, j):

        in_array_aug = self.inter_arrays[j]
        out_array_aug = self.Predictions[self.CurrentFrame  - self.InitialFrame + j + 1] 

        C = Murty_mat(in_array_aug, out_array_aug, self.d, self.d)
        costs, rows_K, cols_K = Murty(C, self.K)

        hyps = get_unique_hyps(cols_K, self.n)
        num_hyps = hyps.shape[0]
        
        for k in range(num_hyps): 
        
            chosen_cols = hyps[k]

            out_array_interp = self.interpolate(in_array_aug, out_array_aug, chosen_cols)

            self.inter_arrays[j+1] = out_array_interp

            if j == 0:
                self.final_arrays[k] = out_array_interp

            this_step_cost = self.get_total_assignment_cost(j, C, chosen_cols)

            self.cost.append(this_step_cost)
            self.order.append(k)

            j += 1
            
            if j < self.N_scan and sum(self.cost) < self.best_cost:

                self.search(j)

            if j == self.N_scan and sum(self.cost) < self.best_cost:

                fc = sum(self.cost)
                
                if fc < self.best_cost:

                    self.best_cost = fc
                    self.final_order = self.order.copy()

            j+= -1
            self.cost.pop()
            self.order.pop()

class MHHT_MSC(MHHT):

    def __init__(self, config):

        MHHT.__init__(self, config)

    def interpolate(self, in_array_aug, out_array, chosen_cols):

        adj = self.adj
        n2 = out_array.shape[0]

        found_nuclei = np.argwhere(chosen_cols < n2)[:,0] # of assigned points, which are actual nuclei
        missing_nuclei = np.argwhere(chosen_cols >= n2)[:,0] # of assigned points, which are missed nuclei
        num_missing = missing_nuclei.shape[0] 
        
        if self.Interpolation == 'Last':
            
            # Use last point. 
            interp_points = {z:in_array_aug[z] for z in missing_nuclei}
                  
        elif self.Interpolation == 'Graph':
            
            interp_points = {}

            for z in missing_nuclei:

                last_pos = in_array_aug[z]

                used_nuclei = np.concatenate([np.nonzero(adj[z,:])[0], np.nonzero(adj[:,z])[0]])
                used_found_nuclei = np.intersect1d(found_nuclei, used_nuclei)
                num_used_found_nuclei = used_found_nuclei.shape[0]

                if num_used_found_nuclei > 0:

                    preds = np.empty((num_used_found_nuclei, 3))

                    for y in range(num_used_found_nuclei):
                            
                        nuc = used_found_nuclei[y]

                        last_nuc = in_array_aug[nuc]
                        this_nuc = out_array[chosen_cols[nuc]]

                        this_pred = this_nuc - (last_nuc - last_pos)

                        preds[y] = this_pred

                    pred_pos = preds.mean(axis=0)
                    
                    interp_points[z] = pred_pos

                elif num_used_found_nuclei == 0:

                    interp_points[z] = in_array_aug[z]
        
        # Update out_array
        out_array_interp = np.empty((self.n, 3))
        for z in range(self.n):
            
            pred_col = chosen_cols[z]
            
            # if missing:
            if pred_col >= n2:
                
                # use interpolated. 
                out_array_interp[z] = interp_points[z]
            
            elif pred_col < n2:
                
                out_array_interp[z] = out_array[pred_col]

        return out_array_interp

    def search(self, j):

        in_array_aug = self.inter_arrays[j]
        out_array_aug = self.Predictions[self.CurrentFrame  - self.InitialFrame + j + 1] 

        C = Murty_mat_MSC(in_array_aug, out_array_aug, self.d)
        costs, rows_K, cols_K = Murty_MSC(C, self.K)

        # hyps = get_unique_hyps(cols_K, self.n)
        # num_hyps = hyps.shape[0]
        
        for k in range(self.K): 
        
            chosen_cols = cols_K[k]

            out_array_interp = self.interpolate(in_array_aug, out_array_aug, chosen_cols)

            self.inter_arrays[j+1] = out_array_interp

            if j == 0:
                self.final_arrays[k] = out_array_interp

            this_step_cost = self.get_total_assignment_cost(j, C, chosen_cols)

            self.cost.append(this_step_cost)
            self.order.append(k)

            j += 1
            
            if j < self.N_scan and sum(self.cost) < self.best_cost:

                self.search(j)

            if j == self.N_scan and sum(self.cost) < self.best_cost:

                fc = sum(self.cost)
                
                if fc < self.best_cost:

                    self.best_cost = fc
                    self.final_order = self.order.copy()

            j+= -1
            self.cost.pop()
            self.order.pop()

class MHHT_BTP(MHHT):

    def __init__(self, config):

        MHHT.__init__(self, config)

    def interpolate(self, in_array_aug, out_array, chosen_cols):

        adj = self.adj
        n2 = out_array.shape[0]

        found_nuclei = np.argwhere(chosen_cols != 0)[:,0] # of assigned points, which are actual nuclei
        missing_nuclei = np.argwhere(chosen_cols == 0)[:,0] # of assigned points, which are missed nuclei
        
        # Set the values back 1 due to missing nucleus offset:
        chosen_cols_copy = chosen_cols.copy()
        chosen_cols_copy[found_nuclei] += -1
        
        if self.Interpolation == 'Last':
            
            # Use last point. 
            interp_points = {z:in_array_aug[z] for z in missing_nuclei}
                  
        elif self.Interpolation == 'Graph':
            
            interp_points = {}

            for z in missing_nuclei:

                last_pos = in_array_aug[z]

                used_nuclei = np.concatenate([np.nonzero(adj[z,:])[0], np.nonzero(adj[:,z])[0]])
                used_found_nuclei = np.intersect1d(found_nuclei, used_nuclei)
                num_used_found_nuclei = used_found_nuclei.shape[0]

                if num_used_found_nuclei > 0:

                    preds = np.empty((num_used_found_nuclei, 3))

                    for y in range(num_used_found_nuclei):
                            
                        nuc = used_found_nuclei[y]

                        last_nuc = in_array_aug[nuc]
                        this_nuc = out_array[chosen_cols_copy[nuc]]

                        this_pred = this_nuc - (last_nuc - last_pos)

                        preds[y] = this_pred

                    pred_pos = preds.mean(axis=0)
                    
                    interp_points[z] = pred_pos

                elif num_used_found_nuclei == 0:

                    interp_points[z] = in_array_aug[z]
        
        # Update out_array
        out_array_interp = np.empty((self.n, 3))
        for z in range(self.n):
   
            if z in found_nuclei:

                pred_pos = chosen_cols_copy[z]
                out_array_interp[z] = out_array[pred_pos]

            elif z in missing_nuclei:

                interp_pos = interp_points[z]
                out_array_interp[z] = interp_pos

        return out_array_interp

    def search(self, j):
                
        in_array_aug = self.inter_arrays[j]
        out_array_aug = self.Predictions[self.CurrentFrame - self.InitialFrame + j + 1] 

        n1 = self.n
        n2 = out_array_aug.shape[0]

        C, A, Aeq, b, beq = build_initial_all(in_array_aug, out_array_aug, self.d)

        x, xv = MBest(C, A, b, Aeq, beq, self.K)

        y = np.reshape(x, (n1, n2+1, self.K))
        y_out = np.argmax(y, axis=1)
        
        for k in range(self.K): 
        
            chosen_cols = y_out[:, k]
            
            out_array_interp = self.interpolate(in_array_aug, out_array_aug, chosen_cols)

            self.inter_arrays[j+1] = out_array_interp

            if j == 0:
                self.final_arrays[k] = out_array_interp

            C_mat = C.copy().reshape((n1, n2+1))
            
            this_step_cost = self.get_total_assignment_cost(j, C_mat, chosen_cols)

            self.cost.append(this_step_cost)
            self.order.append(k)

            j += 1
            
            if j < self.N_scan and sum(self.cost) < self.best_cost:

                self.search(j)

            if j == self.N_scan and sum(self.cost) < self.best_cost:

                fc = sum(self.cost)
                
                if fc < self.best_cost:

                    self.best_cost = fc
                    self.final_order = self.order.copy()

            j+= -1
            self.cost.pop()
            self.order.pop()