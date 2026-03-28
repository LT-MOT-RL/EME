import os
import sys
import numpy as np
import time
import cv2
import pickle
import math
import random
import faulthandler
faulthandler.enable()
from collections import OrderedDict
import torch.distributions.multivariate_normal as torchdist
from toolkits.dataset import LaSOT
from CSSM.utils import * 
from CSSM.metrics import * 
from CSSM.model_mb import social_stgcnn as CSSM_raw


otetrack_path = os.path.join(os.path.dirname(__file__), '..','OTETrack')
if otetrack_path not in sys.path:
    sys.path.append(otetrack_path)

unicorn_path = os.path.join(os.path.dirname(__file__), '..','Unicorn')
if unicorn_path not in sys.path:
    sys.path.append(unicorn_path)
unicorn_path = os.path.join(os.path.dirname(__file__), '..','Unicorn',"external_2")
if unicorn_path not in sys.path:
    sys.path.append(unicorn_path)
unicorn_path = os.path.join(os.path.dirname(__file__), '..','Unicorn',"exps","default")
if unicorn_path not in sys.path:
    sys.path.append(unicorn_path)


from OTETrack.lib.test.evaluation.tracker import Tracker as Local_Tracker
from Unicorn.external_2.lib.test.evaluation.tracker import Tracker as Global_Tracker
def _intersection(rects1, rects2):
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T

def rect_iou(rects1, rects2):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious



def c_e_a_r_c(x1, y1, w, h, Scale, W, H):
    x_min = x1 - w / 2
    y_min = y1 - h / 2
    x_max = x1 + w / 2
    y_max = y1 + h / 2
    clipped_x_min = max(W*(1-Scale), x_min) 
    clipped_y_min = max(H*(1-Scale), y_min)  
    clipped_x_max = min(W*Scale, x_max)  
    clipped_y_max = min(H*Scale, y_max)  
    clipped_w = max(0, clipped_x_max - clipped_x_min)
    clipped_h = max(0, clipped_y_max - clipped_y_min)
    effective_area = clipped_w * clipped_h
    original_area = w * h
    if original_area == 0:
        return 0
    effective_area_ratio = effective_area / original_area
    return effective_area_ratio








def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def compute_oov_risk(
    bbox,            # (x, y, w, h)
    image_size,      # (W, H)
    kappa=1.5,       # scale factor for adaptive s
    tau=5.0,         # sigmoid temperature
    big_thr=0.4      # large-target area ratio
):
    x, y, w, h = bbox
    W, H = image_size

    # ---- 1. adaptive edge width s_t ----
    s = kappa * 0.5 * math.sqrt(w * h)

    # print("s : ",s)
    # ---- 2. area-based risk (IoU with safe region) ----
    bx1, by1, bx2, by2 = x, y, x + w, y + h
    sx1, sy1, sx2, sy2 = s, s, W - s, H - s

    ix1 = max(bx1, sx1)
    iy1 = max(by1, sy1)
    ix2 = min(bx2, sx2)
    iy2 = min(by2, sy2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    bbox_area = max(1e-6, w * h)

    inside_ratio = inter_area / bbox_area
    p_area = 1.0 - inside_ratio

    # ---- 3. distance-based risk ----
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    d = min(cx, W - cx, cy, H - cy)
    p_dist = sigmoid((s - d) / tau)

    # ---- 4. large target handling ----

    extent_ratio = max(w / W, h / H)
    # print("bbox_area : ",bbox_area, "W * H : ",W * H," extent_ratio : ",extent_ratio)
    if extent_ratio >= big_thr:
        p_dist = 0.0

    # area_ratio = bbox_area / (W * H + 1e-9)
    # if area_ratio >= big_thr:
    #     p_dist = 0.0

    # ---- 5. fuse risks (Noisy-OR) ----
    p_oov = 1.0 - (1.0 - p_area) * (1.0 - p_dist)
    return min(1.0, max(0.0, p_oov))

def decide_invoke(p_oov, eta=0.6, eta_hard=0.85, stochastic=False):
    if not stochastic:
        return int(p_oov > eta)
    if p_oov >= eta_hard:
        return 1
    return 1 if random.random() < p_oov else 0








def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]
class LimitedStack:
    def __init__(self, max_size=8):
        self.stack = []
        self.max_size = max_size
    def push(self, item):
        if isinstance(item, list):
            if len(self.stack) >= self.max_size:
                self.stack.pop(0)  
            self.stack.append(item)
        else:
            raise ValueError("Only lists are allowed as elements in the stack.")
    def pop(self):
        if self.stack:
            return self.stack.pop()
        else:
            return None
    def size(self):
        return len(self.stack)
    def to_list(self):
        return self.stack

def get_local_tracker():
    local_tracker_name='otetrack'
    local_tracker_param='otetrack_256_full'
    local_dataset_name='lasot'
    local_runid=None
    test_checkpoint = './OTETrack/test_checkpoint/OTETrack_all.pth.tar'
    update_intervals =None
    update_threshold =None
    hanning_size =None
    pre_seq_number =None
    std_weight =None
    local_tracker_raw = Local_Tracker(local_tracker_name, local_tracker_param, local_dataset_name, local_runid,
                                      test_checkpoint,update_intervals,update_threshold,hanning_size,
                                      pre_seq_number,std_weight)
    params = local_tracker_raw.get_parameters()
    params.debug = 0
    return local_tracker_raw.create_tracker(params)

def get_global_tracker():
    global_tracker_name="unicorn_sot"
    global_tracker_param="unicorn_track_tiny_sot_only"
    global_dataset_name="trackingnet"
    global_run_id=None
    global_tracker_raw = Global_Tracker(global_tracker_name, global_tracker_param, global_dataset_name, global_run_id)
    params = global_tracker_raw.get_parameters()
    params.debug = 0
    return global_tracker_raw.create_tracker(params)


def _lasot_otb_record(record_file, boxes, times):
    # record bounding boxes
    record_dir = os.path.dirname(record_file)
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir)
    np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
    while not os.path.exists(record_file):
        print('warning: recording failed, retrying...')
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
    print('  Results recorded at', record_file)

    # record running times
    time_dir = os.path.join(record_dir, 'times')
    if not os.path.isdir(time_dir):
        os.makedirs(time_dir)
    time_file = os.path.join(time_dir, os.path.basename(
        record_file).replace('.txt', '_time.txt'))
    np.savetxt(time_file, times, fmt='%.8f')




def _lasot_otb_record_iou_scores(record_file, ious):
    # record bounding boxes
    ious_file = record_file.replace('.txt', '_iou_score.txt')

    np.savetxt(ious_file, ious, fmt='%.8f', delimiter=',')

def stabilize_cov(cov, base_eps=1e-6):
    # cov: [T,2,2]
    trace = cov[..., 0, 0] + cov[..., 1, 1]
    eps = base_eps * trace.unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(2, device=cov.device)
    return cov + eps * I



img_num = 0
mFps =0.0
oov_num =0
dataset = LaSOT(root_dir="datasets/LaSOT", subset="test")
len_dataset = len(dataset)
print("len_dataset : ",len_dataset)
local_tracker  = get_local_tracker()
global_tracker = get_global_tracker()

print("global_tracker running")
KSTEPS=20
print("*"*50)
print('Number of samples:',KSTEPS)
print("*"*50)
ade_ls = [] 
fde_ls = [] 
exp_path='./Sot_STGCNN/checkpoint_mamba/sot-mamba-lasot_1_300_nr_6_4'
print("*"*50)
print("Evaluating model:",exp_path)
model_path = exp_path+'/val_best.pth'
args_path = exp_path+'/args.pkl'
with open(args_path,'rb') as f: 
    args = pickle.load(f)
#Data prep     
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
#Defining the model 
model = CSSM_raw(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
output_feat=args.output_size,seq_len=args.obs_seq_len,
kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()




def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


total, trainable = count_params(model)











ckpt = torch.load(model_path)
model.load_state_dict(ckpt)
device = torch.device('cuda:0')

model.to(device)
model.eval()




T1 = args.obs_seq_len #6
T2 = args.pred_seq_len #4
for idx in range(len_dataset):
    img_files, anno = dataset[idx]
    seq_name = dataset.seq_names[idx]
    print("idx :",idx," seq_name : ",seq_name)
    frame_num = len(img_files)
    print("frame_num : ",frame_num)
    boxes = np.zeros((frame_num, 4))
    iou= np.zeros(frame_num)
    scores= np.zeros(frame_num)
    iou_scores= np.zeros((frame_num, 2))
    times= np.zeros(frame_num)
    boxes[0] = anno[0, :]
    iou[0]=1.0
    record_file = os.path.join("results/LaSOT", "Etp_ot", '%s.txt' % seq_name)
    record_dir =  os.path.join("results/LaSOT", "Etp_ot", seq_name)




    if os.path.exists(record_dir) is False:
        os.makedirs(record_dir)

    history_trajectories_1 =LimitedStack(max_size=6)
    history_trajectories_2 =LimitedStack(max_size=6)

    pred_traj =None
    first_flag = True
    edge_flag = False
    ofv_flag = False
    edge_free = 0
    use_global = False
    score_sum=0.0
    score_num=0
    score_list=[]
    for frame in range(0,frame_num):
        image = cv2.imread(img_files[frame])#, cv2.IMREAD_COLOR)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        img_num +=1
        if frame == 0 :
            linit_info = {}
            linit_info['init_bbox'] = boxes[frame]
            x, y, w, h = boxes[frame]
            out = local_tracker.initialize(image, linit_info)
            times[frame] = time.time() - start_time
            if out is None:
                out = {}
            prev_output = OrderedDict(out)

            ginit_info = {}
            ginit_info['init_bbox'] = boxes[frame]
            gout = global_tracker.initialize(image, ginit_info)

            if w<10:
                w=10
            if h<10:
                h=10
            HH,W,C = image.shape
            x1 = x + w/2
            y1 = y + h/2
            x2 = w
            y2 = h


            x1 = (x + w/2)*64/W-32
            y1 = (y + h/2)*48/HH-24
            x2 = w*64/W-32
            y2 = h*48/HH-24
            history_trajectories_1.push([x1, y1, 1])
            history_trajectories_2.push([x2, y2, 1])
 
        else:
            linit_info = {}
            linit_info['previous_output'] = prev_output
            linit_info['gt_bbox'] = boxes[frame]

            out = local_tracker.track(image, frame, info=linit_info)


            x, y, w, h = out["target_bbox"]
            box = out["target_bbox"]
            local_score = out["conf_score"]

            if w<10:
                w=10
            if h<10:
                h=10


            x1 = (x + w/2)*64/W-32
            y1 = (y + h/2)*48/HH-24
            x2 = w*64/W-32
            y2 = h*48/HH-24



            history_trajectories_1.push([x1, y1, 1])
            history_trajectories_2.push([x2, y2, 1])




            e_a_r = c_e_a_r_c(x1, y1, w, h, 0.95, W, HH)
            p_oov = compute_oov_risk(box,(W,HH))
            if_sot = decide_invoke(p_oov)


            if  if_sot == 1  and ofv_flag == False and edge_free == 0:
                edge_flag = True
            else: 
                edge_flag = False
            if edge_free >0:
                edge_free = edge_free-1


            
            edge_flag = True

            if history_trajectories_2.size() == T1:
                if edge_flag == True:
                    with torch.no_grad():
                        oov_time = time.time()
                
                        print(seq_name," ",frame, "ready to traj", "if_sot : ",if_sot,"e_a_r: ",e_a_r," ofv_flag : ",ofv_flag,"edge_free : ",edge_free)

                        pixel_pos=np.array([history_trajectories_1.to_list(),history_trajectories_2.to_list()])
    
                        a_p_j = pixel_pos[:,:,:2]



                        h_p_j = a_p_j.transpose(1,0,2)

                        a_p_j=a_p_j.transpose(0,2,1)

                        ra_p_j = np.zeros(a_p_j.shape)
                        ra_p_j[:,:, 1:] = a_p_j[:,:, 1:] - a_p_j[:,:, :-1]

                        a_p_j = torch.from_numpy(a_p_j).type(torch.float)
                        ra_p_j = torch.from_numpy(ra_p_j).type(torch.float)
                        v_, a_ = seq_to_graph(a_p_j,ra_p_j,True)

                        V_obs = v_.unsqueeze(dim=0)
                        V_obs_tmp = V_obs.permute(0,3,1,2)
                        obs_traj = a_p_j.unsqueeze(dim=0)
                        obs_traj_rel= ra_p_j.unsqueeze(dim=0)
                        V_obs_tmp= V_obs_tmp.to(device)
                        a_ = a_.to(device)

                        model_time = time.time()
                        V_pred, _ = model(V_obs_tmp,a_.squeeze())

                        # print("V_pred : ",V_pred.shape)


                        V_pred = V_pred.permute(0,2,3,1)
                        V_pred = V_pred.squeeze()
                        num_of_objs = obs_traj_rel.shape[1] #2
                        V_pred =  V_pred[:,:num_of_objs,:]

                        # print("V_pred : ",V_pred.shape)

                        sx = torch.exp(V_pred[:,:,2]) #sx
                        sy = torch.exp(V_pred[:,:,3]) #sy
                        corr = torch.tanh(V_pred[:,:,4]) #corr
                        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
                        cov[:,:,0,0]= sx*sx
                        cov[:,:,0,1]= corr*sx*sy
                        cov[:,:,1,0]= corr*sx*sy
                        cov[:,:,1,1]= sy*sy
                        mean = V_pred[:,:,0:2]



                        cov = 0.5 * (cov + cov.transpose(-1, -2))
                        cov = cov + 1e-6 * torch.eye(2, device=cov.device)

                        try:
                            L = torch.linalg.cholesky(cov)
                        except RuntimeError:
                            L = torch.diag_embed(torch.sqrt(torch.clamp(
                                torch.diagonal(cov, dim1=-2, dim2=-1), min=1e-6
                            )))


                        z = torch.randn(KSTEPS, *mean.shape, device=mean.device)  # [K,T,V,2]
                        samples = mean.unsqueeze(0) + torch.einsum(
                            "tvij,ktvj->ktvi", L, z
                        )  # [K,T,V,2]

                        V_pred = samples.mean(dim=0)




                        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())


                        pixel_pos_ = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                                V_x[-1,:,:].copy())
                        



                        point_num = pixel_pos_.shape[0]
                        peds_num = pixel_pos_.shape[1]
                        pred_traj = pixel_pos_[:,0,:].tolist()
                        pred_wh = pixel_pos_[:,1,:].tolist()

                        for pdx in range(T2):
                            x_c = pred_traj[pdx][0]
                            y_c = pred_traj[pdx][1]
                            p_w = pred_wh[pdx][0]
                            p_h = pred_wh[pdx][1]

                            x_c = (x_c + 32)*W/64
                            y_c = (y_c + 24)*HH/48
                            p_w = (p_w + 32)*W/64
                            p_h = (p_h + 24)*HH/48

                            if p_w <= 0 or p_h <= 0: 
                                p_w = w
                                p_h = h
                            e_a_r = c_e_a_r_c(x_c, y_c, p_w, p_h, 1, W, HH)
                            if e_a_r < 0.99 :
                                ofv_flag = True
                                break


                    oov_num += 1

                    if ofv_flag == False:
                        edge_free = T2


            box = clip_box(box, HH, W, margin=10)
            local_tracker.state = box
            box =  np.array(box)

            if ofv_flag == True :
                gout= global_tracker.track(image, info=None)


                global_score = gout["conf_score"]
                global_box = np.array(gout["target_bbox"])


                local_global_iou = rect_iou(box, global_box)

                if use_global == True and local_global_iou >= 0.5 :
                    ofv_flag = False
                    edge_free = 1
                    use_global = False
                elif global_score > 0.6:
                    if local_global_iou <= 0.1 :

                        use_global = True
                        box = global_box
                        local_tracker.state = box
                        x, y, w, h = box
                        x1 = x + w/2
                        y1 = y + h/2
                        x2 = w
                        y2 = h
                        history_trajectories_1.pop()
                        history_trajectories_2.pop()
                        history_trajectories_1.push([x1, y1, 1])
                        history_trajectories_2.push([x2, y2, 1])
                    else:
                        ofv_flag = False
                        use_global = False
                        edge_free = 1

            out["target_bbox"] = box
            prev_output = OrderedDict(out)
            boxes[frame, :]=out["target_bbox"]
            times[frame] = time.time() - start_time
            # scores[frame] = out["conf_score"]
            iou[frame] = rect_iou(boxes[frame, :], anno[frame,:])
            iou_scores[frame] =np.array([iou[frame], scores[frame]])
    print('FPS: {}'.format(len(times)/sum(times)))
    mFps+=(len(times)/sum(times))
    _lasot_otb_record(record_file, boxes, times)
    _lasot_otb_record_iou_scores(record_file, iou_scores)


print("mFps : ", mFps)

