"""
to run pose estimation for one video
performance: ca 60 sec runtime, of which 2 seconds are the processing, file saving and also including the first frame, and the rest is ALL runtime required by the handle_one function for pose Estimation
For reducing the multiplier (different scales) from 4 to just one scale (resolution very low), the runtime for everything is around 8.5, just for handle one at 7 sec (slighly more than claimed in the paper)
"""


import time
from torch import np
import argparse
import pandas as pd
from os.path import isfile, join
from os import listdir
import os
import codecs, json
import tensorflow as tf

from Functions import *
import ast
import cv2
from test import test

parser = argparse.ArgumentParser(description='Pose Estimation Baseball')
parser.add_argument('input_file', metavar='DIR', # Video file to be processed
                    help='folder where merge.csv are')
parser.add_argument('output_folder', metavar='DIR', # output dir
                    help='folder where merge.csv are')
parser.add_argument('center',  #
                    help='specify what kind of file is used for specifying the center of the target person: either path_to_json_dictionary.json, or datPitcher, or datBatter')
restore_first_move = "/scratch/aa2528/pitch_type/saved_models/first_move_more"
restore_release = "/scratch/aa2528/pitch_type/saved_models/release_model"
restore_position = "/scratch/aa2528/pitch_type/saved_models/modelPosition"
sequ_len = 50


args = parser.parse_args()
inp_dir = args.input_file
out_dir = args.output_folder
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if args.center[-5:] == ".json":
    with open(args.center, "r") as infile:
        centers = json.load(infile)


for fi in listdir(inp_dir):
    if fi[0]=="." or fi[-4:]!=".mp4" or fi[:-4]+".json" in listdir(out_dir): 
    	print("already there or wrong ending", fi)
    	continue

    if args.center[:3] == "dat":
        target = args.center[3:]	
	try:	
            for i in open(inp_dir+fi[:-4]+".dat").readlines():
                datContent=ast.literal_eval(i)
        except IOError:
            print("dat file exists:", os.path.exists(inp_dir+fi[:-4]+".dat"))
            print("dat file not found", fi)
            continue
        bottom_b=datContent[target]['bottom']
        left_b=datContent[target]['left']
        right_b=datContent[target]['right']
        top_b=datContent[target]['top']
        print(bottom_b, left_b, right_b, top_b)
	center = np.array([abs(left_b-right_b)/2., abs(bottom_b-top_b)/2.])
    elif args.center[-5:] == ".json":
        try:
            center = centers[fi[:-4]] # np.array([abs(left_b-right_b)/2., abs(bottom_b-top_b)/2.])
        except KeyError:
            print("file does not exist in json file with center points")
            continue
        left_b = 0 # change for batters first move
        top_b = 0
    else:
        print("wrong input for center argument: must be either the file path of a json file containing a dictionary, or datPitcher or datBatter")
        continue
    ##############

    j=0
    #center_dic={}
    tic=time.time()
    f = inp_dir+fi
    print(f)
    video_capture = cv2.VideoCapture(f)

    tic1 = time.time()
    events_dic = {}
    events_dic["video_directory"]= inp_dir
    events_dic["bbox_batter"] = [0,0,0,0]
    events_dic["bbox_pitcher"] = [0,0,0,0]
    events_dic["start_time"]=time.time()
    rel = []


    # LOCALIZATION
    old_norm=10000
    indices = [6,9] # hips to find right person in the first frame
    p=0
    found = False
    pitcher_array = []


    while not found:
        #len(df[player][i])==0:
        ret, frame = video_capture.read()
        if args.center[-5:] == ".json":
            bottom_b = len(frame)
            right_b = len(frame[0])
##############
        canv, out = handle_one(frame[top_b:bottom_b, left_b:right_b]) # changed batter first move
#       out = handle_one(frame[top_b:bottom_b, left_b:right_b]) # changed batter first move
################
        for person in range(len(out)):
            hips=np.asarray(out[person])[indices]
            hips=hips[np.sum(hips,axis=1)!=0]
            if len(hips)==0:
                continue
            mean_hips=np.mean(hips,axis=0)
            norm= abs(mean_hips[0]-center[0])+abs(mean_hips[1]-center[1]) #6 hip
            if norm<old_norm:
                found = True
                loc=person
                old_norm=norm
        p+=1
        if not found:
            pitcher_array.append([[0,0] for j in range(18)])
            print("no person detected in frame", p)

    first_frame = np.array(out[loc])
    first_frame[:,0]+=left_b
    first_frame[first_frame[:,0]==left_b] = 0 # if the output was 0 (missing value), reverse box addition
    first_frame[:,1]+=top_b
    first_frame[first_frame[:,1]==top_b] = 0

    # Save first frame (detection on whole frame)
    pitcher_array.append(first_frame)

    # boundaries to prevent bounding box from being larger than the frame
    boundaries = [0, len(frame[0]), 0, len(frame)]
    print("boundaries = ", boundaries)

    # from first detection, form bbox for next fram
    bbox = define_bbox(first_frame, boundaries)

    # save detection to compare to next one
    globals()['old_array'] = first_frame #first_saved

    # save detection in a second array, in which the missing values are constantly filled with the last detection
    new_res = first_frame.copy()

    print("first output", pitcher_array[-1])

    # START LOOP OVER FRAMES
    t =0
    while t<5:
#    while True:
# Capture frame-by-frame
	print t
	t+=1

        ret, frame = video_capture.read()
        if frame is None:
            print("end of video capture")
            break
        pitcher = frame[bbox[2]:bbox[3], bbox[0]:bbox[1]]


#################################
        canv, out = handle_one(pitcher)
#       out = handle_one(pitcher)       
##########################################        
        out = np.array(out)
        out[:, :,0]+=bbox[0]
        out[out[:, :,0]==bbox[0]] = 0 # if the output was 0 (missing value), reverse box addition
        out[:, :,1]+=bbox[2]
        out[out[:, :,1]==bbox[2]] = 0
        out = player_localization(out, globals()['old_array'])

        out = np.array(out)
        
        if np.all(out==0):
            pitcher_array.append(np.array([[0,0] for j in range(18)]))
        else:
            # update previous detection
            globals()['old_array'] = out.copy()
            pitcher_array.append(out)
            # update bounding box: for missing values, take last detection of this joint (saved in new_res)
            new_res[np.where(out!=0)]=out[np.where(out!=0)]
            # print(new_res)
            bbox = define_bbox(new_res, boundaries)
            print("bbox new", bbox)

        print(p)
        p+=1


######################3
        cv2.imwrite(out_dir+str(t)+".jpg",canv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
########################


    pitcher_array = np.array(pitcher_array)
    print("shape pitcher_array", pitcher_array.shape)

    ## for interpolation, mix right left and smoothing:
    pitcher_array = df_coordinates(pitcher_array, do_interpolate = False, smooth = False, fps = 20)

    # SAVE IN JSON FORMAT
    game_id = fi[:-4]
    file_path_pitcher = out_dir+game_id
    print(events_dic, file_path_pitcher)
    to_json(pitcher_array, events_dic, file_path_pitcher)

    ### batter first move
    # with open(out_dir+game_id+"_video.json", "w") as outfile:
    #      json.dump(videos, outfile)

    toctoc=time.time()
    print("Time for whole video to array: ", toctoc-tic)

