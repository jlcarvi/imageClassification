# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:41:52 2019

@author: jlcar
Augment images
"""
import Augmentor

#try:
    
p = Augmentor.Pipeline("cards/originalData/ten_of_heards")
p.rotate90(probability=0.2)
p.rotate270(probability=0.01)
p.flip_left_right(probability=0.6)
p.flip_top_bottom(probability=0.3)
p.crop_random(probability=1.0, percentage_area=0.9)
p.random_distortion(probability=0.8, grid_width=3, grid_height=3, magnitude=8)
p.resize(probability=1.0, width=120, height=120)
p.sample(600)
print("Aumented Ready, check output folder ")
#except Exception as e: 
#    print(e)
#    print("Something wrong")    
    