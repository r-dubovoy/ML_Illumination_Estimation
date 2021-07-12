# Illumination Estimation using ML



In this work I explore the possibility to estimate the light conditions that illuminate user’s face. Reproducing similar light conditions in virtual object environment would help AR objects (filter headwear, masks, etc.) blend into reality even better. In order to achieve results techniques of data generation, information extraction, and ML development cycle were performed.

**Full Thesis** - https://drive.google.com/file/d/14WtU_lZZ7Q6JrVSmvubjujJeql9YGYfI/view?usp=sharing

**Only Code included**. Ml Models, Blender Scenes and .obj files can be requested by writing to dub.dubovyi@gmail.com**


## Abstract

This research proposes another way of determining light source position using
Machine Learning. In the century that promises rise and rapid growth of Artificial
Intelligence (AI) we can expect that computational capabilities of our machines will enable
us to embed such AI-driven solutions in almost any software and/or hardware solution.
Another promising trend in technologies is Augmented Reality (AR). Today we
already can see that our mobile devices with AR on board can deliver significant amount of
user-driven content. The good examples that generate big revenues and have millions of
daily users are AR filters that allow users to place a virtual object on top of the face. Big
players in this niche are Snapchat and Instagram, I think that everyone smiled at least once
when using one of their filters.
Following the trends, I took the liberty of exploring the possibility of illuminating
virtual object depending on user’s face illumination. The idea is to let robust and compact
Neural Network decide about how to cast light on virtual component while being efficient
enough to be deployed on mobile device. Having limited resources this thesis explores only
the tip of the great potential of such technology.

## TL;DR 

Face is illuminated with random light. Hat is illuminated with estimated light based on image of the face.

![s](https://i.ibb.co/vc07V9q/Showcase2.png)
