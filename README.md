# MiniPlaces
 
[Miniplaces](https://github.com/CSAILVision/miniplaces) is a scene recognition dataset developed by MIT. This dataset has 120K images from 100 scene categories. The categories are mutually exclusive. The dataset is split into 100K images for training, 10K images for validation, and 10K for testing.

The original image resolution for images in MiniPlaces is 128x128. To make the training feasible, the data loader reduces the image resolution to 32x32. dataloader.py will also download the full dataset the first time you run train_miniplaces.py.

predict.py asks the trained model to predict the following images:

<img src = "test_model.png" width = "1000">

These were the labels predicted by the model starting from top left:

palace <br>
bamboo_forest<br>
beauty_salon<br>
swimming_pool/outdoor<br>
lobby<br>
office <br>
canyon <br>
railroad_track  <br>
shed <br>
kitchen <br>
stage/indoor <br>
ski_slope<br> 
chalet <br>
valley <br>
auditorium <br>
lobby <br>
corridor <br>
trench <br>
valley <br>
desert/sand <br>
martial_arts_gym <br>
fire_station <br>
palace <br>
hot_spring <br>
swimming_pool/outdoor <br>
cockpit  <br>
shed <br>
fountain <br>
beauty_salon <br>
monastery/outdoor <br>
laundromat <br>
phone_booth<br>
