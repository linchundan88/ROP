import keras

model_file = '/home/ubuntu/dlp/deploy_models/STAGE/2019_10_10/1/InceptionV3-003-0.994.hdf5'

model1 = keras.models.load_model(model_file, compile=False)

# model1.summary()
# from keras.utils.vis_utils import plot_model
# plot_model(model1, to_file='inception_v3.png', show_shapes=True)

for index, layer1 in enumerate(model1.layers):
    if 'mixed' in model1.layers[86].name:
        pass

    if isinstance(layer1, keras.layers.Concatenate):
        print(index, layer1.name, '\n')

print('OK')


'''
40 mixed0 

63 mixed1 

86 mixed2 

100 mixed3 

132 mixed4 

164 mixed5 

196 mixed6 

228 mixed7 

248 mixed8 

276 mixed9_0 

277 concatenate_1 

279 mixed9 

307 mixed9_1 

308 concatenate_2 

310 mixed10 

'''

