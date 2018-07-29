from keras.models import load_model
import pandas as pd
import numpy as np

batch_size = 128
num_classes = 10

img_rows, img_cols = 28, 28

test = pd.read_csv("../data/test.csv")
test = test.values.reshape(test.values.shape[0], img_rows, img_cols, 1)


test = test.astype('float32')
test /= 255

model = load_model("my_model.h5")
results = model.predict(test, batch_size, verbose=0)

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)