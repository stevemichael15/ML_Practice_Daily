import pickle
pickle.dump(model, open("model.pkl", "wb"))
#python object (model), its attributes and methods are coverted into bytes stream
model = pickle.load(open("model.pkl", "rb")) # to load the model again
print(model.predict(x_test))
