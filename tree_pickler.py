import pickle # standard library

interview_header = ["level", "lang", "tweets", "phd"]
interview_tree = ['Attribute', 'level', ['Value', 'Junior', ['Attribute', 'phd', ['Value', 'yes', ['Leaf', 'False', 2, 5]], ['Value', 'no', ['Leaf', 'True', 3, 5]]]], ['Value', 'Mid', ['Leaf', 'True', 4, 14]], ['Value', 'Senior', ['Attribute', 'tweets', ['Value', 'yes', ['Leaf', 'True', 2, 5]], ['Value', 'no', ['Leaf', 'False', 3, 5]]]]]

# pickle (object serialization): saving a binary representation of an object
# to file for loading and using later
# example: saving a trained a model for inference/prediction later
# in another python process, possibly running on a diff machine (server)
# imagine you just trained an awesome MyRandomForestClassifier
# and now you need to save it for using in your web app on a server later
# de/unpickle (object deserialization): loading a binary representation of
# an object from a file into a python object in program memory
# example: a web app that loads the trained model up for inference/prediction
# requests from clients

# lets pickle header and tree (together)
packaged_obj = (interview_header, interview_tree)
outfile = open("tree.p", "wb")
pickle.dump(packaged_obj, outfile)
outfile.close()