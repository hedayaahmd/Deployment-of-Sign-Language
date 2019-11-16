import os
import json
pathdict={"files":
 "C:/Users/USER/Desktop/Deploy Sign Language/packages/CNNModel/CNNModel/datasets/test_data/ngry/angry_14.avi"
	 }



PWD = os.path.dirname(os.path.abspath(__file__))
file1=os.path.join(PWD,"input_test.json")

print(PWD)
print(file1)
with open(file1,'w') as fp:
    json.dump(pathdict, fp)
