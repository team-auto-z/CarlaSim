from tut2 import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2



class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,64,3,2,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,2,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,2,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,2,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,2,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,2,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,2,1),
            nn.ReLU(),
        )
        self.lin = nn.Linear(64*7*5,2)


    def forward(self,x):
        return F.softmax(self.lin(self.model(x).view(1,-1)),dim=1)




def  main():
    try:
        car = Car()
        
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        model = Model().to(device)



        for i in range(10000000000000):

            # frame = car.frame_buff.copy()


            # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # print(car.fra.shape)
            # print(model(torch.from_numpy(car.frame_buff).unsqueeze(0)))

            cv2.imshow('frame_bonnet',car.frame_buff)
            torch_tensor = torch.from_numpy(car.frame_buff)
            
            torch_tensor = torch_tensor.permute(2,1,0)
            
            torch_tensor = torch_tensor.unsqueeze(0).float()
         
            print(model(torch_tensor.to(device)))

            cv2.waitKey(24)

    except:
        traceback.print_exc()
        cv2.destroyAllWindows()
        for actor in actor_list:
            actor.destroy()
        print("Cleaned Up!")

    finally:

        cv2.destroyAllWindows()
        for actor in actor_list:
            actor.destroy()
        print("Cleaned Up!")







if __name__ == '__main__':
    main()
    # print(Model()(torch.randn(1,3,800,600)))