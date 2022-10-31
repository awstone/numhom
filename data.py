# This script provides an API to interface with PETSc binary matrices

from petsc4py import PETSc as petsc 

class Data:
    @staticmethod
    def load(path):
        viewer = petsc.Viewer().createBinary(path, 'r')
        return petsc.Mat().load(viewer)

    def __init__(self, af_pth, ac_pth, pf_pth):
        self.af = Data.load(af_pth)
        self.ac = Data.load(ac_pth)
        self.pf = Data.load(pf_pth)

# test data class
def main():
    data = Data('af.mat', 'ac.mat', 'pf.mat')
    print(data.af, data.ac, data.pf)
    
if __name__ == '__main__':
    main()
    

        
    
